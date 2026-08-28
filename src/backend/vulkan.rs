use std::{
    ffi::{CStr, c_char},
    io::Cursor,
    ptr::NonNull,
    slice,
};

use anyhow::{Context, Result, bail, ensure};
use ash::{Device, Entry, Instance, vk};
use secp256k1::{All, Secp256k1, SecretKey};
use zeroize::{Zeroize, Zeroizing};

use super::{Address, AddressBackend, cpu, table};

const WINDOW_BITS: u8 = 16;
const CHUNK_SIZE: u32 = 8;
const WORKGROUP_SIZE: u32 = 64;
const INFLIGHT: usize = 2;
const AMD_VENDOR_ID: u32 = 0x1002;

struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    mapped: Option<NonNull<u8>>,
    sensitive: bool,
}

// SAFETY: the backend never shares mapped pointers. CPU access and GPU
// submission stay serialized on the owning thread after a move.
unsafe impl Send for GpuBuffer {}

impl GpuBuffer {
    fn empty() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 0,
            mapped: None,
            sensitive: false,
        }
    }

    #[cfg(test)]
    fn as_bytes(&self, length: usize) -> &[u8] {
        let mapped = self.mapped.expect("host-visible buffer is not mapped");
        assert!(length as vk::DeviceSize <= self.size);
        // SAFETY: mapping is exclusive to this backend and bounds-checked.
        unsafe { slice::from_raw_parts(mapped.as_ptr(), length) }
    }

    fn as_mut_bytes(&mut self, length: usize) -> &mut [u8] {
        let mapped = self.mapped.expect("host-visible buffer is not mapped");
        assert!(length as vk::DeviceSize <= self.size);
        // SAFETY: mapping is exclusive to this backend, bounds-checked, and GPU
        // work on this allocation has completed or not yet been submitted.
        unsafe { slice::from_raw_parts_mut(mapped.as_ptr(), length) }
    }

    fn write(&mut self, bytes: &[u8]) {
        self.as_mut_bytes(bytes.len()).copy_from_slice(bytes);
    }

    fn read(&self, offset: usize, bytes: &mut [u8]) {
        let mapped = self.mapped.expect("host-visible buffer is not mapped");
        assert!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end as vk::DeviceSize <= self.size)
        );
        // SAFETY: caller waits for the GPU before reading address output.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr().add(offset),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
    }

    fn clear(&mut self) {
        if self.mapped.is_some() {
            self.as_mut_bytes(self.size as usize).zeroize();
        }
    }
}

struct GpuSlot {
    keys: GpuBuffer,
    addresses: GpuBuffer,
    descriptor_set: vk::DescriptorSet,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    submitted: bool,
}

pub(crate) struct VulkanBackend {
    _entry: Entry,
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    table: GpuBuffer,
    slots: Vec<GpuSlot>,
    collect_at: usize,
    pending: usize,
    capacity: usize,
    sample_index: usize,
    verifier: Secp256k1<All>,
    device_name: String,
}

impl VulkanBackend {
    pub(crate) fn new(capacity: usize) -> Result<Option<Self>> {
        if cfg!(target_os = "macos") {
            return Ok(None);
        }
        ensure!(
            (1..=super::MAX_GPU_BATCH_SIZE as usize).contains(&capacity),
            "invalid Vulkan batch capacity"
        );
        let entry = match unsafe { Entry::load() } {
            Ok(entry) => entry,
            Err(_) => return Ok(None),
        };
        let instance = match create_instance(&entry) {
            Ok(instance) => instance,
            Err(_) => return Ok(None),
        };
        let Some(picked) = pick_device(&instance)? else {
            unsafe { instance.destroy_instance(None) };
            return Ok(None);
        };
        create_backend(entry, instance, picked, capacity).and_then(|mut backend| {
            backend
                .self_test()
                .context("Vulkan startup self-test failed")?;
            Ok(Some(backend))
        })
    }

    pub(crate) fn device_name(&self) -> String {
        self.device_name.clone()
    }

    fn self_test(&mut self) -> Result<()> {
        let mut one = [0; 32];
        one[31] = 1;
        let mut two = one;
        two[31] = 2;
        let mut last = secp256k1::constants::CURVE_ORDER;
        last[31] -= 1;
        let keys: Vec<_> = [one, two, last]
            .into_iter()
            .map(SecretKey::from_byte_array)
            .collect::<std::result::Result<_, _>>()?;
        for chunk in keys.chunks(self.capacity) {
            let mut addresses = vec![[0; 20]; chunk.len()];
            self.derive_batch(chunk, &mut addresses)?;
            for (key, address) in chunk.iter().zip(&addresses) {
                cpu::verify_address(key, address, &self.verifier)?;
            }
        }
        self.sample_index = 0;
        Ok(())
    }

    fn write_keys(buffer: &mut GpuBuffer, keys: &[SecretKey]) {
        let bytes = keys.len() * 32;
        let destination = buffer.as_mut_bytes(bytes);
        for (slot, key) in destination.chunks_exact_mut(32).zip(keys) {
            let secret = Zeroizing::new(key.secret_bytes());
            slot.copy_from_slice(secret.as_ref());
        }
    }

    fn wait_slot(&mut self, index: usize) -> Result<()> {
        let slot = &mut self.slots[index];
        if !slot.submitted {
            return Ok(());
        }
        unsafe {
            self.device
                .wait_for_fences(&[slot.fence], true, u64::MAX)
                .context("Vulkan fence wait failed")?;
        }
        slot.submitted = false;
        Ok(())
    }
}

impl AddressBackend for VulkanBackend {
    fn inflight_capacity(&self) -> usize {
        self.slots.len()
    }

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        self.begin_batch(keys)?;
        self.end_batch(keys, addresses)
    }

    fn begin_batch(&mut self, keys: &[SecretKey]) -> Result<()> {
        ensure!(keys.len() <= self.capacity, "batch exceeds Vulkan capacity");
        ensure!(
            self.pending < self.slots.len(),
            "Vulkan in-flight slots exhausted"
        );
        if keys.is_empty() {
            return Ok(());
        }
        let submit_at = (self.collect_at + self.pending) % self.slots.len();
        ensure!(
            !self.slots[submit_at].submitted,
            "Vulkan slot still holds an in-flight command"
        );
        Self::write_keys(&mut self.slots[submit_at].keys, keys);
        let threads = (keys.len() as u32).div_ceil(CHUNK_SIZE);
        let groups = threads.div_ceil(WORKGROUP_SIZE);
        let count = keys.len() as u32;
        let slot = &self.slots[submit_at];
        unsafe {
            self.device
                .reset_fences(&[slot.fence])
                .context("Vulkan fence reset failed")?;
            self.device
                .reset_command_buffer(slot.command_buffer, vk::CommandBufferResetFlags::empty())
                .context("Vulkan command reset failed")?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(slot.command_buffer, &begin)
                .context("Vulkan command begin failed")?;
            self.device.cmd_bind_pipeline(
                slot.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                slot.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[slot.descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                slot.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &count.to_le_bytes(),
            );
            self.device.cmd_dispatch(slot.command_buffer, groups, 1, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                slot.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.device
                .end_command_buffer(slot.command_buffer)
                .context("Vulkan command end failed")?;
            let buffers = [slot.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&buffers);
            self.device
                .queue_submit(self.queue, &[submit], slot.fence)
                .context("Vulkan queue submit failed")?;
        }
        self.slots[submit_at].submitted = true;
        self.pending += 1;
        Ok(())
    }

    fn end_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        ensure!(
            keys.len() == addresses.len(),
            "batch input/output lengths differ"
        );
        if keys.is_empty() {
            return Ok(());
        }
        ensure!(self.pending > 0, "no in-flight Vulkan batch to collect");
        let collect_at = self.collect_at;
        self.wait_slot(collect_at)?;
        let slot = &mut self.slots[collect_at];
        for (index, address) in addresses.iter_mut().enumerate() {
            slot.addresses.read(index * 20, address);
        }
        let sample = self.sample_index % keys.len();
        cpu::verify_address(&keys[sample], &addresses[sample], &self.verifier)?;
        self.sample_index = self.sample_index.wrapping_add(1);
        slot.keys.clear();
        self.collect_at = (collect_at + 1) % self.slots.len();
        self.pending -= 1;
        Ok(())
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for slot in &mut self.slots {
                if slot.submitted {
                    let _ = self.device.wait_for_fences(&[slot.fence], true, u64::MAX);
                    slot.submitted = false;
                }
                slot.keys.clear();
                if slot.fence != vk::Fence::null() {
                    self.device.destroy_fence(slot.fence, None);
                }
                destroy_buffer(&self.device, &mut slot.keys);
                destroy_buffer(&self.device, &mut slot.addresses);
            }
            destroy_buffer(&self.device, &mut self.table);
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.shader != vk::ShaderModule::null() {
                self.device.destroy_shader_module(self.shader, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct PickedDevice {
    physical: vk::PhysicalDevice,
    queue_family: u32,
    name: String,
}

fn create_instance(entry: &Entry) -> Result<Instance> {
    let app_info = vk::ApplicationInfo {
        p_application_name: c"vanity-rs".as_ptr(),
        application_version: vk::make_api_version(0, 0, 1, 0),
        api_version: vk::API_VERSION_1_1,
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    unsafe { entry.create_instance(&create_info, None) }.context("Vulkan instance creation failed")
}

fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    let raw = properties.device_name.as_ptr().cast::<c_char>();
    unsafe { CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned()
}

fn pick_device(instance: &Instance) -> Result<Option<PickedDevice>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .context("Vulkan physical device enumeration failed")?;
    let table_bytes = table::table_bytes(WINDOW_BITS) as u32;
    let mut best: Option<(u8, u8, PickedDevice)> = None;
    for physical in physical_devices {
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let features = unsafe { instance.get_physical_device_features(physical) };
        let type_score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 2,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => continue,
        };
        if features.shader_int64 != vk::TRUE {
            continue;
        }
        if properties.limits.max_storage_buffer_range < table_bytes {
            continue;
        }
        if properties.limits.max_compute_work_group_size[0] < WORKGROUP_SIZE
            || properties.limits.max_compute_work_group_invocations < WORKGROUP_SIZE
            || properties.limits.max_push_constants_size < 4
        {
            continue;
        }
        let Some(queue_family) = compute_queue_family(instance, physical) else {
            continue;
        };
        let amd = u8::from(properties.vendor_id == AMD_VENDOR_ID);
        let candidate = PickedDevice {
            physical,
            queue_family,
            name: device_name(&properties),
        };
        match &best {
            Some((best_amd, best_type, _)) if (*best_amd, *best_type) >= (amd, type_score) => {}
            _ => best = Some((amd, type_score, candidate)),
        }
    }
    Ok(best.map(|(_, _, device)| device))
}

fn compute_queue_family(instance: &Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    unsafe { instance.get_physical_device_queue_family_properties(physical) }
        .iter()
        .enumerate()
        .find_map(|(index, properties)| {
            properties
                .queue_flags
                .contains(vk::QueueFlags::COMPUTE)
                .then_some(index as u32)
        })
}

fn create_backend(
    entry: Entry,
    instance: Instance,
    picked: PickedDevice,
    capacity: usize,
) -> Result<VulkanBackend> {
    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(picked.queue_family)
        .queue_priorities(&queue_priorities);
    let features = vk::PhysicalDeviceFeatures {
        shader_int64: vk::TRUE,
        ..Default::default()
    };
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_features(&features);
    let device = unsafe { instance.create_device(picked.physical, &device_info, None) }
        .context("Vulkan device creation failed")?;
    let queue = unsafe { device.get_device_queue(picked.queue_family, 0) };
    let mut backend = VulkanBackend {
        _entry: entry,
        instance,
        device,
        queue,
        command_pool: vk::CommandPool::null(),
        descriptor_pool: vk::DescriptorPool::null(),
        descriptor_layout: vk::DescriptorSetLayout::null(),
        pipeline_layout: vk::PipelineLayout::null(),
        pipeline: vk::Pipeline::null(),
        shader: vk::ShaderModule::null(),
        table: GpuBuffer::empty(),
        slots: Vec::new(),
        collect_at: 0,
        pending: 0,
        capacity,
        sample_index: 0,
        verifier: Secp256k1::new(),
        device_name: picked.name,
    };
    let pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(picked.queue_family);
    backend.command_pool = unsafe { backend.device.create_command_pool(&pool_info, None) }
        .context("Vulkan command pool creation failed")?;

    let bindings = [
        storage_binding(0, vk::ShaderStageFlags::COMPUTE),
        storage_binding(1, vk::ShaderStageFlags::COMPUTE),
        storage_binding(2, vk::ShaderStageFlags::COMPUTE),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    backend.descriptor_layout = unsafe {
        backend
            .device
            .create_descriptor_set_layout(&layout_info, None)
    }
    .context("Vulkan descriptor layout creation failed")?;

    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: (INFLIGHT * 3) as u32,
    };
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(INFLIGHT as u32)
        .pool_sizes(std::slice::from_ref(&pool_size));
    backend.descriptor_pool = unsafe {
        backend
            .device
            .create_descriptor_pool(&descriptor_pool_info, None)
    }
    .context("Vulkan descriptor pool creation failed")?;

    let layouts = [backend.descriptor_layout; INFLIGHT];
    let alloc_sets = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(backend.descriptor_pool)
        .set_layouts(&layouts);
    let descriptor_sets = unsafe { backend.device.allocate_descriptor_sets(&alloc_sets) }
        .context("Vulkan descriptor set allocation failed")?;

    let push_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: 4,
    };
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&backend.descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));
    backend.pipeline_layout = unsafe {
        backend
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    }
    .context("Vulkan pipeline layout creation failed")?;

    let spirv = ash::util::read_spv(&mut Cursor::new(include_bytes!("shader.spv").as_slice()))
        .context("invalid embedded Vulkan SPIR-V")?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv);
    backend.shader = unsafe { backend.device.create_shader_module(&shader_info, None) }
        .context("Vulkan shader module creation failed")?;

    let entry_name = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(backend.shader)
        .name(entry_name);
    let compute_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(backend.pipeline_layout);
    let pipelines = unsafe {
        backend.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&compute_info),
            None,
        )
    }
    .map_err(|(_, error)| error)
    .context("Vulkan compute pipeline creation failed")?;
    backend.pipeline = pipelines[0];

    let memory_props = unsafe {
        backend
            .instance
            .get_physical_device_memory_properties(picked.physical)
    };
    backend.table = create_buffer(
        &backend.device,
        &memory_props,
        table::table_bytes(WINDOW_BITS) as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        false,
        false,
    )?;
    upload_table(
        &backend.device,
        backend.queue,
        backend.command_pool,
        &memory_props,
        &backend.table,
        &backend.verifier,
    )?;

    let command_alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(backend.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(INFLIGHT as u32);
    let command_buffers = unsafe { backend.device.allocate_command_buffers(&command_alloc) }
        .context("Vulkan command buffer allocation failed")?;

    let key_bytes = (capacity * 32) as vk::DeviceSize;
    let address_bytes = (capacity * 20) as vk::DeviceSize;
    for (index, command_buffer) in command_buffers.into_iter().enumerate() {
        let keys = create_host_buffer(
            &backend.device,
            &memory_props,
            key_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
        )?;
        let addresses = create_host_buffer(
            &backend.device,
            &memory_props,
            address_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )?;
        let descriptor_set = descriptor_sets[index];
        write_slot_descriptors(
            &backend.device,
            descriptor_set,
            &keys,
            &backend.table,
            &addresses,
        );
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { backend.device.create_fence(&fence_info, None) }
            .context("Vulkan fence creation failed")?;
        backend.slots.push(GpuSlot {
            keys,
            addresses,
            descriptor_set,
            command_buffer,
            fence,
            submitted: false,
        });
    }
    Ok(backend)
}

fn storage_binding(
    binding: u32,
    stage: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding {
        binding,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: stage,
        ..Default::default()
    }
}

fn write_slot_descriptors(
    device: &Device,
    set: vk::DescriptorSet,
    keys: &GpuBuffer,
    table: &GpuBuffer,
    addresses: &GpuBuffer,
) {
    let key_info = vk::DescriptorBufferInfo::default()
        .buffer(keys.buffer)
        .range(vk::WHOLE_SIZE);
    let table_info = vk::DescriptorBufferInfo::default()
        .buffer(table.buffer)
        .range(vk::WHOLE_SIZE);
    let address_info = vk::DescriptorBufferInfo::default()
        .buffer(addresses.buffer)
        .range(vk::WHOLE_SIZE);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&key_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&table_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&address_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn memory_type_index(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|&index| {
        (type_bits & (1 << index)) != 0
            && properties.memory_types[index as usize]
                .property_flags
                .contains(required)
    })
}

fn create_buffer(
    device: &Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory: vk::MemoryPropertyFlags,
    map: bool,
    sensitive: bool,
) -> Result<GpuBuffer> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .context("Vulkan buffer creation failed")?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let Some(memory_type) =
        memory_type_index(properties, requirements.memory_type_bits, required_memory)
    else {
        unsafe { device.destroy_buffer(buffer, None) };
        bail!("Vulkan memory type is unavailable");
    };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error).context("Vulkan memory allocation failed");
        }
    };
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(error).context("Vulkan bind memory failed");
    }
    let mapped = if map {
        match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
            Ok(pointer) => NonNull::new(pointer.cast()),
            Err(error) => {
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_buffer(buffer, None);
                }
                return Err(error).context("Vulkan memory map failed");
            }
        }
    } else {
        None
    };
    Ok(GpuBuffer {
        buffer,
        memory,
        size,
        mapped,
        sensitive,
    })
}

fn create_host_buffer(
    device: &Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    sensitive: bool,
) -> Result<GpuBuffer> {
    let rebar = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    match create_buffer(device, properties, size, usage, rebar, true, sensitive) {
        Ok(buffer) => Ok(buffer),
        Err(_) => create_buffer(device, properties, size, usage, host, true, sensitive),
    }
}

fn upload_table(
    device: &Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    properties: &vk::PhysicalDeviceMemoryProperties,
    table: &GpuBuffer,
    secp: &Secp256k1<All>,
) -> Result<()> {
    let bytes = table::build_table(secp, WINDOW_BITS)?;
    let mut staging = create_buffer(
        device,
        properties,
        table.size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        true,
        false,
    )?;
    staging.write(&bytes);
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command = unsafe { device.allocate_command_buffers(&alloc) }
        .context("Vulkan table copy command allocation failed")?[0];
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(command, &begin)
            .context("Vulkan table copy begin failed")?;
        let region = vk::BufferCopy::default().size(table.size);
        device.cmd_copy_buffer(command, staging.buffer, table.buffer, &[region]);
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .buffer(table.buffer)
            .size(vk::WHOLE_SIZE);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
        device
            .end_command_buffer(command)
            .context("Vulkan table copy end failed")?;
        let buffers = [command];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        device
            .queue_submit(queue, &[submit], vk::Fence::null())
            .context("Vulkan table copy submit failed")?;
        device
            .queue_wait_idle(queue)
            .context("Vulkan table copy wait failed")?;
        device.free_command_buffers(command_pool, &[command]);
    }
    destroy_buffer(device, &mut staging);
    Ok(())
}

fn destroy_buffer(device: &Device, buffer: &mut GpuBuffer) {
    if buffer.sensitive {
        buffer.clear();
    }
    unsafe {
        if buffer.mapped.take().is_some() {
            device.unmap_memory(buffer.memory);
        }
        if buffer.buffer != vk::Buffer::null() {
            device.destroy_buffer(buffer.buffer, None);
            buffer.buffer = vk::Buffer::null();
        }
        if buffer.memory != vk::DeviceMemory::null() {
            device.free_memory(buffer.memory, None);
            buffer.memory = vk::DeviceMemory::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn vulkan_is_unavailable_on_macos() {
        #[cfg(target_os = "macos")]
        {
            assert!(VulkanBackend::new(1024).unwrap().is_none());
        }
    }

    #[test]
    #[ignore = "requires a real Vulkan compute device; absence is a failure"]
    fn vulkan_differential() -> Result<()> {
        let mut backend = VulkanBackend::new(super::super::MAX_GPU_BATCH_SIZE as usize)?
            .context("GPU required for hardware acceptance")?;
        let mut rng = ChaCha20Rng::from_seed([93; 32]);
        let mut keys = Vec::with_capacity(backend.capacity);
        let mut last = secp256k1::constants::CURVE_ORDER;
        last[31] -= 1;
        keys.push(SecretKey::from_byte_array(last)?);
        for bit in 0..256 {
            let mut scalar = [0; 32];
            scalar[31 - bit / 8] |= 1 << (bit % 8);
            if let Ok(key) = SecretKey::from_byte_array(scalar) {
                keys.push(key);
            }
        }
        for _ in 0..64 {
            let mut scalar = [0; 32];
            rng.fill_bytes(&mut scalar);
            if let Ok(key) = SecretKey::from_byte_array(scalar) {
                keys.push(key);
            }
        }
        for chunk in keys.chunks(backend.capacity) {
            let mut addresses = vec![[0; 20]; chunk.len()];
            backend.derive_batch(chunk, &mut addresses)?;
            for (key, address) in chunk.iter().zip(&addresses) {
                cpu::verify_address(key, address, &backend.verifier)?;
            }
        }
        for count in [1usize, 7, 8, 9, 33, 65] {
            let batch: Vec<_> = keys.iter().copied().take(count).collect();
            let mut addresses = vec![[0; 20]; batch.len()];
            backend.derive_batch(&batch, &mut addresses)?;
            for (key, address) in batch.iter().zip(&addresses) {
                cpu::verify_address(key, address, &backend.verifier)?;
            }
        }
        inflight_overlap_differential(&mut backend, &keys[..66.min(keys.len())])?;
        input_is_cleared_after_batch(&mut backend)?;
        Ok(())
    }

    fn inflight_overlap_differential(
        backend: &mut VulkanBackend,
        keys: &[SecretKey],
    ) -> Result<()> {
        ensure!(keys.len() >= 2, "overlap test needs two keys");
        let mid = keys.len() / 2;
        let first = &keys[..mid];
        let second = &keys[mid..];
        ensure!(
            backend.inflight_capacity() >= 2,
            "expected two in-flight slots"
        );
        backend.begin_batch(first)?;
        backend.begin_batch(second)?;
        let mut first_out = vec![[0; 20]; first.len()];
        let mut second_out = vec![[0; 20]; second.len()];
        backend.end_batch(first, &mut first_out)?;
        backend.end_batch(second, &mut second_out)?;
        for (key, address) in first.iter().zip(&first_out) {
            cpu::verify_address(key, address, &backend.verifier)?;
        }
        for (key, address) in second.iter().zip(&second_out) {
            cpu::verify_address(key, address, &backend.verifier)?;
        }
        Ok(())
    }

    fn input_is_cleared_after_batch(backend: &mut VulkanBackend) -> Result<()> {
        // After end_batch, collect_at points at the next slot; the previous slot
        // is the one whose input must be wiped.
        let cleared = (backend.collect_at + backend.slots.len() - 1) % backend.slots.len();
        let size = backend.slots[cleared].keys.size as usize;
        let bytes = backend.slots[cleared].keys.as_bytes(size);
        ensure!(
            bytes.iter().all(|&byte| byte == 0),
            "Vulkan key input was not wiped"
        );
        Ok(())
    }
}
