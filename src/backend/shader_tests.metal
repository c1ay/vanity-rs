// Test-only entry points call the exact same arithmetic as the production kernel.
kernel void derive_public_keys(device const uchar *keys [[buffer(0)]],
                               device const uint *table [[buffer(1)]],
                               device uchar *output [[buffer(2)]],
                               constant uint &count [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    Point p = public_point(keys + gid*32, table);
    for (uint i = 0; i < 32; ++i) {
        output[gid*64+i] = coordinate_byte(p.x, i);
        output[gid*64+32+i] = coordinate_byte(p.y, i);
    }
}

kernel void field_operations(device const uint *input [[buffer(0)]],
                             device const uint *unused [[buffer(1)]],
                             device uint *output [[buffer(2)]],
                             constant uint &count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    Fe a, b;
    for (uint i = 0; i < 8; ++i) { a.v[i] = input[gid*16+i]; b.v[i] = input[gid*16+8+i]; }
    Fe values[5] = {fe_add(a,b), fe_sub(a,b), fe_mul(a,b), fe_square(a), fe_inverse(a)};
    for (uint op = 0; op < 5; ++op)
        for (uint i = 0; i < 8; ++i) output[gid*40+op*8+i] = values[op].v[i];
}
