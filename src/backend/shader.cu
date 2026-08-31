// Production path only: 16-bit fixed-base windows, fused per-thread chunked
// inversion, Ethereum Keccak-256. Layout matches shader.metal / shader.comp
// byte-for-byte for keys (32 raw secret bytes) and addresses (20 raw bytes).
//
// Offline compile (not invoked by cargo build):
//   nvcc -ptx -arch=compute_60 -o src/backend/shader.ptx src/backend/shader.cu

#include <cstdint>

#ifndef __forceinline__
#define __forceinline__ inline
#endif

#define WINDOW_BITS 16
#define CHUNK_SIZE 8
#ifndef INCREMENT_STRIDE
#define INCREMENT_STRIDE 32
#endif

struct Fe {
    uint32_t v[8];
};
struct Point {
    Fe x;
    Fe y;
    Fe z;
};

__device__ __constant__ uint32_t FIELD_P[8] = {0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
                                               0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};

static __device__ __forceinline__ Fe fe_zero() {
    Fe r;
    for (uint32_t i = 0; i < 8; ++i) r.v[i] = 0u;
    return r;
}

static __device__ __forceinline__ Fe fe_one() {
    Fe r = fe_zero();
    r.v[0] = 1u;
    return r;
}

static __device__ __forceinline__ uint32_t mask_if(bool b) { return uint32_t(-int32_t(b)); }

static __device__ __forceinline__ Fe fe_select(Fe a, Fe b, uint32_t mask) {
    Fe r;
    for (uint32_t i = 0; i < 8; ++i) r.v[i] = (a.v[i] & ~mask) | (b.v[i] & mask);
    return r;
}

static __device__ __forceinline__ uint32_t fe_zero_mask(Fe a) {
    uint32_t bits = 0u;
    for (uint32_t i = 0; i < 8; ++i) bits |= a.v[i];
    return mask_if(bits == 0u);
}

static __device__ __forceinline__ Fe fe_normalize(Fe a) {
    Fe d;
    uint64_t borrow = 0ull;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sub = uint64_t(FIELD_P[i]) + borrow;
        d.v[i] = a.v[i] - uint32_t(sub);
        borrow = uint64_t(uint64_t(a.v[i]) < sub);
    }
    return fe_select(d, a, mask_if(borrow != 0ull));
}

static __device__ __forceinline__ Fe fe_reduce(uint32_t t[16]) {
    uint32_t r[10];
    for (uint32_t i = 0; i < 10; ++i) r[i] = 0u;
    uint64_t carry = 0ull;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sum = uint64_t(t[i]) + uint64_t(t[i + 8]) * 977ull + carry;
        if (i != 0u) sum += uint64_t(t[i + 7]);
        r[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    uint64_t top = uint64_t(t[15]) + carry;
    r[8] = uint32_t(top);
    r[9] = uint32_t(top >> 32);

    carry = 0ull;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sum = uint64_t(r[i]) + carry;
        if (i == 0u) sum += uint64_t(r[8]) * 977ull;
        if (i == 1u) sum += uint64_t(r[8]) + uint64_t(r[9]) * 977ull;
        if (i == 2u) sum += uint64_t(r[9]);
        r[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    uint64_t high = carry;
    carry = 0ull;
    Fe reduced;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sum = uint64_t(r[i]) + carry;
        if (i == 0u) sum += high * 977ull;
        if (i == 1u) sum += high;
        reduced.v[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    return fe_normalize(reduced);
}

static __device__ __forceinline__ Fe fe_add(Fe a, Fe b) {
    uint32_t t[16];
    for (uint32_t i = 0; i < 16; ++i) t[i] = 0u;
    uint64_t carry = 0ull;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sum = uint64_t(a.v[i]) + uint64_t(b.v[i]) + carry;
        t[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    t[8] = uint32_t(carry);
    return fe_reduce(t);
}

static __device__ __forceinline__ Fe fe_sub(Fe a, Fe b) {
    Fe r;
    uint64_t borrow = 0ull;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sub = uint64_t(b.v[i]) + borrow;
        r.v[i] = a.v[i] - uint32_t(sub);
        borrow = uint64_t(uint64_t(a.v[i]) < sub);
    }
    uint32_t mask = mask_if(borrow != 0ull);
    uint64_t carry = 0ull;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t sum = uint64_t(r.v[i]) + uint64_t(FIELD_P[i] & mask) + carry;
        r.v[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    return r;
}

static __device__ __forceinline__ Fe fe_mul(Fe a, Fe b) {
    uint32_t t[16];
    for (uint32_t i = 0; i < 16; ++i) t[i] = 0u;
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t carry = 0ull;
        for (uint32_t j = 0; j < 8; ++j) {
            uint64_t product = uint64_t(a.v[i]) * uint64_t(b.v[j]) + uint64_t(t[i + j]) + carry;
            t[i + j] = uint32_t(product);
            carry = product >> 32;
        }
        t[i + 8] = uint32_t(carry);
    }
    return fe_reduce(t);
}

static __device__ __forceinline__ Fe fe_square(Fe a) { return fe_mul(a, a); }

static __device__ __forceinline__ Fe fe_squares(Fe a, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) a = fe_square(a);
    return a;
}

static __device__ __forceinline__ Fe fe_inverse(Fe a) {
    Fe x2 = fe_mul(fe_square(a), a);
    Fe x3 = fe_mul(fe_square(x2), a);
    Fe x6 = fe_mul(fe_squares(x3, 3u), x3);
    Fe x9 = fe_mul(fe_squares(x6, 3u), x3);
    Fe x11 = fe_mul(fe_squares(x9, 2u), x2);
    Fe x22 = fe_mul(fe_squares(x11, 11u), x11);
    Fe x44 = fe_mul(fe_squares(x22, 22u), x22);
    Fe x88 = fe_mul(fe_squares(x44, 44u), x44);
    Fe x176 = fe_mul(fe_squares(x88, 88u), x88);
    Fe x220 = fe_mul(fe_squares(x176, 44u), x44);
    Fe x223 = fe_mul(fe_squares(x220, 3u), x3);
    Fe t = fe_mul(fe_squares(x223, 23u), x22);
    t = fe_mul(fe_squares(t, 5u), a);
    t = fe_mul(fe_squares(t, 3u), x2);
    return fe_mul(fe_squares(t, 2u), a);
}

static __device__ __forceinline__ Point point_select(Point a, Point b, uint32_t mask) {
    Point r;
    r.x = fe_select(a.x, b.x, mask);
    r.y = fe_select(a.y, b.y, mask);
    r.z = fe_select(a.z, b.z, mask);
    return r;
}

static __device__ __forceinline__ Point add_mixed(Point a, Fe bx, Fe by) {
    Fe zz = fe_square(a.z);
    Fe u = fe_mul(bx, zz);
    Fe s = fe_mul(by, fe_mul(a.z, zz));
    Fe h = fe_sub(u, a.x);
    Fe r = fe_sub(s, a.y);
    Fe hh = fe_square(h);
    Fe hhh = fe_mul(h, hh);
    Fe v = fe_mul(a.x, hh);
    Point summed;
    summed.x = fe_sub(fe_sub(fe_square(r), hhh), fe_add(v, v));
    summed.y = fe_sub(fe_mul(r, fe_sub(v, summed.x)), fe_mul(a.y, hhh));
    summed.z = fe_mul(a.z, h);
    return summed;
}

static __device__ __forceinline__ Point add_window(Point a, Fe bx, Fe by, uint32_t digit) {
    Point summed = add_mixed(a, bx, by);
    Point b;
    b.x = bx;
    b.y = by;
    b.z = fe_one();
    summed = point_select(summed, b, fe_zero_mask(a.z));
    return point_select(summed, a, mask_if(digit == 0u));
}

static __device__ __forceinline__ uint32_t key_byte(const uint32_t *keys, uint32_t key_index, uint32_t i) {
    uint32_t word = keys[key_index * 8u + i / 4u];
    return (word >> ((i % 4u) * 8u)) & 0xffu;
}

static __device__ __forceinline__ void window_point(const uint32_t *keys, const uint32_t *table, uint32_t key_index,
                             uint32_t window, Fe &x, Fe &y, uint32_t &digit) {
    x = fe_zero();
    y = fe_zero();
    digit = key_byte(keys, key_index, 31u - 2u * window) |
            (key_byte(keys, key_index, 30u - 2u * window) << 8);
    uint32_t offset = (window * 65536u + digit) * 16u;
    for (uint32_t limb = 0; limb < 8; ++limb) {
        x.v[limb] = table[offset + limb];
        y.v[limb] = table[offset + 8u + limb];
    }
}

static __device__ __forceinline__ Point public_jacobian(const uint32_t *keys, const uint32_t *table, uint32_t key_index) {
    Point sum;
    sum.x = fe_zero();
    sum.y = fe_one();
    sum.z = fe_zero();
    for (uint32_t window = 0; window < (256u / WINDOW_BITS); ++window) {
        Fe x, y;
        uint32_t digit;
        window_point(keys, table, key_index, window, x, y, digit);
        sum = add_window(sum, x, y, digit);
    }
    return sum;
}

static __device__ __forceinline__ Point add_generator(Point p, const uint32_t *table) {
    Fe gx, gy;
    for (uint32_t limb = 0; limb < 8u; ++limb) {
        gx.v[limb] = table[16u + limb];
        gy.v[limb] = table[24u + limb];
    }
    return add_mixed(p, gx, gy);
}

static __device__ __forceinline__ uint32_t coordinate_byte(Fe a, uint32_t index) {
    return (a.v[7u - index / 4u] >> ((3u - index % 4u) * 8u)) & 0xffu;
}

__device__ __constant__ uint32_t KECCAK_ROT[25] = {
    0u,  1u, 62u, 28u, 27u, 36u, 44u, 6u,  55u, 20u, 3u,  10u, 43u,
    25u, 39u, 41u, 45u, 15u, 21u, 8u,  18u, 2u,  61u, 56u, 14u};

__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};

static __device__ __forceinline__ uint64_t rol(uint64_t x, uint32_t n) { return (x << n) | (x >> ((64u - n) & 63u)); }

static __device__ __forceinline__ void store_address(uint32_t *addresses, uint32_t index, uint32_t b[20]) {
    for (uint32_t w = 0; w < 5u; ++w) {
        addresses[index * 5u + w] = b[w * 4u] | (b[w * 4u + 1u] << 8) | (b[w * 4u + 2u] << 16) |
                                    (b[w * 4u + 3u] << 24);
    }
}

static __device__ __forceinline__ void eth_address(Point point, uint32_t *addresses, uint32_t index) {
    uint64_t state[25];
    for (uint32_t i = 0; i < 25; ++i) state[i] = 0ull;
    for (uint32_t i = 0; i < 32; ++i) {
        state[i / 8u] |= uint64_t(coordinate_byte(point.x, i)) << ((i % 8u) * 8u);
        state[4u + i / 8u] |= uint64_t(coordinate_byte(point.y, i)) << ((i % 8u) * 8u);
    }
    state[8] = 1ull;
    state[16] = 0x8000000000000000ull;
    for (uint32_t round = 0; round < 24; ++round) {
        uint64_t c[5];
        uint64_t b[25];
        for (uint32_t x = 0; x < 5; ++x)
            c[x] = state[x] ^ state[x + 5u] ^ state[x + 10u] ^ state[x + 15u] ^ state[x + 20u];
        for (uint32_t x = 0; x < 5; ++x) {
            uint64_t d = c[(x + 4u) % 5u] ^ rol(c[(x + 1u) % 5u], 1u);
            for (uint32_t y = 0; y < 5; ++y) state[x + 5u * y] ^= d;
        }
        for (uint32_t x = 0; x < 5; ++x)
            for (uint32_t y = 0; y < 5; ++y)
                b[y + 5u * ((2u * x + 3u * y) % 5u)] = rol(state[x + 5u * y], KECCAK_ROT[x + 5u * y]);
        for (uint32_t x = 0; x < 5; ++x)
            for (uint32_t y = 0; y < 5; ++y)
                state[x + 5u * y] =
                    b[x + 5u * y] ^ ((~b[(x + 1u) % 5u + 5u * y]) & b[(x + 2u) % 5u + 5u * y]);
        state[0] ^= KECCAK_RC[round];
    }
    uint32_t digest[20];
    for (uint32_t i = 0; i < 20; ++i)
        digest[i] = uint32_t((state[(i + 12u) / 8u] >> (((i + 12u) % 8u) * 8u)) & 0xffull);
    store_address(addresses, index, digest);
}

static __device__ __forceinline__ void montgomery_chunk_affine_keccak(Point pts[CHUNK_SIZE], uint32_t *addresses,
                                               uint32_t base, uint32_t count) {
    Fe prefix[CHUNK_SIZE];
    uint32_t zero_mask[CHUNK_SIZE];
    Fe acc = fe_one();
    for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
        uint32_t index = base + i;
        Fe z = index < count ? pts[i].z : fe_one();
        zero_mask[i] = fe_zero_mask(z);
        acc = fe_mul(acc, fe_select(z, fe_one(), zero_mask[i]));
        prefix[i] = acc;
    }
    Fe inv = fe_inverse(prefix[CHUNK_SIZE - 1]);
    for (uint32_t i = CHUNK_SIZE; i-- > 0u;) {
        uint32_t index = base + i;
        Fe z_inv = i == 0u ? inv : fe_mul(inv, prefix[i - 1u]);
        if (index < count) {
            inv = fe_mul(inv, fe_select(pts[i].z, fe_one(), zero_mask[i]));
            z_inv = fe_select(z_inv, fe_zero(), zero_mask[i]);
            Fe inverse2 = fe_square(z_inv);
            Point point;
            point.x = fe_mul(pts[i].x, inverse2);
            point.y = fe_mul(pts[i].y, fe_mul(inverse2, z_inv));
            point.z = fe_one();
            eth_address(point, addresses, index);
        }
    }
}

extern "C" __global__ void chunk_derive_addresses(const uint32_t *keys, const uint32_t *table,
                                                  uint32_t *addresses, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
#if INCREMENT_STRIDE > 1
    uint32_t base = gid * INCREMENT_STRIDE;
    if (base >= count) return;
    uint32_t chain = INCREMENT_STRIDE < count - base ? INCREMENT_STRIDE : count - base;
    Point p = public_jacobian(keys, table, base);
    for (uint32_t offset = 0; offset < chain; offset += CHUNK_SIZE) {
        uint32_t n = CHUNK_SIZE < chain - offset ? CHUNK_SIZE : chain - offset;
        Point pts[CHUNK_SIZE];
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            if (i < n) {
                pts[i] = p;
                if (offset + i + 1u < chain) p = add_generator(p, table);
            }
        }
        montgomery_chunk_affine_keccak(pts, addresses, base + offset, count);
    }
#else
    uint32_t base = gid * CHUNK_SIZE;
    if (base >= count) return;
    Point pts[CHUNK_SIZE];
    for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
        uint32_t index = base + i;
        if (index < count) pts[i] = public_jacobian(keys, table, index);
    }
    montgomery_chunk_affine_keccak(pts, addresses, base, count);
#endif
}
