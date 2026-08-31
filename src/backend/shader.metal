#include <metal_stdlib>
using namespace metal;

#ifndef OPT_INVERT
#define OPT_INVERT 0
#endif
#ifndef WINDOW_BITS
#define WINDOW_BITS 4
#endif
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 0
#endif
#ifndef OPT_KECCAK
#define OPT_KECCAK 0
#endif
#ifndef INCREMENT_STRIDE
#define INCREMENT_STRIDE 1
#endif

// Little-endian 32-bit limbs, canonical modulo p = 2^256 - 2^32 - 977.
// All scalar-dependent choices below use masks, not branches or table indices.
struct Fe { uint v[8]; };
struct Point { Fe x; Fe y; Fe z; };
constant uint FIELD_P[8] = {0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
                            0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};

inline Fe fe_zero() { Fe r = {}; return r; }
inline Fe fe_one() { Fe r = {}; r.v[0] = 1; return r; }
inline uint mask_if(bool b) { return 0u - uint(b); }
inline Fe fe_select(Fe a, Fe b, uint mask) {
    Fe r;
    for (uint i = 0; i < 8; ++i) r.v[i] = (a.v[i] & ~mask) | (b.v[i] & mask);
    return r;
}
inline uint fe_zero_mask(Fe a) {
    uint bits = 0;
    for (uint i = 0; i < 8; ++i) bits |= a.v[i];
    return mask_if(bits == 0);
}
inline Fe fe_normalize(Fe a) {
    Fe d;
    ulong borrow = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sub = ulong(FIELD_P[i]) + borrow;
        d.v[i] = a.v[i] - uint(sub);
        borrow = ulong(a.v[i]) < sub;
    }
    return fe_select(d, a, mask_if(borrow != 0));
}

inline Fe fe_reduce(thread const uint *t) {
    // Fold the upper 256 bits using 2^256 == 2^32 + 977 (mod p).
    uint r[10] = {};
    ulong carry = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(t[i]) + ulong(t[i + 8]) * 977ul + carry;
        if (i != 0) sum += t[i + 7]; // public loop index
        r[i] = uint(sum);
        carry = sum >> 32;
    }
    ulong top = ulong(t[15]) + carry;
    r[8] = uint(top);
    r[9] = uint(top >> 32);

    // The residual upper value is at most 33 bits. Fold both limbs together.
    carry = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(r[i]) + carry;
        if (i == 0) sum += ulong(r[8]) * 977ul;
        if (i == 1) sum += ulong(r[8]) + ulong(r[9]) * 977ul;
        if (i == 2) sum += r[9];
        r[i] = uint(sum);
        carry = sum >> 32;
    }
    // One final carry fold suffices: when it overflows, the low part is <2^66.
    ulong high = carry;
    carry = 0;
    Fe out;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(r[i]) + carry;
        if (i == 0) sum += high * 977ul;
        if (i == 1) sum += high;
        out.v[i] = uint(sum);
        carry = sum >> 32;
    }
    return fe_normalize(out);
}

inline Fe fe_add(Fe a, Fe b) {
#if OPT_ADD
    Fe out;
    ulong carry = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(a.v[i]) + b.v[i] + carry;
        out.v[i] = uint(sum);
        carry = sum >> 32;
    }
    // For canonical a,b, a+b <= 2p-2. If the 257th bit is set,
    // low <= 2^256-2*(2^32+977)-2; adding 2^32+977 cannot overflow
    // and is already <p. Without that bit, one subtraction normalizes.
    ulong high = carry;
    carry = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(out.v[i]) + carry;
        if (i == 0) sum += high * 977ul;
        if (i == 1) sum += high;
        out.v[i] = uint(sum);
        carry = sum >> 32;
    }
    return fe_normalize(out);
#else
    uint t[16] = {};
    ulong carry = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(a.v[i]) + b.v[i] + carry;
        t[i] = uint(sum);
        carry = sum >> 32;
    }
    t[8] = uint(carry);
    return fe_reduce(t);
#endif
}
inline Fe fe_sub(Fe a, Fe b) {
    Fe r;
    ulong borrow = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sub = ulong(b.v[i]) + borrow;
        r.v[i] = a.v[i] - uint(sub);
        borrow = ulong(a.v[i]) < sub;
    }
    uint mask = mask_if(borrow != 0);
    ulong carry = 0;
    for (uint i = 0; i < 8; ++i) {
        ulong sum = ulong(r.v[i]) + (FIELD_P[i] & mask) + carry;
        r.v[i] = uint(sum);
        carry = sum >> 32;
    }
    return r;
}
inline Fe fe_mul(Fe a, Fe b) {
    uint t[16] = {};
    for (uint i = 0; i < 8; ++i) {
        ulong carry = 0;
        for (uint j = 0; j < 8; ++j) {
            // Each sum fits u64: (2^32-1)^2 + 2*(2^32-1) <= 2^64-1.
            ulong product = ulong(a.v[i]) * b.v[j] + t[i + j] + carry;
            t[i + j] = uint(product);
            carry = product >> 32;
        }
        t[i + 8] = uint(carry);
    }
    return fe_reduce(t);
}
#if OPT_SQUARE
inline void square_accumulate(thread uint *acc, ulong product) {
    ulong sum = ulong(acc[0]) + uint(product);
    acc[0] = uint(sum);
    sum = ulong(acc[1]) + uint(product >> 32) + (sum >> 32);
    acc[1] = uint(sum);
    acc[2] += uint(sum >> 32);
}
inline Fe fe_square(Fe a) {
    uint t[16] = {};
    uint acc[3] = {};
    // At most eight 64-bit terms plus the preceding column's carry:
    // the accumulator needs <68 bits, safely inside its explicit 96 bits.
    // Add cross terms twice instead of overflowing a doubled u64 product.
    #pragma unroll
    for (uint column = 0; column < 15; ++column) {
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            if (i <= column) {
                uint j = column - i;
                if (j < 8 && i <= j) {
                    ulong product = ulong(a.v[i]) * a.v[j];
                    square_accumulate(acc, product);
                    if (i != j) square_accumulate(acc, product);
                }
            }
        }
        t[column] = acc[0];
        acc[0] = acc[1]; acc[1] = acc[2]; acc[2] = 0;
    }
    // A 256-bit square is <2^512: no limb above t[15] can remain.
    t[15] = acc[0];
    return fe_reduce(t);
}
#else
inline Fe fe_square(Fe a) { return fe_mul(a, a); }
#endif
inline Fe fe_squares(Fe a, uint count) {
    for (uint i = 0; i < count; ++i) a = fe_square(a);
    return a;
}
inline Fe fe_inverse(Fe a) {
    // Fixed addition chain for p-2 = 2^256 - 2^32 - 979. xK = a^(2^K-1).
    Fe x2 = fe_mul(fe_square(a), a);
    Fe x3 = fe_mul(fe_square(x2), a);
    Fe x6 = fe_mul(fe_squares(x3, 3), x3);
    Fe x9 = fe_mul(fe_squares(x6, 3), x3);
    Fe x11 = fe_mul(fe_squares(x9, 2), x2);
    Fe x22 = fe_mul(fe_squares(x11, 11), x11);
    Fe x44 = fe_mul(fe_squares(x22, 22), x22);
    Fe x88 = fe_mul(fe_squares(x44, 44), x44);
    Fe x176 = fe_mul(fe_squares(x88, 88), x88);
    Fe x220 = fe_mul(fe_squares(x176, 44), x44);
    Fe x223 = fe_mul(fe_squares(x220, 3), x3);
    Fe t = fe_mul(fe_squares(x223, 23), x22);
    t = fe_mul(fe_squares(t, 5), a);
    t = fe_mul(fe_squares(t, 3), x2);
    return fe_mul(fe_squares(t, 2), a);
}

inline Point point_select(Point a, Point b, uint mask) {
    Point r = {fe_select(a.x, b.x, mask), fe_select(a.y, b.y, mask), fe_select(a.z, b.z, mask)};
    return r;
}
inline Point add_mixed(Point a, Fe bx, Fe by) {
    Fe zz = fe_square(a.z);
    Fe u = fe_mul(bx, zz);
    Fe s = fe_mul(by, fe_mul(a.z, zz));
    Fe h = fe_sub(u, a.x);
    Fe r = fe_sub(s, a.y);
    Fe hh = fe_square(h);
    Fe hhh = fe_mul(h, hh);
    Fe v = fe_mul(a.x, hh);
    Point out;
    out.x = fe_sub(fe_sub(fe_square(r), hhh), fe_add(v, v));
    out.y = fe_sub(fe_mul(r, fe_sub(v, out.x)), fe_mul(a.y, hhh));
    out.z = fe_mul(a.z, h);
    return out;
}
inline Point add_window(Point a, Fe bx, Fe by, uint digit) {
    Point out = add_mixed(a, bx, by);
    Point b = {bx, by, fe_one()};
    out = point_select(out, b, fe_zero_mask(a.z));
    return point_select(out, a, mask_if(digit == 0));
}

inline void store_fe(device uint *out, Fe a) {
    for (uint i = 0; i < 8; ++i) out[i] = a.v[i];
}
inline Fe load_fe(device const uint *in) {
    Fe a;
    for (uint i = 0; i < 8; ++i) a.v[i] = in[i];
    return a;
}
inline void tg_store(threadgroup uint *slot, Fe a) {
    for (uint i = 0; i < 8; ++i) slot[i] = a.v[i];
}
inline Fe tg_load(threadgroup const uint *slot) {
    Fe a;
    for (uint i = 0; i < 8; ++i) a.v[i] = slot[i];
    return a;
}

// 4-bit windows: scan every table entry so addresses depend only on public
// loop indices. 8/16-bit windows cannot scan the whole row; digit indexes it
// (secret-dependent address, a performance choice, not constant-time scanning).
inline void window_point(device const uchar *key, device const uint *table,
                         uint window, thread Fe &x, thread Fe &y, thread uint &digit) {
    x = fe_zero();
    y = fe_zero();
#if WINDOW_BITS == 8
    digit = uint(key[31 - window]);
    uint offset = (window * 256u + digit) * 16u;
    for (uint limb = 0; limb < 8; ++limb) {
        x.v[limb] = table[offset + limb];
        y.v[limb] = table[offset + 8 + limb];
    }
#elif WINDOW_BITS == 16
    // Big-endian scalar bytes: key[31-2w] carries bits 16w..16w+7 (low byte).
    digit = uint(key[31 - 2 * window]) | (uint(key[30 - 2 * window]) << 8);
    uint offset = (window * 65536u + digit) * 16u;
    for (uint limb = 0; limb < 8; ++limb) {
        x.v[limb] = table[offset + limb];
        y.v[limb] = table[offset + 8 + limb];
    }
#else
    digit = (uint(key[31 - window / 2]) >> ((window % 2) * 4)) & 15u;
    for (uint entry = 0; entry < 16; ++entry) {
        uint mask = mask_if(entry == digit);
        uint offset = (window * 16 + entry) * 16;
        for (uint limb = 0; limb < 8; ++limb) {
            x.v[limb] |= table[offset + limb] & mask;
            y.v[limb] |= table[offset + 8 + limb] & mask;
        }
    }
#endif
}

inline Point public_jacobian(device const uchar *key, device const uint *table) {
    Point sum = {fe_zero(), fe_one(), fe_zero()};
    for (uint window = 0; window < (256u / WINDOW_BITS); ++window) {
        Fe x, y;
        uint digit;
        window_point(key, table, window, x, y, digit);
        // Inputs are host-validated scalars 0<k<n. Nonzero windows are disjoint
        // positive scalar terms; partial sums never reach n. Thus a finite sum
        // cannot equal or negate the next window point (the h=0 exceptions).
        // Infinity/zero digits are handled with masks in add_window.
        sum = add_window(sum, x, y, digit);
    }
    return sum;
}

// Window 0 digit 1 is G in every table layout (4/8/16-bit). Host chains stay in
// [2, n-1], so P+G never hits the mixed-add doubling/infinity exceptions.
inline Point add_generator(Point p, device const uint *table) {
    Fe gx, gy;
    for (uint limb = 0; limb < 8; ++limb) {
        gx.v[limb] = table[16 + limb];
        gy.v[limb] = table[24 + limb];
    }
    return add_mixed(p, gx, gy);
}

inline Point to_affine(Point sum) {
    Fe inverse = fe_inverse(sum.z);
    Fe inverse2 = fe_square(inverse);
    sum.x = fe_mul(sum.x, inverse2);
    sum.y = fe_mul(sum.y, fe_mul(inverse2, inverse));
    sum.z = fe_one();
    return sum;
}

inline Point public_point(device const uchar *key, device const uint *table) {
    return to_affine(public_jacobian(key, table));
}

// Replace a zero Z with 1 so the threadgroup product stays invertible, then
// restore a zero inverse. Inactive (padding) lanes also pass Z=1.
inline Fe montgomery_threadgroup_inverse(
    Fe z, uint lid, uint n,
    threadgroup uint *zs, threadgroup uint *prefix, threadgroup uint *inverses) {
    uint z_zero = fe_zero_mask(z);
    Fe z_work = fe_select(z, fe_one(), z_zero);
    tg_store(zs + lid * 8, z_work);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        Fe acc = tg_load(zs);
        tg_store(prefix, acc);
        for (uint i = 1; i < n; ++i) {
            acc = fe_mul(acc, tg_load(zs + i * 8));
            tg_store(prefix + i * 8, acc);
        }
        Fe inv = fe_inverse(acc);
        for (uint i = n; i-- > 0; ) {
            Fe prev = i == 0 ? fe_one() : tg_load(prefix + (i - 1) * 8);
            tg_store(inverses + i * 8, fe_mul(inv, prev));
            inv = fe_mul(inv, tg_load(zs + i * 8));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return fe_select(tg_load(inverses + lid * 8), fe_zero(), z_zero);
}
inline uchar coordinate_byte(Fe a, uint index) {
    return uchar(a.v[7 - index / 4] >> ((3 - index % 4) * 8));
}

constant uint KECCAK_ROT[25] = {0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
#if OPT_KECCAK
// Bit-interleaved Keccak-f[1600]: every 64-bit lane is split into two native
// 32-bit words (even bits in .e, odd bits in .o), so all rotations become
// 32-bit rotations. Metal has no native 64-bit ALU; this removes the emulation
// from the hot rounds. Conversions only run at absorb/squeeze time.
struct Lane { uint e; uint o; };
inline uint rol32(uint x, uint n) { return (x << (n & 31u)) | (x >> ((32u - n) & 31u)); }
// rol64 by n: even/odd words swap when n is odd (bit 2i+1 -> even position).
inline Lane lane_rol(Lane a, uint n) {
    Lane r;
    if (n & 1u) {
        r.e = rol32(a.o, n / 2u + 1u);
        r.o = rol32(a.e, n / 2u);
    } else {
        r.e = rol32(a.e, n / 2u);
        r.o = rol32(a.o, n / 2u);
    }
    return r;
}
inline uint compact_even_bits(ulong x) {
    x &= 0x5555555555555555ul;
    x = (x | (x >> 1)) & 0x3333333333333333ul;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0ful;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00fful;
    x = (x | (x >> 8)) & 0x0000ffff0000fffful;
    return uint(x | (x >> 16));
}
inline Lane lane_from(ulong w) {
    Lane r = { compact_even_bits(w), compact_even_bits(w >> 1) };
    return r;
}
inline ulong expand_even_bits(uint x) {
    ulong t = ulong(x);
    t = (t | (t << 16)) & 0x0000ffff0000fffful;
    t = (t | (t << 8)) & 0x00ff00ff00ff00fful;
    t = (t | (t << 4)) & 0x0f0f0f0f0f0f0f0ful;
    t = (t | (t << 2)) & 0x3333333333333333ul;
    t = (t | (t << 1)) & 0x5555555555555555ul;
    return t;
}
inline ulong lane_to(Lane a) { return expand_even_bits(a.e) | (expand_even_bits(a.o) << 1); }
// Generated by interleaving KECCAK_RC (see repository history); verified by a
// round-trip check and by the address differential tests.
constant uint KECCAK_RC_E[24] = {
    0x00000001u,0x00000000u,0x00000000u,0x00000000u,
    0x00000001u,0x00000001u,0x00000001u,0x00000001u,
    0x00000000u,0x00000000u,0x00000001u,0x00000000u,
    0x00000001u,0x00000001u,0x00000001u,0x00000001u,
    0x00000000u,0x00000000u,0x00000000u,0x00000000u,
    0x00000001u,0x00000000u,0x00000001u,0x00000000u
};
constant uint KECCAK_RC_O[24] = {
    0x00000000u,0x00000089u,0x8000008bu,0x80008080u,
    0x0000008bu,0x00008000u,0x80008088u,0x80000082u,
    0x0000000bu,0x0000000au,0x00008082u,0x00008003u,
    0x0000808bu,0x8000000bu,0x8000008au,0x80000081u,
    0x80000081u,0x80000008u,0x00000083u,0x80008003u,
    0x80008088u,0x80000088u,0x00008000u,0x80008082u
};
inline void eth_address(Point point, device uchar *address) {
    Lane state[25] = {};
    for (uint word = 0; word < 4; ++word) {
        ulong x = 0, y = 0;
        for (uint i = 0; i < 8; ++i) {
            x |= ulong(coordinate_byte(point.x, word * 8 + i)) << (i * 8);
            y |= ulong(coordinate_byte(point.y, word * 8 + i)) << (i * 8);
        }
        state[word] = lane_from(x);
        state[4 + word] = lane_from(y);
    }
    // Ethereum legacy Keccak padding: 0x01, not SHA3's 0x06. Rate = 136 bytes.
    // interleave(1) = (1, 0); interleave(1 << 63) = (0, 1 << 31).
    state[8].e = 1u;
    state[16].o = 0x80000000u;
    for (uint round = 0; round < 24; ++round) {
        Lane c[5], b[25];
        for (uint x = 0; x < 5; ++x) {
            c[x].e = state[x].e ^ state[x+5].e ^ state[x+10].e ^ state[x+15].e ^ state[x+20].e;
            c[x].o = state[x].o ^ state[x+5].o ^ state[x+10].o ^ state[x+15].o ^ state[x+20].o;
        }
        for (uint x = 0; x < 5; ++x) {
            Lane rotated = lane_rol(c[(x+1)%5], 1);
            Lane d = { c[(x+4)%5].e ^ rotated.e, c[(x+4)%5].o ^ rotated.o };
            for (uint y = 0; y < 5; ++y) {
                state[x+5*y].e ^= d.e;
                state[x+5*y].o ^= d.o;
            }
        }
        for (uint x = 0; x < 5; ++x)
            for (uint y = 0; y < 5; ++y)
                b[y+5*((2*x+3*y)%5)] = lane_rol(state[x+5*y], KECCAK_ROT[x+5*y]);
        for (uint x = 0; x < 5; ++x)
            for (uint y = 0; y < 5; ++y) {
                state[x+5*y].e = b[x+5*y].e ^ ((~b[(x+1)%5+5*y].e) & b[(x+2)%5+5*y].e);
                state[x+5*y].o = b[x+5*y].o ^ ((~b[(x+1)%5+5*y].o) & b[(x+2)%5+5*y].o);
            }
        state[0].e ^= KECCAK_RC_E[round];
        state[0].o ^= KECCAK_RC_O[round];
    }
    // Address = digest bytes 12..31, i.e. parts of state words 1..3.
    for (uint word = 1; word < 4; ++word) {
        ulong lane = lane_to(state[word]);
        for (uint byte = 0; byte < 8; ++byte) {
            uint index = word * 8 + byte;
            if (index >= 12) address[index - 12] = uchar(lane >> (byte * 8));
        }
    }
}
#else
constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul,0x0000000000008082ul,0x800000000000808aul,0x8000000080008000ul,
    0x000000000000808bul,0x0000000080000001ul,0x8000000080008081ul,0x8000000000008009ul,
    0x000000000000008aul,0x0000000000000088ul,0x0000000080008009ul,0x000000008000000aul,
    0x000000008000808bul,0x800000000000008bul,0x8000000000008089ul,0x8000000000008003ul,
    0x8000000000008002ul,0x8000000000000080ul,0x000000000000800aul,0x800000008000000aul,
    0x8000000080008081ul,0x8000000000008080ul,0x0000000080000001ul,0x8000000080008008ul
};
inline ulong rol(ulong x, uint n) { return (x << n) | (x >> ((64u - n) & 63u)); }
inline void eth_address(Point point, device uchar *address) {
    ulong state[25] = {};
    for (uint i = 0; i < 32; ++i) {
        state[i / 8] |= ulong(coordinate_byte(point.x, i)) << ((i % 8) * 8);
        state[4 + i / 8] |= ulong(coordinate_byte(point.y, i)) << ((i % 8) * 8);
    }
    // Ethereum legacy Keccak padding: 0x01, not SHA3's 0x06. Rate = 136 bytes.
    state[8] = 1;
    state[16] = 0x8000000000000000ul;
    for (uint round = 0; round < 24; ++round) {
        ulong c[5], b[25];
        for (uint x = 0; x < 5; ++x) c[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
        for (uint x = 0; x < 5; ++x) {
            ulong d = c[(x+4)%5] ^ rol(c[(x+1)%5], 1);
            for (uint y = 0; y < 5; ++y) state[x+5*y] ^= d;
        }
        for (uint x = 0; x < 5; ++x)
            for (uint y = 0; y < 5; ++y)
                b[y+5*((2*x+3*y)%5)] = rol(state[x+5*y], KECCAK_ROT[x+5*y]);
        for (uint x = 0; x < 5; ++x)
            for (uint y = 0; y < 5; ++y)
                state[x+5*y] = b[x+5*y] ^ ((~b[(x+1)%5+5*y]) & b[(x+2)%5+5*y]);
        state[0] ^= KECCAK_RC[round];
    }
    for (uint i = 0; i < 20; ++i) address[i] = uchar(state[(i+12)/8] >> (((i+12)%8)*8));
}
#endif

kernel void derive_addresses(device const uchar *keys [[buffer(0)]],
                             device const uint *table [[buffer(1)]],
                             device uchar *addresses [[buffer(2)]],
                             constant uint &count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return; // public batch boundary
    eth_address(public_point(keys + gid*32, table), addresses + gid*20);
}

kernel void jacobian_points(device const uchar *keys [[buffer(0)]],
                            device const uint *table [[buffer(1)]],
                            device uint *xyz [[buffer(2)]],
                            constant uint &count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    Point p = public_jacobian(keys + gid * 32, table);
    device uint *slot = xyz + gid * 24;
    store_fe(slot, p.x);
    store_fe(slot + 8, p.y);
    store_fe(slot + 16, p.z);
}

kernel void invert_affine_keccak(device const uint *xyz [[buffer(0)]],
                                 device const uint *unused [[buffer(1)]],
                                 device uchar *addresses [[buffer(2)]],
                                 constant uint &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint tpg [[threads_per_threadgroup]]) {
    (void)unused;
    threadgroup uint zs[256 * 8];
    threadgroup uint prefix[256 * 8];
    threadgroup uint inverses[256 * 8];
    bool active = gid < count;
    Fe z = fe_one();
    Fe x = fe_zero();
    Fe y = fe_zero();
    if (active) {
        device const uint *slot = xyz + gid * 24;
        x = load_fe(slot);
        y = load_fe(slot + 8);
        z = load_fe(slot + 16);
    }
    Fe z_inv = montgomery_threadgroup_inverse(z, lid, tpg, zs, prefix, inverses);
    if (!active) return;
    Fe inverse2 = fe_square(z_inv);
    Point point = {fe_mul(x, inverse2), fe_mul(y, fe_mul(inverse2, z_inv)), fe_one()};
    eth_address(point, addresses + gid * 20);
}

#if CHUNK_SIZE > 0
// Each thread owns CHUNK_SIZE consecutive points and amortizes one fe_inverse
// over them with a private Montgomery ladder: no barriers, no threadgroup
// memory (the rejected threadgroup variant serialized the whole group on one
// lane). Zero Z never occurs for host-validated scalars, but the same mask
// defense as invert_affine_keccak is kept: a zero Z contributes 1 to the
// running product and its inverse is forced back to zero.
inline void montgomery_chunk_affine_keccak(thread const Point *pts, uint base, uint count,
                                           device uchar *addresses) {
    Fe prefix[CHUNK_SIZE];
    uint zero_mask[CHUNK_SIZE];
    Fe acc = fe_one();
    for (uint i = 0; i < CHUNK_SIZE; ++i) {
        uint index = base + i;
        Fe z = index < count ? pts[i].z : fe_one();
        zero_mask[i] = fe_zero_mask(z);
        acc = fe_mul(acc, fe_select(z, fe_one(), zero_mask[i]));
        prefix[i] = acc;
    }
    Fe inv = fe_inverse(prefix[CHUNK_SIZE - 1]);
    // Walk backwards: before step i, inv is the inverse of prefix[i], so
    // multiplying by prefix[i-1] isolates this element's Z inverse.
    for (uint i = CHUNK_SIZE; i-- > 0; ) {
        uint index = base + i;
        Fe z_inv = i == 0 ? inv : fe_mul(inv, prefix[i - 1]);
        if (index < count) {
            // Padding lanes contributed 1, so skipping their inv update is exact.
            inv = fe_mul(inv, fe_select(pts[i].z, fe_one(), zero_mask[i]));
            z_inv = fe_select(z_inv, fe_zero(), zero_mask[i]);
            Fe inverse2 = fe_square(z_inv);
            Point point = {fe_mul(pts[i].x, inverse2),
                           fe_mul(pts[i].y, fe_mul(inverse2, z_inv)), fe_one()};
            eth_address(point, addresses + index * 20);
        }
    }
}

kernel void chunk_invert_affine_keccak(device const uint *xyz [[buffer(0)]],
                                       device const uint *unused [[buffer(1)]],
                                       device uchar *addresses [[buffer(2)]],
                                       constant uint &count [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    (void)unused;
    uint base = gid * CHUNK_SIZE;
    if (base >= count) return; // public batch boundary
    Point pts[CHUNK_SIZE];
    for (uint i = 0; i < CHUNK_SIZE; ++i) {
        uint index = base + i;
        if (index < count) {
            device const uint *slot = xyz + index * 24;
            pts[i].x = load_fe(slot);
            pts[i].y = load_fe(slot + 8);
            pts[i].z = load_fe(slot + 16);
        }
    }
    montgomery_chunk_affine_keccak(pts, base, count, addresses);
}

// Same chunk invert, but Jacobian stays in thread storage: no 96-byte/point
// device round-trip. Register pressure may spill; that is a measured tradeoff.
kernel void chunk_derive_addresses(device const uchar *keys [[buffer(0)]],
                                   device const uint *table [[buffer(1)]],
                                   device uchar *addresses [[buffer(2)]],
                                   constant uint &count [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
#if INCREMENT_STRIDE > 1
    uint base = gid * INCREMENT_STRIDE;
    if (base >= count) return;
    uint chain = min(uint(INCREMENT_STRIDE), count - base);
    Point p = public_jacobian(keys + base * 32, table);
    // CHUNK_SIZE == INCREMENT_STRIDE inverts the whole chain once; smaller
    // chunks repeat invert to cut thread-private Point arrays.
    for (uint offset = 0; offset < chain; offset += CHUNK_SIZE) {
        uint n = min(uint(CHUNK_SIZE), chain - offset);
        Point pts[CHUNK_SIZE];
        for (uint i = 0; i < CHUNK_SIZE; ++i) {
            if (i < n) {
                pts[i] = p;
                if (offset + i + 1 < chain) p = add_generator(p, table);
            }
        }
        montgomery_chunk_affine_keccak(pts, base + offset, count, addresses);
    }
#else
    uint base = gid * CHUNK_SIZE;
    if (base >= count) return;
    Point pts[CHUNK_SIZE];
    for (uint i = 0; i < CHUNK_SIZE; ++i) {
        uint index = base + i;
        if (index < count) pts[i] = public_jacobian(keys + index * 32, table);
    }
    montgomery_chunk_affine_keccak(pts, base, count, addresses);
#endif
}
#endif
