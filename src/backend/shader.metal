#include <metal_stdlib>
using namespace metal;

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
inline Point add_window(Point a, Fe bx, Fe by, uint digit) {
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
    Point b = {bx, by, fe_one()};
    out = point_select(out, b, fe_zero_mask(a.z));
    return point_select(out, a, mask_if(digit == 0));
}

inline Point public_point(device const uchar *key, device const uint *table) {
    Point sum = {fe_zero(), fe_one(), fe_zero()};
    for (uint window = 0; window < 64; ++window) {
        uint digit = (uint(key[31 - window / 2]) >> ((window % 2) * 4)) & 15u;
        Fe x = {}, y = {};
        // Scan all entries. Memory addresses depend only on public loop indices.
        for (uint entry = 0; entry < 16; ++entry) {
            uint mask = mask_if(entry == digit);
            uint offset = (window * 16 + entry) * 16;
            for (uint limb = 0; limb < 8; ++limb) {
                x.v[limb] |= table[offset + limb] & mask;
                y.v[limb] |= table[offset + 8 + limb] & mask;
            }
        }
        // Inputs are host-validated scalars 0<k<n. Nonzero windows are disjoint
        // positive scalar terms; partial sums never reach n. Thus a finite sum
        // cannot equal or negate the next window point (the h=0 exceptions).
        // Infinity/zero digits are handled with masks in add_window.
        sum = add_window(sum, x, y, digit);
    }
    Fe inverse = fe_inverse(sum.z);
    Fe inverse2 = fe_square(inverse);
    sum.x = fe_mul(sum.x, inverse2);
    sum.y = fe_mul(sum.y, fe_mul(inverse2, inverse));
    sum.z = fe_one();
    return sum;
}
inline uchar coordinate_byte(Fe a, uint index) {
    return uchar(a.v[7 - index / 4] >> ((3 - index % 4) * 8));
}

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul,0x0000000000008082ul,0x800000000000808aul,0x8000000080008000ul,
    0x000000000000808bul,0x0000000080000001ul,0x8000000080008081ul,0x8000000000008009ul,
    0x000000000000008aul,0x0000000000000088ul,0x0000000080008009ul,0x000000008000000aul,
    0x000000008000808bul,0x800000000000008bul,0x8000000000008089ul,0x8000000000008003ul,
    0x8000000000008002ul,0x8000000000000080ul,0x000000000000800aul,0x800000008000000aul,
    0x8000000080008081ul,0x8000000000008080ul,0x0000000080000001ul,0x8000000080008008ul
};
constant uint KECCAK_ROT[25] = {0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
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

kernel void derive_addresses(device const uchar *keys [[buffer(0)]],
                             device const uint *table [[buffer(1)]],
                             device uchar *addresses [[buffer(2)]],
                             constant uint &count [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return; // public batch boundary
    eth_address(public_point(keys + gid*32, table), addresses + gid*20);
}
