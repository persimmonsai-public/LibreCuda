// Writes blas/src/mt19937_jump.inc: the jump polynomials of LibreCUDA's cuRAND MT19937 generator, x^(2^(1000 + i))
// mod the characteristic polynomial of MT19937 for i = 0 .. MT_JUMP_LEVELS - 1. Applied to a state (Horner's scheme
// with single steps), such a polynomial advances it by 2^(1000 + i) outputs: generator g of the 8192 starts at output
// g * 2^1000 of the sequence.
//
// The characteristic polynomial is the minimal polynomial of the sequence of the outputs' lowest bits
// (Berlekamp-Massey); the program checks a jump by 12345 outputs against stepping.
//
// usage: g++ -O2 -o gen_mt_jump gen_mt_jump.cpp && ./gen_mt_jump > src/mt19937_jump.inc
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int N = 624, M = 397, DEGREE = 19937, WORDS = (DEGREE + 64) / 64, LEVELS = 13;

struct Mt {
    uint32_t mt[N];
    int ptr = 0;

    explicit Mt(uint32_t seed) {
        mt[0] = seed;
        for (int i = 1; i < N; i++) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + uint32_t(i);
    }

    Mt() { memset(mt, 0, sizeof(mt)); }

    // one step: the new word replaces mt[ptr] (its output is the tempered new word)
    uint32_t step() {
        const uint32_t y = (mt[ptr] & 0x80000000u) | (mt[(ptr + 1) % N] & 0x7fffffffu);
        mt[ptr] = mt[(ptr + M) % N] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        const uint32_t out = mt[ptr];
        ptr = (ptr + 1) % N;
        return out;
    }

    static uint32_t temper(uint32_t y) {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        return y ^ (y >> 18);
    }

    void add(const Mt &o) {
        for (int k = 0; k < N; k++) mt[(ptr + k) % N] ^= o.mt[(o.ptr + k) % N];
    }
};

using Poly = std::vector<uint64_t>;  // GF(2) polynomial, bit i = coefficient of x^i

bool bit(const Poly &p, int i) {
    return i / 64 < int(p.size()) && (p[i / 64] >> (i % 64)) & 1;
}

int degree(const Poly &p) {
    for (int w = int(p.size()) - 1; w >= 0; w--) {
        if (p[w]) return w * 64 + 63 - __builtin_clzll(p[w]);
    }
    return -1;
}

// Berlekamp-Massey over GF(2): the connection polynomial C (C(0) = 1) of the shortest LFSR producing s
Poly berlekampMassey(const std::vector<uint8_t> &s) {
    const int n = int(s.size()), words = (n + 64) / 64;
    Poly c(words, 0), b(words, 0);
    c[0] = b[0] = 1;
    int l = 0, m = 1;
    for (int i = 0; i < n; i++) {
        int d = s[i];
        for (int j = 1; j <= l; j++) d ^= bit(c, j) & s[i - j];
        if (!d) {
            m++;
            continue;
        }
        Poly t = c;
        // c ^= b << m
        const int ws = m / 64, bs = m % 64;
        for (int w = words - 1; w >= ws; w--) {
            uint64_t v = b[w - ws] << bs;
            if (bs && w - ws - 1 >= 0) v |= b[w - ws - 1] >> (64 - bs);
            c[w] ^= v;
        }
        if (2 * l <= i) {
            l = i + 1 - l;
            b = t;
            m = 1;
        } else {
            m++;
        }
    }
    c.resize(WORDS);
    return c;
}

struct Reducer {
    Poly phi;
    std::vector<Poly> shifted;  // phi << s for s = 0 .. 63, 2 * WORDS words

    explicit Reducer(const Poly &p) : phi(p), shifted(64, Poly(2 * WORDS + 1, 0)) {
        for (int s = 0; s < 64; s++) {
            for (int w = 0; w < WORDS; w++) {
                shifted[s][w] ^= phi[w] << s;
                if (s) shifted[s][w + 1] ^= phi[w] >> (64 - s);
            }
        }
    }

    // p mod phi (p has 2 * WORDS + 1 words)
    Poly reduce(Poly p) const {
        for (int i = 64 * int(p.size()) - 1; i >= DEGREE; i--) {
            if (!bit(p, i)) continue;
            const int shift = i - DEGREE, ws = shift / 64, bs = shift % 64;
            const Poly &q = shifted[bs];
            for (int w = 0; w < int(q.size()) && w + ws < int(p.size()); w++) p[w + ws] ^= q[w];
        }
        p.resize(WORDS);
        return p;
    }

    Poly square(const Poly &a) const {
        Poly p(2 * WORDS + 1, 0);
        for (int w = 0; w < WORDS; w++) {
            for (int b = 0; b < 64; b++) {
                if ((a[w] >> b) & 1) {
                    const int i = 2 * (w * 64 + b);
                    p[i / 64] |= 1ull << (i % 64);
                }
            }
        }
        return reduce(p);
    }

    Poly mulX(const Poly &a) const {
        Poly p(2 * WORDS + 1, 0);
        for (int w = 0; w < WORDS; w++) {
            p[w] |= a[w] << 1;
            p[w + 1] |= a[w] >> 63;
        }
        return reduce(p);
    }
};

// the state advanced by the jump polynomial (Horner's scheme)
Mt jump(const Mt &s, const Poly &p) {
    Mt acc;
    acc.ptr = s.ptr;
    for (int i = degree(p); i >= 0; i--) {
        acc.step();
        if (bit(p, i)) acc.add(s);
    }
    return acc;
}

} // namespace

int main() {
    // the minimal polynomial of the lowest output bits: reversed connection polynomial
    Mt gen(5489u);
    std::vector<uint8_t> bits(2 * DEGREE + 100);
    for (auto &b: bits) b = Mt::temper(gen.step()) & 1;
    const Poly c = berlekampMassey(bits);
    if (degree(c) != DEGREE) {
        fprintf(stderr, "unexpected linear complexity %d\n", degree(c));
        return 1;
    }
    Poly phi(WORDS, 0);
    for (int i = 0; i <= DEGREE; i++) {
        if (bit(c, i)) phi[(DEGREE - i) / 64] |= 1ull << ((DEGREE - i) % 64);
    }
    const Reducer r(phi);

    // check: x^12345 mod phi jumps by 12345
    Poly x12345(WORDS, 0);
    x12345[0] = 1;
    for (int i = 0; i < 12345; i++) x12345 = r.mulX(x12345);
    Mt a(4357u), b(4357u);
    for (int i = 0; i < 1000; i++) a.step(), b.step();
    Mt jumped = jump(a, x12345);
    for (int i = 0; i < 12345; i++) b.step();
    for (int i = 0; i < 2000; i++) {
        if (jumped.step() != b.step()) {
            fprintf(stderr, "jump check failed at output %d\n", i);
            return 1;
        }
    }

    // check the squaring: x^(2^14) jumps by 16384
    Poly x16384(WORDS, 0);
    x16384[0] = 2;
    for (int i = 0; i < 14; i++) x16384 = r.square(x16384);
    jumped = jump(a, x16384);
    Mt c2(4357u);
    for (int i = 0; i < 1000 + 16384; i++) c2.step();
    for (int i = 0; i < 2000; i++) {
        if (jumped.step() != c2.step()) {
            fprintf(stderr, "squaring check failed at output %d\n", i);
            return 1;
        }
    }

    // x^(2^1000), then squares
    Poly p(WORDS, 0);
    p[0] = 2;
    for (int i = 0; i < 1000; i++) p = r.square(p);
    printf("// Generated by blas/gen_mt_jump.cpp; do not edit. Jump polynomials of MT19937: level i is x^(2^(1000 + i))\n");
    printf("// mod the characteristic polynomial, bit k of the 64 bit words = coefficient of x^k.\n\n");
    printf("constexpr int MT_JUMP_LEVELS = %d, MT_JUMP_WORDS = %d;\n", LEVELS, WORDS);
    printf("constexpr unsigned long long MT_JUMP[MT_JUMP_LEVELS][MT_JUMP_WORDS] = {\n");
    for (int level = 0; level < LEVELS; level++) {
        printf("    {\n");
        for (int w = 0; w < WORDS; w += 4) {
            printf("       ");
            for (int k = w; k < w + 4 && k < WORDS; k++) printf(" 0x%016llxull,", (unsigned long long) p[k]);
            printf("\n");
        }
        printf("    },\n");
        p = r.square(p);
    }
    printf("};\n");
    return 0;
}
