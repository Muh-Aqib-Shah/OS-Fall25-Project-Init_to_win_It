// user/sha256.c
// SHA-256 implementation for xv6
// Based on FIPS 180-4 specification

#include "kernel/types.h"
#include "user/user.h"
#include "user/sha256.h"

// SHA-256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
static const uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Helper functions
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

// SHA-256 context structure
typedef struct {
    unsigned char data[64];
    uint datalen;
    unsigned long long bitlen;
    unsigned long long total_bitlen;   // NEW: stores total message length in bits
    uint state[8];
} SHA256_CTX;

// Initialize SHA-256 context
static void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->total_bitlen = 0;   // NEW
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

// Process one 512-bit block
static void sha256_transform(SHA256_CTX *ctx, const unsigned char data[]) {
    uint a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    // Prepare message schedule
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    // Initialize working variables
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    // Main loop
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Update state
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

// Update SHA-256 context with new data
static void sha256_update(SHA256_CTX *ctx, const unsigned char data[], uint len) {
    uint i;
    
    ctx->total_bitlen += len * 8;  // ADD THIS LINE - Track total message length
    for (i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

// Finalize SHA-256 hash
static void sha256_final(SHA256_CTX *ctx, unsigned char hash[]) {
    uint i;

    i = ctx->datalen;

    // Pad with 0x80 byte
    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56)
            ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
        sha256_transform(ctx, ctx->data);
        for (i = 0; i < 56; ++i)
            ctx->data[i] = 0x00;
    }

    // Append length in bits as 64-bit big-endian integer
    // Use total_bitlen which tracks the ORIGINAL message length
    unsigned long long lenbits = ctx->total_bitlen;
    printf("DEBUG: total_bitlen = %lld bits (%lld bytes)\n", lenbits, lenbits / 8);
    // Write the 64-bit length in big-endian format
    ctx->data[56] = (unsigned char)((lenbits >> 56) & 0xff);
    ctx->data[57] = (unsigned char)((lenbits >> 48) & 0xff);
    ctx->data[58] = (unsigned char)((lenbits >> 40) & 0xff);
    ctx->data[59] = (unsigned char)((lenbits >> 32) & 0xff);
    ctx->data[60] = (unsigned char)((lenbits >> 24) & 0xff);
    ctx->data[61] = (unsigned char)((lenbits >> 16) & 0xff);
    ctx->data[62] = (unsigned char)((lenbits >> 8) & 0xff);
    ctx->data[63] = (unsigned char)(lenbits & 0xff);
    sha256_transform(ctx, ctx->data);

    // Produce final hash (big-endian)
    for (i = 0; i < 4; ++i) {
        hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
    }
}

// Public API: Compute SHA-256 hash
void sha256_hash(const unsigned char *data, unsigned int len, unsigned char hash[32]) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash);
}

// Helper function to compare hashes
static int hash_equals(const unsigned char *h1, const unsigned char *h2) {
    for (int i = 0; i < 32; i++) {
        if (h1[i] != h2[i]) return 0;
    }
    return 1;
}

// Helper function to print hash in hex
static void print_hash(const unsigned char *hash) {
    const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        printf("%c%c", hex[hash[i] >> 4], hex[hash[i] & 0xf]);
    }
}

// Test function
static void run_test(const char *name, const char *input, const unsigned char *expected) {
    unsigned char hash[32];
    unsigned int len = 0;
    
    // Calculate length manually
    if (input) {
        while (input[len] != '\0') len++;
    }
    
    sha256_hash((const unsigned char *)input, len, hash);
    
    printf("Test: %s\n", name);
    printf("  Input: \"%s\"\n", input ? input : "");
    printf("  Expected: ");
    print_hash(expected);
    printf("\n  Computed: ");
    print_hash(hash);
    printf("\n  Result: %s\n\n", hash_equals(hash, expected) ? "PASS" : "FAIL");
}

int main(int argc, char *argv[]) {
    printf("SHA-256 Test Suite\n");
    printf("==================\n\n");

    // Test 1: Empty string
    const unsigned char expected1[32] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8,
        0x99, 0x6f, 0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
    };
    run_test("Empty string", "", expected1);

    // Test 2: Single character
    const unsigned char expected2[32] = {
        0xca, 0x97, 0x81, 0x12, 0xca, 0x1b, 0xbd, 0xca, 0xfa, 0xc2, 0x31, 0xb3,
        0x9a, 0x23, 0xdc, 0x4d, 0xa7, 0x86, 0xef, 0xf8, 0x14, 0x7c, 0x4e, 0x72,
        0xb9, 0x80, 0x77, 0x85, 0xaf, 0xee, 0x48, 0xbb
    };
    run_test("Single character 'a'", "a", expected2);

    // Test 3: "hello world"
    const unsigned char expected3[32] = {
        0xb9, 0x4d, 0x27, 0xb9, 0x93, 0x4d, 0x3e, 0x08, 0xa5, 0x2e, 0x52, 0xd7,
        0xda, 0x7d, 0xab, 0xfa, 0xc4, 0x84, 0xef, 0xe3, 0x7a, 0x53, 0x80, 0xee,
        0x90, 0x88, 0xf7, 0xac, 0xe2, 0xef, 0xcd, 0xe9
    };
    run_test("hello world", "hello world", expected3);

    // Test 4: 62-byte string
const unsigned char expected4[32] = {
    0x54, 0x03, 0x63, 0xd1, 0x07, 0x1a, 0x00, 0x29, 0x97, 0x29, 0x0c, 0xd8,
    0xf4, 0xa2, 0xbd, 0xf3, 0xac, 0xd0, 0x35, 0x5f, 0xfa, 0xd3, 0xb3, 0xf2,
    0x5f, 0x52, 0xaa, 0xd6, 0xeb, 0xad, 0x93, 0x6a
};
run_test("Block boundary", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", expected4);

// Test 5: Multi-block string (101 bytes)
const unsigned char expected5[32] = {
    0x65, 0xda, 0xb9, 0xc0, 0xa2, 0x77, 0x2f, 0x0e, 0xa4, 0x65, 0x4a, 0xab,
    0xc5, 0xcb, 0x63, 0xc8, 0x3a, 0x6e, 0xe0, 0x18, 0x24, 0x9e, 0xf5, 0xd1,
    0x04, 0xbe, 0xd2, 0xad, 0x71, 0x41, 0xa9, 0xe1
};
run_test("Multi-block", "The quick brown fox jumps over the lazy dog. This is a longer test string that spans multiple blocks.", expected5);

    printf("All tests completed!\n");
    exit(0);
}
