// user/udp_client.c
// UDP Client Library for Reliable File Transfer

#include "kernel/types.h"
#include "user/user.h"
#include "user/udp_client.h"

// Only include the sha256_hash function, not the whole file
// We'll declare it extern and link separately
//extern void sha256_hash(const unsigned char *data, unsigned int len, unsigned char hash[32]);
// Include SHA-256 implementation
#define SHA256_AS_LIBRARY
#include "sha256.c"

// ============================================
// Helper Functions
// ============================================

static uint htonl_local(uint x) {
    return ((x & 0xFF000000) >> 24) |
           ((x & 0x00FF0000) >> 8)  |
           ((x & 0x0000FF00) << 8)  |
           ((x & 0x000000FF) << 24);
}

static uint ntohl_local(uint x) {
    return htonl_local(x);
}

static ushort ntohs_local(ushort x) {
    return ((x & 0xFF00) >> 8) | ((x & 0x00FF) << 8);
}

static void my_memcpy(char *dst, const char *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

static void my_memset(char *dst, char val, int n) {
    for (int i = 0; i < n; i++) dst[i] = val;
}

static int my_memcmp(const char *s1, const char *s2, int n) {
    for (int i = 0; i < n; i++) {
        if (s1[i] != s2[i]) return (unsigned char)s1[i] - (unsigned char)s2[i];
    }
    return 0;
}

static void print_hash_hex(const unsigned char *hash) {
    const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        printf("%c%c", hex[hash[i] >> 4], hex[hash[i] & 0xf]);
    }
}

// ============================================
// File Metadata Structure
// ============================================
struct file_metadata {
    uint size;
    uint packet_count;
    unsigned char sha256[32];
};

// ============================================
// Protocol Functions
// ============================================

static int request_metadata(int file_id, struct file_metadata *meta) {
    char req[2];
    char resp[64];
    uint src_ip;
    ushort src_port;
    int n;
    
    req[0] = MSG_METADATA_REQUEST;
    req[1] = (char)file_id;
    
    printf("Requesting metadata for file %d...\n", file_id);
    
    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        if (send(CLIENT_PORT, SERVER_IP, SERVER_PORT, req, 2) < 0) {
            printf("  send failed (attempt %d)\n", retry + 1);
            continue;
        }
        
        n = recv(CLIENT_PORT, &src_ip, &src_port, resp, sizeof(resp));
        if (n < 0) {
            printf("  recv failed (attempt %d)\n", retry + 1);
            continue;
        }
        
        if (n > 0 && (unsigned char)resp[0] == MSG_ERROR) {
            printf("  Server error: 0x%x\n", (unsigned char)resp[1]);
            return -1;
        }
        
        if (n < 41 || (unsigned char)resp[0] != MSG_METADATA_RESPONSE) {
            printf("  Invalid response (len=%d, type=0x%x)\n", n, (unsigned char)resp[0]);
            continue;
        }
        
        uint size_net, count_net;
        my_memcpy((char*)&size_net, resp + 1, 4);
        my_memcpy((char*)&count_net, resp + 5, 4);
        
        meta->size = ntohl_local(size_net);
        meta->packet_count = ntohl_local(count_net);
        my_memcpy((char*)meta->sha256, resp + 9, 32);
        
        printf("  File size: %d bytes\n", meta->size);
        printf("  Packet count: %d\n", meta->packet_count);
        printf("  SHA-256: ");
        print_hash_hex(meta->sha256);
        printf("\n");
        
        return 0;
    }
    
    printf("  Failed after %d retries\n", MAX_RETRIES);
    return -1;
}

static int request_packet(int file_id, uint packet_idx, char *data_out, int *len_out) {
    char req[6];
    char resp[PACKET_SIZE + 16];
    uint src_ip;
    ushort src_port;
    int n;
    
    req[0] = MSG_DATA_REQUEST;
    req[1] = (char)file_id;
    uint idx_net = htonl_local(packet_idx);
    my_memcpy(req + 2, (char*)&idx_net, 4);
    
    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        if (send(CLIENT_PORT, SERVER_IP, SERVER_PORT, req, 6) < 0) continue;
        
        n = recv(CLIENT_PORT, &src_ip, &src_port, resp, sizeof(resp));
        if (n < 0) continue;
        
        if (n > 0 && (unsigned char)resp[0] == MSG_ERROR) return -1;
        
        if (n < 7 || (unsigned char)resp[0] != MSG_DATA_RESPONSE) continue;
        
        uint recv_idx_net;
        ushort data_len_net;
        my_memcpy((char*)&recv_idx_net, resp + 1, 4);
        my_memcpy((char*)&data_len_net, resp + 5, 2);
        
        uint recv_idx = ntohl_local(recv_idx_net);
        ushort data_len = ntohs_local(data_len_net);
        
        if (recv_idx != packet_idx) continue;
        if (data_len > PACKET_SIZE || n < 7 + data_len) continue;
        
        my_memcpy(data_out, resp + 7, data_len);
        *len_out = data_len;
        return 0;
    }
    return -1;
}

// ============================================
// Main Fetch Function
// ============================================
static int port_bound = 0;
char* fetch_file(int file_id, int *size_out) {
    struct file_metadata meta;
    char *file_data = 0;
    char *received = 0;
    int success = 0;
    
    printf("\n========================================\n");
    printf("Starting file transfer for file ID %d\n", file_id);
    printf("========================================\n\n");
    
    if (!port_bound) {
         if(bind(CLIENT_PORT) < 0) {
              printf("ERROR: Failed to bind to port %d\n", CLIENT_PORT);
              return 0;
        }
        port_bound = 1;
    }
    
    if (request_metadata(file_id, &meta) < 0) {
        printf("ERROR: Failed to get metadata\n");
        goto cleanup;
    }
    
    file_data = malloc(meta.size);
    if (!file_data) {
        printf("ERROR: Failed to allocate file buffer (%d bytes)\n", meta.size);
        goto cleanup;
    }
    
    received = malloc(meta.packet_count);
    if (!received) {
        printf("ERROR: Failed to allocate tracking buffer\n");
        goto cleanup;
    }
    my_memset(received, 0, meta.packet_count);
    
    printf("\nFetching %d packets...\n", meta.packet_count);
    
    char packet_buf[PACKET_SIZE];
    
    for (uint i = 0; i < meta.packet_count; i++) {
        int pkt_len;
        if (request_packet(file_id, i, packet_buf, &pkt_len) == 0) {
            uint offset = i * PACKET_SIZE;
            if (offset + pkt_len > meta.size) pkt_len = meta.size - offset;
            my_memcpy(file_data + offset, packet_buf, pkt_len);
            received[i] = 1;
            if ((i + 1) % 1000 == 0 || i == meta.packet_count - 1) {
                printf("  Progress: %d/%d packets (%d%%)\n", i + 1, meta.packet_count, (i + 1) * 100 / meta.packet_count);
            }
        }
    }
    
    printf("\nChecking for missing packets...\n");
    for (int pass = 0; pass < MAX_RETRIES; pass++) {
        int missing = 0;
        for (uint i = 0; i < meta.packet_count; i++) {
            if (!received[i]) {
                missing++;
                int pkt_len;
                if (request_packet(file_id, i, packet_buf, &pkt_len) == 0) {
                    uint offset = i * PACKET_SIZE;
                    if (offset + pkt_len > meta.size) pkt_len = meta.size - offset;
                    my_memcpy(file_data + offset, packet_buf, pkt_len);
                    received[i] = 1;
                    printf("  Recovered packet %d\n", i);
                }
            }
        }
        if (missing == 0) { printf("  All packets received!\n"); break; }
        else { printf("  Pass %d: %d packets still missing\n", pass + 1, missing); }
    }
    
    int final_missing = 0;
    for (uint i = 0; i < meta.packet_count; i++) {
        if (!received[i]) final_missing++;
    }
    if (final_missing > 0) {
        printf("ERROR: %d packets still missing after retries\n", final_missing);
        goto cleanup;
    }
    
    printf("\nVerifying SHA-256 checksum...\n");
    unsigned char computed_hash[32];
    sha256_hash((unsigned char*)file_data, meta.size, computed_hash);
    
    printf("  Expected: "); print_hash_hex(meta.sha256); printf("\n");
    printf("  Computed: "); print_hash_hex(computed_hash); printf("\n");
    
    if (my_memcmp((char*)computed_hash, (char*)meta.sha256, 32) != 0) {
        printf("ERROR: SHA-256 MISMATCH - file corrupted!\n");
        goto cleanup;
    }
    
    printf("  SHA-256 VERIFIED!\n");
    printf("\nFile transfer complete: %d bytes\n", meta.size);
    *size_out = meta.size;
    success = 1;
    
cleanup:
    if (received) free(received);
    if (!success && file_data) { free(file_data); file_data = 0; }
    return file_data;
}

char* fetch_model_weights(int *size_out) {
    printf("\n################################################\n");
    printf("#  Fetching Model Weights (stories15M.bin)     #\n");
    printf("################################################\n");
    return fetch_file(FILE_ID_MODEL, size_out);
}

char* fetch_tokenizer(int *size_out) {
    printf("\n################################################\n");
    printf("#  Fetching Tokenizer (tokenizer.bin)          #\n");
    printf("################################################\n");
    return fetch_file(FILE_ID_TOKENIZER, size_out);
}

int main(int argc, char *argv[]) {
    printf("\n========================================================\n");
    printf("   UDP Client Test - RFTP File Transfer\n");
    printf("========================================================\n");
    printf("\nMake sure file_server.py is running on the host!\n");
    
    int model_size = 0, tokenizer_size = 0;
    int model_ok = 0, tokenizer_ok = 0;
    
    char *model_data = fetch_model_weights(&model_size);
    if (model_data) { printf("\n*** Model weights: SUCCESS (%d bytes) ***\n", model_size); model_ok = 1; free(model_data); }
    else { printf("\n*** Model weights: FAILED ***\n"); }
    
    char *tokenizer_data = fetch_tokenizer(&tokenizer_size);
    if (tokenizer_data) { printf("\n*** Tokenizer: SUCCESS (%d bytes) ***\n", tokenizer_size); tokenizer_ok = 1; free(tokenizer_data); }
    else { printf("\n*** Tokenizer: FAILED ***\n"); }
    
    printf("\n========================================================\n");
    printf("   Test Summary\n");
    printf("========================================================\n");
    printf("  Model weights: %s (%d bytes)\n", model_ok ? "PASS" : "FAIL", model_size);
    printf("  Tokenizer:     %s (%d bytes)\n", tokenizer_ok ? "PASS" : "FAIL", tokenizer_size);
    printf("========================================================\n\n");
    
    exit((model_ok && tokenizer_ok) ? 0 : 1);
}
