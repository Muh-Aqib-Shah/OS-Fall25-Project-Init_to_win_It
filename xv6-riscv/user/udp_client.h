// user/udp_client.h
// UDP Client Library for Reliable File Transfer
// Implements RFTP protocol client side

#ifndef UDP_CLIENT_H
#define UDP_CLIENT_H

// ============================================
// Protocol Constants
// ============================================
#define SERVER_IP       0x0A000202   // 10.0.2.2 (QEMU host)
#define SERVER_PORT     9999
#define CLIENT_PORT     8888
#define PACKET_SIZE     1024
#define MAX_RETRIES     3

// File IDs
#define FILE_ID_MODEL       0
#define FILE_ID_TOKENIZER   1

// Message Types
#define MSG_METADATA_REQUEST    0x01
#define MSG_METADATA_RESPONSE   0x02
#define MSG_DATA_REQUEST        0x03
#define MSG_DATA_RESPONSE       0x04
#define MSG_ERROR               0xFF

// ============================================
// Base Protocol Functions
// ============================================

// Fetch a file by its ID
// Parameters:
//   file_id: FILE_ID_MODEL (0) or FILE_ID_TOKENIZER (1)
//   size_out: pointer to store the file size
// Returns:
//   Pointer to allocated buffer containing file data
//   NULL on failure
// Caller must free() the returned pointer
char* fetch_file(int file_id, int *size_out);

// ============================================
// LLM-Specific Wrapper Functions
// ============================================

// Fetch model weights file (stories15M.bin)
// Returns pointer to data and sets size_out to number of bytes
// Returns NULL on failure
// Caller must free() the returned pointer
char* fetch_model_weights(int *size_out);

// Fetch tokenizer file (tokenizer.bin)
// Returns pointer to data and sets size_out to number of bytes
// Returns NULL on failure
// Caller must free() the returned pointer
char* fetch_tokenizer(int *size_out);

#endif // UDP_CLIENT_H
