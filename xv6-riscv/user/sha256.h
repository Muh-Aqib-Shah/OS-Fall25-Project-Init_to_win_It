// user/sha256.h
// SHA-256 cryptographic hash function
// Public interface for computing SHA-256 hashes

#ifndef SHA256_H
#define SHA256_H

// Compute SHA-256 hash of data
// Parameters:
//   data: pointer to input data buffer
//   len: length of input data in bytes
//   hash: output buffer for 32-byte hash (must be allocated by caller)
void sha256_hash(const unsigned char *data, unsigned int len, unsigned char hash[32]);

#endif // SHA256_H
