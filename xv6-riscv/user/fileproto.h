// user/fileproto.h

#ifndef FILEPROTO_H
#define FILEPROTO_H

#include "types.h"

#define FILEPROTO_CHUNK_SIZE     1024
#define FILEPROTO_MAX_FILENAME   64

// Message types
#define MSG_META_REQ   1
#define MSG_META_RESP  2
#define MSG_DATA_REQ   3
#define MSG_DATA_RESP  4
#define MSG_ERROR      5

struct msg_header {
  uint8 type;     // MSG_*
  uint8 reserved; // unused
  uint16 flags;   // unused, set to 0
};

struct msg_meta_req {
  struct msg_header hdr;          // type = MSG_META_REQ
  uint8 filename_len;             // <= FILEPROTO_MAX_FILENAME
  char  filename[FILEPROTO_MAX_FILENAME];
};

struct msg_meta_resp {
  struct msg_header hdr;          // type = MSG_META_RESP
  uint64 file_size;               // network byte order
  uint32 chunk_count;             // network byte order
  uint8  sha256[32];              // full-file SHA-256
};

struct msg_data_req {
  struct msg_header hdr;          // type = MSG_DATA_REQ
  uint32 chunk_index;             // network byte order
};

struct msg_data_resp {
  struct msg_header hdr;          // type = MSG_DATA_RESP
  uint32 chunk_index;             // network byte order
  uint16 data_len;                // network byte order
  // followed by data_len bytes
};

#endif

