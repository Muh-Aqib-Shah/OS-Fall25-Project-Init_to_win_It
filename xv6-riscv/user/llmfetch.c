#include "types.h"
#include "stat.h"
#include "user.h"
#include "fileproto.h"
#include "sha256.h"
#include "net.h"

#define CLIENT_PORT  3000
#define SERVER_PORT  3001
#define SERVER_IP    MAKE_IP_ADDR(10,0,2,2)  // host in qemu user net

extern uint64 htonll(uint64 x);
extern uint64 ntohll(uint64 x);

static int
send_meta_req(short sport, int dst, short dport, const char *fname)
{
  char buf[sizeof(struct msg_meta_req)];
  struct msg_meta_req *req = (struct msg_meta_req *)buf;

  int flen = strlen(fname);
  if (flen > FILEPROTO_MAX_FILENAME)
    flen = FILEPROTO_MAX_FILENAME;

  req->hdr.type = MSG_META_REQ;
  req->hdr.reserved = 0;
  req->hdr.flags = htons(0);
  req->filename_len = flen;
  memmove(req->filename, fname, flen);

  int len = sizeof(struct msg_header) + 1 + flen;
  return send(sport, dst, dport, buf, len);
}

static int
recv_meta_resp(short dport, struct msg_meta_resp *out)
{
  int src_ip;
  short src_port;
  char buf[128];

  int n = recv(dport, &src_ip, &src_port, buf, sizeof(buf));
  if (n < (int)sizeof(struct msg_meta_resp))
    return -1;

  struct msg_meta_resp *resp = (struct msg_meta_resp *)buf;
  if (resp->hdr.type != MSG_META_RESP)
    return -1;

  memmove(out, resp, sizeof(struct msg_meta_resp));
  return 0;
}

static int
recv_meta_resp(short dport, struct msg_meta_resp *out)
{
  int src_ip;
  short src_port;
  char buf[128];

  int n = recv(dport, &src_ip, &src_port, buf, sizeof(buf));
  if (n < (int)sizeof(struct msg_meta_resp))
    return -1;

  struct msg_meta_resp *resp = (struct msg_meta_resp *)buf;
  if (resp->hdr.type != MSG_META_RESP)
    return -1;

  memmove(out, resp, sizeof(struct msg_meta_resp));
  return 0;
}

static int
send_data_req(short sport, int dst, short dport, uint32 idx)
{
  struct msg_data_req req;
  req.hdr.type = MSG_DATA_REQ;
  req.hdr.reserved = 0;
  req.hdr.flags = htons(0);
  req.chunk_index = htonl(idx);

  return send(sport, dst, dport, (char *)&req, sizeof(req));
}

static int
recv_data_resp(short dport, uint32 *idx_out, char *data_buf, int *len_out)
{
  int src_ip;
  short src_port;
  char buf[FILEPROTO_CHUNK_SIZE + 32];

  int n = recv(dport, &src_ip, &src_port, buf, sizeof(buf));
  if (n < (int)sizeof(struct msg_data_resp))
    return -1;

  struct msg_data_resp *resp = (struct msg_data_resp *)buf;
  if (resp->hdr.type != MSG_DATA_RESP)
    return -1;

  uint32 idx = ntohl(resp->chunk_index);
  uint16 dlen = ntohs(resp->data_len);

  if (dlen > FILEPROTO_CHUNK_SIZE)
    return -1;

  if (n < (int)(sizeof(struct msg_data_resp) + dlen))
    return -1;

  *idx_out = idx;
  *len_out = dlen;

  char *payload = buf + sizeof(struct msg_data_resp);
  memmove(data_buf, payload, dlen);

  return 0;
}

static int
download_file(const char *fname)
{
  // 1. bind to CLIENT_PORT once
  if (bind(CLIENT_PORT) < 0) {
    printf("bind failed\n");
    return -1;
  }

  // 2. request metadata
  if (send_meta_req(CLIENT_PORT, SERVER_IP, SERVER_PORT, fname) < 0) {
    printf("send_meta_req failed\n");
    return -1;
  }

  struct msg_meta_resp meta;
  if (recv_meta_resp(CLIENT_PORT, &meta) < 0) {
    printf("recv_meta_resp failed\n");
    return -1;
  }

  uint64 file_size = ntohll(meta.file_size);
  uint32 chunk_count = ntohl(meta.chunk_count);

  printf("Downloading %s: %d bytes, %d chunks\n", fname, (int)file_size, chunk_count);

  // 3. allocate buffer + bitmap
  char *file_buf = malloc(file_size);
  char *received = malloc(chunk_count);
  if (!file_buf || !received) {
    printf("malloc failed\n");
    if (file_buf) free(file_buf);
    if (received) free(received);
    return -1;
  }
  memset(received, 0, chunk_count);

  // 4. first pass: request each chunk once
  char chunk[FILEPROTO_CHUNK_SIZE];
  for (uint32 i = 0; i < chunk_count; i++) {
    send_data_req(CLIENT_PORT, SERVER_IP, SERVER_PORT, i);

    uint32 idx;
    int dlen;
    if (recv_data_resp(CLIENT_PORT, &idx, chunk, &dlen) < 0) {
      printf("recv_data_resp failed\n");
      free(file_buf);
      free(received);
      return -1;
    }

    if (idx >= chunk_count) {
      // ignore bogus
      i--;
      continue;
    }

    if (!received[idx]) {
      uint64 offset = (uint64)idx * FILEPROTO_CHUNK_SIZE;
      if (offset + dlen > file_size) {
        dlen = file_size - offset;
      }
      memmove(file_buf + offset, chunk, dlen);
      received[idx] = 1;
    }
  }

  // 5. retry missing chunks a few times
  for (int pass = 0; pass < 3; pass++) {
    int missing = 0;
    for (uint32 i = 0; i < chunk_count; i++) {
      if (!received[i]) {
        missing++;
        send_data_req(CLIENT_PORT, SERVER_IP, SERVER_PORT, i);

        uint32 idx;
        int dlen;
        if (recv_data_resp(CLIENT_PORT, &idx, chunk, &dlen) == 0 &&
            idx < chunk_count) {
          uint64 offset = (uint64)idx * FILEPROTO_CHUNK_SIZE;
          if (offset + dlen > file_size) {
            dlen = file_size - offset;
          }
          memmove(file_buf + offset, chunk, dlen);
          received[idx] = 1;
        }
      }
    }
    if (missing == 0)
      break;
  }

  for (uint32 i = 0; i < chunk_count; i++) {
    if (!received[i]) {
      printf("Failed: missing chunk %d for %s\n", i, fname);
      free(file_buf);
      free(received);
      return -1;
    }
  }

  // 6. SHA-256
  unsigned char hash[32];
  sha256_hash((unsigned char *)file_buf, (unsigned int)file_size, hash);

  printf("%s SHA256 (computed): ", fname);
  for (int i = 0; i < 32; i++)
    printf("%02x", hash[i]);
  printf("\n");

  printf("%s SHA256 (expected): ", fname);
  for (int i = 0; i < 32; i++)
    printf("%02x", meta.sha256[i]);
  printf("\n");

  int match = 1;
  for (int i = 0; i < 32; i++) {
    if (hash[i] != meta.sha256[i]) {
      match = 0;
      break;
    }
  }

  if (match)
    printf("%s: SHA256 MATCH\n", fname);
  else
    printf("%s: SHA256 MISMATCH\n", fname);

  free(file_buf);
  free(received);
  return match ? 0 : -1;
}

int
main(int argc, char *argv[])
{
  if (download_file("model.bin") < 0)
    exit(1);
  if (download_file("tokenizer.bin") < 0)
    exit(1);
  exit(0);
}

