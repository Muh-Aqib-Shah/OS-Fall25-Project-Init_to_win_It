#include "types.h"

uint64 htonll(uint64 x) {
  return ((uint64)htonl((uint32)(x >> 32))) |
         (((uint64)htonl((uint32)(x & 0xffffffff))) << 32);
}

uint64 ntohll(uint64 x) {
  return ((uint64)ntohl((uint32)(x >> 32))) |
         (((uint64)ntohl((uint32)(x & 0xffffffff))) << 32);
}

