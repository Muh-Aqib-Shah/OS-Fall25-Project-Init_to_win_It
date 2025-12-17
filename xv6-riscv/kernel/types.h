typedef unsigned int   uint;
typedef unsigned short ushort;
typedef unsigned char  uchar;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int  uint32;
typedef unsigned long uint64;

#ifndef _MUTEX_T_DEFINED
#define _MUTEX_T_DEFINED

typedef struct {
  int locked;
  int owner_tid;
} mutex_t;

#endif


typedef uint64 pde_t;
