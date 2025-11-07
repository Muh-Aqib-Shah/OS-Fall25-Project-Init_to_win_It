// llama2c/cstring.c
#include "mini_lib.h"

/* Only provide what xv6 userlib does NOT: isprint/isspace.
   memcpy/memset/strcmp/strlen/strcpy are already in user/ulib.c. */

int isprint(int c) {
  unsigned char uc = (unsigned char)c;
  return (uc >= 0x20 && uc <= 0x7e);
}

int isspace(int c) {
  unsigned char uc = (unsigned char)c;
  return uc==' ' || uc=='\t' || uc=='\n' || uc=='\r' || uc=='\v' || uc=='\f';
}

