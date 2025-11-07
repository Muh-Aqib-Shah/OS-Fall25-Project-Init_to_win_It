#pragma once
#include <stdarg.h>

/* Only the extras we provide in llama2c.
   DO NOT redeclare memcpy/memset/strcmp/strlen/strcpy here,
   those come from xv6 user/ulib.c and/or toolchain headers. */

int vsnprintf(char *buf, int bufsz, const char *fmt, va_list ap);
int snprintf(char *buf, int bufsz, const char *fmt, ...);
int sprintf(char *buf, const char *fmt, ...);

int sscanf(const char *str, const char *fmt, ...);

int isprint(int c);
int isspace(int c);

