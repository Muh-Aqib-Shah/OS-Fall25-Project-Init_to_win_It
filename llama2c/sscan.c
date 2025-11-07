// sscan.c — minimal vsscanf/sscanf for xv6 userland
// Supports: %d, %f, %lf, %s, %c, %%
// Whitespace in format matches any amount of whitespace in input.
// Return value = number of conversions assigned (like libc scanf).

#include "mini_lib.h"   // must declare: isspace, isprint, memcpy, memset, strlen, va_list, etc.

/* ---------- small helpers (no libc deps) ---------- */

static const char *skip_spaces(const char *p) {
  while (*p==' ' || *p=='\t' || *p=='\n' || *p=='\r' || *p=='\v' || *p=='\f') p++;
  return p;
}

static const char *parse_signed_int(const char *p, int *out) {
  p = skip_spaces(p);
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p=='-'); p++; }

  long v = 0; int any = 0;
  while (*p >= '0' && *p <= '9') { any = 1; v = v*10 + (*p - '0'); p++; }
  if (!any) return (const char *)0;

  *out = neg ? -(int)v : (int)v;
  return p;
}

// Fixed-format float parser: [-+]? (digits)? ( . digits )? ( [eE] [-+]? digits )?
// Writes into *out as double; caller casts to float if needed.
static const char *parse_float(const char *p, double *out) {
  p = skip_spaces(p);
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p=='-'); p++; }

  double val = 0.0;
  int any = 0;

  while (*p >= '0' && *p <= '9') { any = 1; val = val*10.0 + (*p - '0'); p++; }

  if (*p == '.') {
    p++;
    double place = 0.1;
    while (*p >= '0' && *p <= '9') { any = 1; val += (*p - '0') * place; place *= 0.1; p++; }
  }

  if (!any) return (const char *)0;

  if (*p == 'e' || *p == 'E') {
    const char *q = p + 1;
    int eneg = 0, have_exp = 0, expv = 0;
    if (*q == '+' || *q == '-') { eneg = (*q=='-'); q++; }
    while (*q >= '0' && *q <= '9') { have_exp = 1; expv = expv*10 + (*q - '0'); q++; }
    if (have_exp) {
      double pow10 = 1.0;
      for (int i = 0; i < expv; i++) pow10 *= 10.0;
      val = eneg ? (val / pow10) : (val * pow10);
      p = q;
    }
  }

  *out = neg ? -val : val;
  return p;
}

/* ---------- core scanner ---------- */

int vsscanf(const char *str, const char *fmt, va_list ap) {
  const char *s = str;
  int nconv = 0;  // number of successful assignments

  for (const char *f = fmt; *f; f++) {
    if (*f != '%') {
      if (*f == ' ' || *f == '\t' || *f == '\n' || *f == '\r' || *f == '\v' || *f == '\f') {
        // Any whitespace in format matches any amount of whitespace in input
        s = skip_spaces(s);
      } else {
        // literal character match
        if (*s != *f) break;
        s++;
      }
      continue;
    }

    // handle "%%" -> literal '%'
    f++;
    if (*f == '%') {
      if (*s != '%') break;
      s++;
      continue;
    }

    // (Width/length modifiers ignored except we handle 'l' before 'f' for %lf.)
    int length_l = 0;
    while (*f == 'l') { length_l++; f++; }  // we only care about 1 'l' for %lf

    // Conversion specifier
    char spec = *f;
    if (!spec) break;

    if (spec == 'd') {
      int *ip = va_arg(ap, int *);
      int v;
      const char *np = parse_signed_int(s, &v);
      if (!np) break;           // conversion fails -> stop
      *ip = v;
      s = np;
      nconv++;
    } else if (spec == 'f') {
      double d;
      const char *np = parse_float(s, &d);
      if (!np) break;           // fail
      if (length_l) {
        // %lf expects double*
        double *dp = va_arg(ap, double *);
        *dp = d;
      } else {
        // %f expects float*
        float *fp = va_arg(ap, float *);
        *fp = (float)d;
      }
      s = np;
      nconv++;
    } else if (spec == 's') {
      char *dst = va_arg(ap, char *);
      // %s skips leading whitespace, then copies non-space sequence
      s = skip_spaces(s);
      if (*s == 0) break;       // nothing to read
      int k = 0;
      while (*s && !(*s==' ' || *s=='\t' || *s=='\n' || *s=='\r' || *s=='\v' || *s=='\f')) {
        dst[k++] = *s++;
      }
      dst[k] = 0;
      nconv++;
    } else if (spec == 'c') {
      // %c does NOT skip whitespace; read the next character if present
      char *cp = va_arg(ap, char *);
      if (*s == 0) break;       // no input left
      *cp = *s++;
      nconv++;
    } else {
      // Unknown specifier — behave like failure and stop.
      break;
    }
  }

  return nconv;
}

int sscanf(const char *str, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int r = vsscanf(str, fmt, ap);
  va_end(ap);
  return r;
}

