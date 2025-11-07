// printfmt.c
#include "mini_lib.h"
/* Avoid including <string.h>; just declare strlen so we have a prototype. */
int strlen(const char *s);

/* ------------ small internal helpers ------------ */

static int outch(char *buf, int cap, int *pos, char c) {
  // Count regardless of truncation
  if (*pos < cap) buf[*pos] = c;
  (*pos)++;
  return 1;
}

static int outstr_n(char *buf, int cap, int *pos, const char *s, int n) {
  int wrote = 0;
  for (int i = 0; i < n; i++) {
    wrote += outch(buf, cap, pos, s[i]);
  }
  return wrote;
}

static int pad_left(char *buf, int cap, int *pos, int width, int content_len) {
  int pads = (width > content_len) ? (width - content_len) : 0;
  for (int i = 0; i < pads; i++) outch(buf, cap, pos, ' ');
  return pads;
}

/* Convert signed int to decimal ASCII (no sign), returns length */
static int itoa_dec_abs(int v, char *tmp, int tmpsz) {
  unsigned int x = (v < 0) ? (unsigned)(-v) : (unsigned)v;
  int i = 0;
  if (x == 0) tmp[i++] = '0';
  while (x && i < tmpsz) {
    tmp[i++] = (char)('0' + (x % 10));
    x /= 10;
  }
  // reverse
  for (int l = 0, r = i - 1; l < r; l++, r--) {
    char t = tmp[l]; tmp[l] = tmp[r]; tmp[r] = t;
  }
  tmp[i] = 0;
  return i;
}

/* Minimal fixed-format float to ASCII with rounding. Returns length in tmp. */
static int dtoa_fixed(double d, int prec, char *tmp, int tmpsz) {
  if (tmpsz <= 0) return 0;
  if (prec < 0) prec = 6;

  int pos = 0;
  int neg = (d < 0.0);
  if (neg) d = -d;

  // compute 10^prec
  double pow10 = 1.0;
  for (int i = 0; i < prec; i++) pow10 *= 10.0;

  // scale & round (half-up)
  unsigned long long scaled = (unsigned long long)(d * pow10 + 0.5);

  unsigned long long ip = (prec > 0) ? (scaled / (unsigned long long)pow10) : (unsigned long long)(d + 0.5);
  unsigned long long fp = (prec > 0) ? (scaled % (unsigned long long)pow10) : 0ULL;

  // integer part
  char ibuf[32]; int il = 0;
  if (ip == 0ULL) ibuf[il++] = '0';
  while (ip && il < (int)sizeof(ibuf)) {
    ibuf[il++] = (char)('0' + (ip % 10ULL));
    ip /= 10ULL;
  }
  for (int l = 0, r = il - 1; l < r; l++, r--) {
    char t = ibuf[l]; ibuf[l] = ibuf[r]; ibuf[r] = t;
  }

  if (neg && pos < tmpsz) tmp[pos++] = '-';
  for (int i = 0; i < il && pos < tmpsz; i++) tmp[pos++] = ibuf[i];

  if (prec > 0) {
    if (pos < tmpsz) tmp[pos++] = '.';
    // write fractional part, zero-padded to 'prec'
    char fbuf[64];
    if (prec > (int)sizeof(fbuf)) prec = (int)sizeof(fbuf); // clamp
    for (int i = 0; i < prec; i++) { fbuf[prec - 1 - i] = (char)('0' + (fp % 10ULL)); fp /= 10ULL; }
    for (int i = 0; i < prec && pos < tmpsz; i++) tmp[pos++] = fbuf[i];
  }

  if (pos >= tmpsz) pos = tmpsz - 1;
  tmp[pos] = 0;
  return pos;
}

/* ------------ public formatting functions ------------ */

int vsnprintf(char *buf, int bufsz, const char *fmt, va_list ap) {
  // 'cap' is the number of bytes available for actual characters (reserve 1 for NUL if bufsz > 0)
  int cap = (bufsz > 0) ? (bufsz - 1) : 0;
  int pos = 0;      // how many chars we *attempted* to write (pos may exceed cap)
  int total = 0;    // same as pos; keep a separate running total for readability

  for (const char *p = fmt; *p; p++) {
    if (*p != '%') { total += outch(buf, cap, &pos, *p); continue; }

    // parse: % [width] [.precision] [specifier]
    p++; // skip '%'
    if (*p == '%') { total += outch(buf, cap, &pos, '%'); continue; }

    // width (only positive decimal, no flags)
    int width = 0;
    while (*p >= '0' && *p <= '9') { width = width * 10 + (*p - '0'); p++; }

    // precision
    int prec = -1;
    if (*p == '.') {
      p++;
      prec = 0;
      while (*p >= '0' && *p <= '9') { prec = prec * 10 + (*p - '0'); p++; }
    }

    char spec = *p ? *p : '\0';
    if (!spec) break;

    if (spec == 'd') {
      int v = va_arg(ap, int);
      char tmp[32];
      int neg = (v < 0);
      int len = itoa_dec_abs(v, tmp, sizeof(tmp));
      int content_len = len + (neg ? 1 : 0);
      total += pad_left(buf, cap, &pos, width, content_len);
      if (neg) total += outch(buf, cap, &pos, '-');
      total += outstr_n(buf, cap, &pos, tmp, len);

    } else if (spec == 's') {
      const char *s = va_arg(ap, const char *);
      if (!s) s = "(null)";
      int sl = strlen(s);
      int use = sl;
      if (prec >= 0 && prec < use) use = prec;
      total += pad_left(buf, cap, &pos, width, use);
      total += outstr_n(buf, cap, &pos, s, use);

    } else if (spec == 'c') {
      char c = (char)va_arg(ap, int);
      total += pad_left(buf, cap, &pos, width, 1);
      total += outch(buf, cap, &pos, c);

    } else if (spec == 'f') {
      double d = va_arg(ap, double);
      int pprec = (prec >= 0) ? prec : 6;
      char tmp[128];
      int len = dtoa_fixed(d, pprec, tmp, sizeof(tmp));
      total += pad_left(buf, cap, &pos, width, len);
      total += outstr_n(buf, cap, &pos, tmp, len);

    } else {
      // Unknown specifier: print literally to be conservative.
      total += outch(buf, cap, &pos, '%');
      total += outch(buf, cap, &pos, spec);
    }
  }

  // NUL-terminate if we have a buffer
  if (bufsz > 0) {
    if (pos <= cap) buf[pos] = 0;
    else buf[cap] = 0;
  }

  // Return the number of characters that *would* have been written (like snprintf)
  return total;
}

int snprintf(char *buf, int bufsz, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int r = vsnprintf(buf, bufsz, fmt, ap);
  va_end(ap);
  return r;
}

int sprintf(char *buf, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  // big cap to mimic unbounded sprintf; vsnprintf still NUL-terminates
  int r = vsnprintf(buf, 0x7fffffff, fmt, ap);
  va_end(ap);
  return r;
}

