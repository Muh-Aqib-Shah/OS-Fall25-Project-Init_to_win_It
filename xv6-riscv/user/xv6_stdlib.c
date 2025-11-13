// user/xv6_stdlib.c
#include "kernel/types.h"
#include "user/user.h"
#include "xv6_stdlib.h"

// -------------------------------------
// Helpers
// -------------------------------------
static inline int isspace_local(char c) {
  return c==' ' || c=='\t' || c=='\n' || c=='\r' || c=='\v' || c=='\f';
}
static inline int isdigit_local(char c) { return c >= '0' && c <= '9'; }
static inline int tolower_local(int c) { return (c >= 'A' && c <= 'Z') ? (c + 32) : c; }

// -------------------------------------
// calloc
// -------------------------------------
void* xv6_calloc(int nmemb, int size) {
  if (nmemb <= 0 || size <= 0) return 0;
  // naive overflow guard (best-effort)
  if (nmemb > 0x7fffffff / size) return 0;
  int bytes = nmemb * size;
  void* p = malloc(bytes);
  if (!p) return 0;
  memset(p, 0, bytes);
  return p;
}

// -------------------------------------
// bsearch (binary search on a sorted array)
// returns pointer to matching element or 0
// -------------------------------------
void* xv6_bsearch(const void* key, const void* base, int nmemb, int size,
                  int (*compar)(const void*, const void*)) {
  int lo = 0, hi = nmemb - 1;
  while (lo <= hi) {
    int mid = lo + ((hi - lo) >> 1);
    const char* p = (const char*)base + mid * size;
    int cmp = compar(key, p);
    if (cmp == 0) return (void*)p;
    if (cmp < 0) hi = mid - 1;
    else         lo = mid + 1;
  }
  return 0;
}

// -------------------------------------
// qsort (quicksort with Lomuto partition, tail recursion elimination)
// -------------------------------------
static void swap_bytes(char* a, char* b, int n) {
  while (n--) { char t = *a; *a++ = *b; *b++ = t; }
}

static int partition(char* base, int size, int lo, int hi,
                     int (*compar)(const void*, const void*)) {
  char* pivot = base + hi * size;
  int i = lo;
  for (int j = lo; j < hi; j++) {
    char* elem = base + j * size;
    if (compar(elem, pivot) <= 0) {
      if (i != j) swap_bytes(base + i * size, elem, size);
      i++;
    }
  }
  swap_bytes(base + i * size, pivot, size);
  return i;
}

void xv6_qsort(void* basev, int nmemb, int size,
               int (*compar)(const void*, const void*)) {
  if (nmemb <= 1 || size <= 0) return;
  char* base = (char*)basev;

  // manual stack to avoid deep recursion in weird inputs
  struct Frame { int lo, hi; } st[64];
  int sp = 0;
  st[sp++] = (struct Frame){0, nmemb - 1};

  while (sp) {
    struct Frame fr = st[--sp];
    int lo = fr.lo, hi = fr.hi;
    while (lo < hi) {
      int p = partition(base, size, lo, hi, compar);
      // sort smaller side first, tail-recurse the larger
      if (p - 1 - lo < hi - (p + 1)) {
        if (lo < p - 1) st[sp++] = (struct Frame){p + 1, hi}, hi = p - 1;
        else             lo = p + 1;
      } else {
        if (p + 1 < hi) st[sp++] = (struct Frame){lo, p - 1}, lo = p + 1;
        else             hi = p - 1;
      }
      if (sp >= 63) { /* extremely pathological; fall back */ break; }
    }
  }
}

// -------------------------------------
// atoi
// -------------------------------------
int xv6_atoi(const char* s) {
  if (!s) return 0;
  while (isspace_local(*s)) s++;
  int neg = 0;
  if (*s == '+' || *s == '-') { neg = (*s == '-'); s++; }
  int val = 0;
  while (isdigit_local(*s)) {
    val = val * 10 + (*s - '0');
    s++;
  }
  return neg ? -val : val;
}

// -------------------------------------
// atof   (whitespace, sign, integer, fraction, optional exponent e/E[+/-]d+)
// No libm: implement with simple pow10 scaling; good enough for tests.
// -------------------------------------
static float pow10_int(int e) {
  // fast small integer 10^e
  float r = 1.0f;
  int k = (e < 0) ? -e : e;
  float p = 10.0f;
  while (k) {
    if (k & 1) r *= p;
    p *= 10.0f;
    k >>= 1;
  }
  return (e < 0) ? (1.0f / r) : r;
}

float xv6_atof(const char* s) {
  if (!s) return 0.0f;

  while (isspace_local(*s)) s++;

  int neg = 0;
  if (*s == '+' || *s == '-') { neg = (*s == '-'); s++; }

  // integer part
  long long ip = 0;
  while (isdigit_local(*s)) { ip = ip * 10 + (*s - '0'); s++; }

  // fractional part
  float frac = 0.0f, scale = 1.0f;
  if (*s == '.') {
    s++;
    while (isdigit_local(*s)) {
      frac = frac * 10.0f + (float)(*s - '0');
      scale *= 10.0f;
      s++;
    }
  }
  float val = (float)ip + frac / scale;

  // exponent
  if (tolower_local(*s) == 'e') {
    s++;
    int eneg = 0;
    if (*s == '+' || *s == '-') { eneg = (*s == '-'); s++; }
    int expv = 0;
    while (isdigit_local(*s)) { expv = expv * 10 + (*s - '0'); s++; }
    if (eneg) expv = -expv;
    val *= pow10_int(expv);
  }

  return neg ? -val : val;
}

