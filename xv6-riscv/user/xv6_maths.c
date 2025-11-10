// user/xv6_maths.c
#include "user/user.h"
#include "xv6_maths.h"

// user/xv6_maths.c
// Self-contained math routines for xv6 user space (no libm).
// Implements: xv6m_fabsf, xv6m_sqrtf, xv6m_expf, xv6m_logf, xv6m_powf, xv6m_sinf, xv6m_cosf
// Accuracy target: ~1e-5 (single-precision-safe).

// NOTE: We intentionally avoid including standard headers.
// If your project expects user/user.h, you can include it,
// but these functions do not rely on external prototypes.

// ----- constants -----
#ifndef XV6_MATH_CONSTS
#define XV6_MATH_CONSTS
#define XV6_PI        3.14159265358979323846f
#define XV6_PI_2      1.57079632679489661923f
#define XV6_PI_4      0.78539816339744830962f
#define XV6_TWO_PI    6.28318530717958647692f
#define XV6_LN2       0.69314718055994530942f
#define XV6_INV_LN2   1.44269504088896340736f  // 1/ln(2)
#endif

// ----- helpers to manufacture NaN / +/-INF without libm -----
static float xv6m_nan(void) {
  volatile float z = 0.0f;
  return z / z; // NaN
}
static float xv6m_pos_inf(void) {
  volatile float z = 0.0f;
  return 1.0f / z; // +INF
}
static float xv6m_neg_inf(void) {
  volatile float z = 0.0f;
  return -1.0f / z; // -INF
}

// ----- fabs -----
float xv6m_fabsf(float x) {
  return (x >= 0.0f) ? x : -x;
}

// ----- sqrt (Newton-Raphson) -----
// Handles x < 0 -> NaN; x=0 -> 0.
// A few Newton steps with a guarded initial guess works well for float.
float xv6m_sqrtf(float x) {
  if (x < 0.0f) return xv6m_nan();
  if (x == 0.0f) return 0.0f;

  // Initial guess: scale-based crude estimate.
  float guess = x;
  if (x > 1.0f)      guess = x * 0.5f;
  else if (x > 0.25f) guess = 0.75f;
  else                guess = 0.5f;

  // 6 iterations are ample for float accuracy
  for (int i = 0; i < 6; i++) {
    guess = 0.5f * (guess + x / guess);
  }
  return guess;
}

// ----- exp (range-reduced Maclaurin on [-ln2/2, ln2/2]) -----
// exp(x) = 2^k * exp(r), k = round(x / ln2), r in ~[-0.3466, 0.3466]
float xv6m_expf(float x) {
  // Rough under/overflow clamps for float
  if (x > 88.0f)  return xv6m_pos_inf(); // overflow
  if (x < -100.0f) return 0.0f;          // underflow

  // Compute k and r
  float kf = x * XV6_INV_LN2;
  int k = (int)(kf + (kf >= 0 ? 0.5f : -0.5f)); // round to nearest int
  float r = x - k * XV6_LN2;

  // exp(r) via 5th-order Maclaurin: 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5!
  float r2 = r * r;
  float r3 = r2 * r;
  float r4 = r3 * r;
  float r5 = r4 * r;
  float er = 1.0f + r + r2 * 0.5f + r3 * (1.0f/6.0f) + r4 * (1.0f/24.0f) + r5 * (1.0f/120.0f);

  // 2^k by simple scaling (k is small for practical ranges; this is fine)
  float two_to_k = 1.0f;
  if (k > 0)       { for (int i = 0; i < k; i++) two_to_k *= 2.0f; }
  else if (k < 0)  { for (int i = 0; i < -k; i++) two_to_k *= 0.5f; }

  return er * two_to_k;
}

// ----- log (range reduction to [1/sqrt(2), sqrt(2)] and atanh series) -----
// x = 2^k * r, r∈[~0.7071, ~1.4142]; y=(r-1)/(r+1); ln r = 2*(y + y^3/3 + y^5/5 + ...)
float xv6m_logf(float x) {
  if (x < 0.0f)  return xv6m_nan();
  if (x == 0.0f) return xv6m_neg_inf();
  if (x == 1.0f) return 0.0f;

  int k = 0;
  float r = x;
  const float SQRT2      = 1.4142135623730950488f;
  const float INV_SQRT2  = 0.70710678118654752440f;

  while (r > SQRT2)     { r *= 0.5f; k++; }
  while (r < INV_SQRT2) { r *= 2.0f; k--; }

  float y  = (r - 1.0f) / (r + 1.0f);
  float y2 = y * y;

  float y3 = y  * y2;
  float y5 = y3 * y2;
  float y7 = y5 * y2;
  float y9 = y7 * y2;

  float series = y
               + y3 * (1.0f/3.0f)
               + y5 * (1.0f/5.0f)
               + y7 * (1.0f/7.0f)
               + y9 * (1.0f/9.0f);

  float ln_r = 2.0f * series;
  return k * XV6_LN2 + ln_r;
}

// ----- pow -----
// Handles special cases: 0^0 -> 1, a^0 -> 1, 1^x -> 1,
// negative base with integer exponent (sign handling),
// negative base with non-integer exponent -> NaN.
float xv6m_powf(float a, float b) {
  // Specials
  if (b == 0.0f) return 1.0f;
  if (a == 1.0f) return 1.0f;
  if (a == 0.0f) {
    if (b > 0.0f) return 0.0f;
    if (b == 0.0f) return 1.0f;     // 0^0 -> 1 by spec here
    return xv6m_pos_inf();          // 0^negative -> +INF
  }

  // Negative base?
  if (a < 0.0f) {
    // Is b an integer?
    int bi = (int)(b + (b >= 0 ? 0.5f : -0.5f));
    if ((float)bi == b) {
      // integer exponent: sign depends on parity
      float ax = xv6m_fabsf(a);
      // compute ax^bi using exp/log for range; for small |bi| we could do fast pow, but this suffices
      float res = xv6m_expf(b * xv6m_logf(ax));
      if ((bi & 1) != 0) res = -res;
      return res;
    } else {
      return xv6m_nan(); // negative base, non-integer exponent undefined in reals
    }
  }

  // Positive base general case
  return xv6m_expf(b * xv6m_logf(a));
}

// ----- trig helpers -----
// Reduce x near multiples of PI/2: n = round(x/(PI/2)), t = x - n*(PI/2) ∈ ~[-PI/4, PI/4]
static float xv6m_reduce_pi_over_4(float x, int *q_mod4) {
  float inv_half_pi = 1.0f / XV6_PI_2;
  int n = (int)(x * inv_half_pi + (x >= 0 ? 0.5f : -0.5f));
  *q_mod4 = n & 3;
  return x - n * XV6_PI_2;
}

// Cosine via small-interval polynomial (to ~1e-7 on [-pi/4, pi/4])
static void xv6m_sin_cos_core(float t, float *s_out, float *c_out) {
  float t2 = t * t;

  // sin t ≈ t - t^3/6 + t^5/120 - t^7/5040
  float s = t * (1.0f
          - t2 * (1.0f/6.0f
          - t2 * (1.0f/120.0f
          - t2 * (1.0f/5040.0f))));

  // cos t ≈ 1 - t^2/2 + t^4/24 - t^6/720 + t^8/40320
  float c = 1.0f
          - t2 * (0.5f
          - t2 * (1.0f/24.0f
          - t2 * (1.0f/720.0f
          - t2 * (1.0f/40320.0f))));

  *s_out = s;
  *c_out = c;
}

float xv6m_cosf(float x) {
  if (x != x) return x; // NaN passthrough
  int q;
  float t = xv6m_reduce_pi_over_4(x, &q);
  float s, c; xv6m_sin_cos_core(t, &s, &c);
  // n mod 4: 0->cos(t), 1->-sin(t), 2->-cos(t), 3->sin(t)
  switch (q) {
    case 0: return c;
    case 1: return -s;
    case 2: return -c;
    default: return s;
  }
}

float xv6m_sinf(float x) {
  if (x != x) return x; // NaN passthrough
  int q;
  float t = xv6m_reduce_pi_over_4(x, &q);
  float s, c; xv6m_sin_cos_core(t, &s, &c);
  // n mod 4: 0->sin(t), 1->cos(t), 2->-sin(t), 3->-cos(t)
  switch (q) {
    case 0: return s;
    case 1: return c;
    case 2: return -s;
    default: return -c;
  }
}

