// user/mathtests.c
#include "user.h"
#include "xv6_maths.h"

// If you generate big tables, paste them into maths_testdata.h
// and (optionally) define HAS_GEN_DATA 1 below.
#define HAS_GEN_DATA 0
#if HAS_GEN_DATA
#include "maths_testdata.h"
#endif

static float fabsf_local(float x){ return x<0? -x : x; }

static void chk(const char *name, float got, float want, float tol, int *ok, int *fail){
  float err = fabsf_local(got - want);
  if (err <= tol) (*ok)++;
  else {
    (*fail)++;
    printf("  [FAIL] %s got=%.8f want=%.8f err=%.3e\n", name, got, want, err);
  }
}

int
main(void)
{
  int ok=0, fail=0;
  const float T=1e-5f;

  // Quick sanity set (immediate run)
  chk("fabs", xv6m_fabsf(-3.5f), 3.5f, T, &ok,&fail);
  chk("sqrt(0)", xv6m_sqrtf(0.f), 0.f, T, &ok,&fail);
  chk("sqrt(9)", xv6m_sqrtf(9.f), 3.f, T, &ok,&fail);
  chk("exp(0)", xv6m_expf(0.f), 1.f, 3e-6f, &ok,&fail);
  chk("exp(1)", xv6m_expf(1.f), 2.7182817f, 5e-5f, &ok,&fail);
  chk("log(e)", xv6m_logf(2.7182817f), 1.f, 1e-4f, &ok,&fail);
  chk("pow(2,10)", xv6m_powf(2.f,10.f), 1024.f, T, &ok,&fail);
  chk("pow(9,0.5)", xv6m_powf(9.f,0.5f), 3.f, 3e-5f, &ok,&fail);
  chk("sin(pi/6)", xv6m_sinf(0.5235988f), 0.5f, 1e-5f, &ok,&fail);
  chk("cos(pi/3)", xv6m_cosf(1.0471976f), 0.5f, 1e-5f, &ok,&fail);

#if HAS_GEN_DATA
  // Example sweep for generated data (arrays defined in maths_testdata.h)
  for (int i=0;i<SQRT_N;i++)  chk("sqrt_gen", xv6m_sqrtf(sqrt_in[i]), sqrt_out[i], 1e-5f, &ok,&fail);
  for (int i=0;i<EXP_N;i++)   chk("exp_gen",  xv6m_expf(exp_in[i]),  exp_out[i],  1e-5f, &ok,&fail);
  for (int i=0;i<POW_N;i++)   chk("pow_gen",  xv6m_powf(pow_x[i], pow_y[i]), pow_out[i], 1e-4f, &ok,&fail);
  for (int i=0;i<SIN_N;i++)   chk("sin_gen",  xv6m_sinf(sin_in[i]),  sin_out[i],  1e-5f, &ok,&fail);
  for (int i=0;i<COS_N;i++)   chk("cos_gen",  xv6m_cosf(cos_in[i]),  cos_out[i],  1e-5f, &ok,&fail);
  for (int i=0;i<FABS_N;i++)  chk("fabs_gen", xv6m_fabsf(fabs_in[i]),fabs_out[i], 0.0f,  &ok,&fail);
#endif

  printf("=== MATH TESTS: %d ok, %d fail ===\n", ok, fail);
  exit(0);
}

