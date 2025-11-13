// user/stringtests.c
#include "test_mileSt2.h"
#include "kernel/types.h"
#include "kernel/fcntl.h"
#include "user/user.h"

// your tiny libc declarations
#include "../llama2c/mini_lib.h"


// simple test helpers (xv6 has no assert)
static int pass = 0;
static int fails = 0;
static void ok(const char *name)  { printf("  [OK] %s\n", name); ++pass; }
static void fail(const char *name){ printf("  [FAIL] %s\n", name); fails++; }

static void expect_int(const char *name, int got, int want) {
  if (got == want) ok(name); else { printf("    got=%d want=%d\n", got, want); fail(name); }
}
static void expect_str(const char *name, const char *got, const char *want) {
  if (strcmp(got, want) == 0) ok(name);
  else { printf("    got=\"%s\" want=\"%s\"\n", got, want); fail(name); }
}
static void expect_mem(const char *name, const void *a, const void *b, int n) {
  const unsigned char *pa=a, *pb=b;
  int eq=1; for (int i=0;i<n;i++) if (pa[i]!=pb[i]) {eq=0;break;}
  if (eq) ok(name); else fail(name);
}

int stringtests(){
   printf("\n\n======================String Tests======================\n");

  // --- strlen/strcpy/strcmp ---
  {
    char dst[64];
    strcpy(dst, "hello");
    expect_int("strlen(\"hello\")", (int)strlen(dst), 5);
    expect_int("strcmp eq", strcmp(dst, "hello"), 0);
    expect_int("strcmp lt", strcmp("abc", "abd") < 0, 1);
    expect_int("strcmp gt", strcmp("abd", "abc") > 0, 1);
  }

  // --- memset/memcpy ---
  {
    char a[16]; memset(a, 0xAA, sizeof(a));
    int allAA=1; for (int i=0;i<16;i++) if ((unsigned char)a[i]!=0xAA) {allAA=0;break;}
    expect_int("memset 0xAA", allAA, 1);

    char src[8]; for (int i=0;i<8;i++) src[i]=(char)(i+1);
    char b[8];  memset(b, 0, sizeof(b));
    memcpy(b, src, 8);
    expect_mem("memcpy basic", b, src, 8);
  }

  // --- isprint / isspace ---
  {
    expect_int("isprint('A')", isprint('A'), 1);
    expect_int("isprint('\\n')", isprint('\n'), 0);
    expect_int("isspace(' ')", isspace(' '), 1);
    expect_int("isspace('\\t')", isspace('\t'), 1);
    expect_int("isspace('X')", isspace('X'), 0);
  }

  // --- sprintf/snprintf integer/string/char/percent ---
  {
    char buf[64];

    sprintf(buf, "n=%d", 42);
    expect_str("sprintf %d", buf, "n=42");

    sprintf(buf, "s=%s c=%c %%", "hi", 'Z');
    expect_str("sprintf %s %c %%", buf, "s=hi c=Z %");

    // field width / padding for ints (if you implemented it; otherwise relax this)
    // Many minimal printers only support precision for %f. Still safe to check a simple width:
    // e.g., "%4d" expected "  42" (two spaces, then 42)
    snprintf(buf, sizeof(buf), "%4d", 42);
    expect_str("snprintf %4d", buf, "  42");
  }

  // --- %f with precision/width (requires your float formatting to be implemented) ---
  {
    char buf[64];

    // basic float
    snprintf(buf, sizeof(buf), "%f", 3.0f);
    // Many minimal implementations print "3.000000"; accept either "3.000000" or "3"
    if (strcmp(buf, "3.000000")==0 || strcmp(buf,"3")==0) ok("%f basic");
    else { printf("    got=\"%s\" want \"3.000000\"\n", buf); fail("%f basic"); }

    // precision
    snprintf(buf, sizeof(buf), "%.3f", 3.14159f);
    expect_str("%.3f", buf, "3.142");

    // width + precision
    snprintf(buf, sizeof(buf), "%6.2f", 7.0f); // width 6: "  7.00"
    expect_str("%6.2f", buf, "  7.00");

    // negative value with precision
    snprintf(buf, sizeof(buf), "%.4f", -1.23456f);
    expect_str("%.4f negative", buf, "-1.2346");
  }

  // --- snprintf bounds (truncation + NUL-termination) ---
  {
    char small[6]; // can hold at most 5 chars + '\0'
    int n = snprintf(small, sizeof(small), "abcdef");
    // C standard: return value is number of chars that *would* have been written (6), excluding NUL
    // buffer contains "abcde"
    expect_int("snprintf ret (trunc)", n, 6);
    expect_str("snprintf trunc content", small, "abcde");
  }

  // --- sscanf: ints, floats, strings, chars ---
  {
    int ii=0; float ff=0.0f; char ss[16]; char cc=0;
    int n = sscanf("42 -3.14 hello Z", "%d %f %s %c", &ii, &ff, ss, &cc);
    expect_int("sscanf count", n, 4);
    expect_int("sscanf int", ii, 42);

    // compare float by reformatting to fixed precision
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", ff);
    expect_str("sscanf float", buf, "-3.14");

    expect_str("sscanf string", ss, "hello");
    expect_int("sscanf char", cc, 'Z');
  }

  // --- tokenization-ish: combine sprintf+strlen+strcmp ---
  {
    char tok[64];
    const char *a = "foo";
    const char *b = "bar";
    sprintf(tok, "%s_%s_%d", a, b, 7);
    expect_str("combo format", tok, "foo_bar_7");
    expect_int("combo strlen", (int)strlen(tok), 9);
    expect_int("combo strcmp", strcmp(tok, "foo_bar_7"), 0);
  }

  if (fails == 0) {
    printf("=== ALL TESTS PASSED ===\n");
  } else {
    printf("=== %d TEST(S) FAILED ===\n", fails);
  }
  return pass;
}

