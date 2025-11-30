// user/xv6_string.h - String library header
#ifndef XV6_STRING_H
#define XV6_STRING_H

// ============================================================
// Memory functions
// ============================================================
void* xv6_memcpy(void* dest, const void* src, int n);
void* xv6_memset(void* s, int c, int n);

// ============================================================
// String comparison and manipulation
// ============================================================
int xv6_strcmp(const char* s1, const char* s2);
int xv6_strlen(const char* s);
char* xv6_strcpy(char* dest, const char* src);
char* xv6_strcat(char *dst, const char *src);

// ============================================================
// Character classification
// ============================================================
int xv6_isspace(int c);
int xv6_isprint(int c);
int xv6_isdigit(int c);

// ============================================================
// Conversion helpers
// ============================================================

// Convert integer to ASCII string
// base: 10 for decimal, 16 for hex, etc.
// Returns: length of resulting string
int xv6_itoa(int value, char* buf, int base);

// Convert double to ASCII string with specified precision
// precision: number of digits after decimal point
// Returns: length of resulting string
int xv6_ftoa(double value, char* buf, int precision);


// Single format specifiers
int xv6_sprintf_d(char* buf, const char* fmt, int arg1);
int xv6_sprint_s(char* buf, const char* fmt, const char* arg1);
int xv6_sprintf(char* buf, const char* fmt, double arg1);
int xv6_sprintf_c(char* buf, const char* fmt, char arg1);


// Single format specifiers
int xv6_sscanf_d(const char* buf, const char* fmt, int* arg1);
int xv6_sscanf_f(const char* buf, const char* fmt, double* arg1);
int xv6_sscanf_s(const char* buf, const char* fmt, char* arg1, int maxlen);


#endif // XV6_STRING_H
