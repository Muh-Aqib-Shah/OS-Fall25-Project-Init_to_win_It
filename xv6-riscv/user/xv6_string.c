// user/xv6_string.c - Complete String and Formatting Library
#include "xv6_string.h"

int xv6_itoa(int value, char* buf, int base) {
    int i = 0, is_neg = 0;
    
    // Handle INT_MIN special case
    if (value == -2147483648 && base == 10) {
        const char* min_str = "-2147483648";
        int len = 0;
        while (min_str[len]) { buf[len] = min_str[len]; len++; }
        buf[len] = '\0';
        return len;
    }
    
    if (value == 0) { buf[i++] = '0'; buf[i] = '\0'; return i; }
    if (value < 0 && base == 10) { is_neg = 1; value = -value; }
    
    char temp[32];
    while (value) { 
        int rem = value % base;
        temp[i++] = (rem > 9) ? (rem - 10) + 'a' : rem + '0';
        value /= base; 
    }
    int len = 0;
    if (is_neg) buf[len++] = '-';
    while (i > 0) buf[len++] = temp[--i];
    buf[len] = '\0';
    return len;
}

int xv6_ftoa(double value, char* buf, int precision) {
    int i = 0;
    if (value < 0) { buf[i++] = '-'; value = -value; }
    
    long long int_part = (long long)value;
    char temp[32];
    int temp_len = 0;
    
    // Convert integer part
    if (int_part == 0) {
        temp[temp_len++] = '0';
    } else {
        long long n = int_part;
        while (n > 0) {
            temp[temp_len++] = '0' + (n % 10);
            n /= 10;
        }
    }
    
    // Reverse and copy integer part
    while (temp_len > 0) {
        buf[i++] = temp[--temp_len];
    }
    
    buf[i++] = '.';
    double frac = value - (double)int_part;
    for (int p = 0; p < precision; p++) {
        frac *= 10.0;
        int digit = (int)frac;
        buf[i++] = '0' + digit;
        frac -= digit;
    }
    buf[i] = '\0';
    return i;
}
// ============================================================
// Basic Memory and String Functions
// ============================================================
void* xv6_memcpy(void* dest, const void* src, int n) {
    char* d = (char*)dest;
    const char* s = (const char*)src;
    for (int i = 0; i < n; i++) d[i] = s[i];
    return dest;
}

void* xv6_memset(void* s, int c, int n) {
    char* p = (char*)s;
    for (int i = 0; i < n; i++) p[i] = (char)c;
    return s;
}

int xv6_strcmp(const char* s1, const char* s2) {
    while (*s1 && (*s1 == *s2)) { s1++; s2++; }
    return *(unsigned char*)s1 - *(unsigned char*)s2;
}

int xv6_strlen(const char* s) {
    int len = 0;
    while (s[len] != '\0') len++;
    return len;
}

char* xv6_strcpy(char* dest, const char* src) {
    int i = 0;
    while (src[i] != '\0') { dest[i] = src[i]; i++; }
    dest[i] = '\0';
    return dest;
}

char* xv6_strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d) d++;          // move to end of dst
    while ((*d++ = *src++)); // copy src including null terminator
    return dst;
}

// ============================================================
// Character classification helpers
// ============================================================
int xv6_isspace(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
int xv6_isprint(int c) { return c >= 0x20 && c <= 0x7E; }
int xv6_isdigit(int c) { return c >= '0' && c <= '9'; }


// ============================================================
// Simplified sprintf Implementations
// ============================================================

// %d
int xv6_sprintf_d(char* buf, const char* fmt, int arg1) {
    int i = 0, j = 0;
    while (fmt[i]) {
        if (fmt[i] == '%' && fmt[i+1] == 'd') {
            char temp[32];
            int len = xv6_itoa(arg1, temp, 10);
            for (int k = 0; k < len; k++) buf[j++] = temp[k];
            i += 2;
        } else if (fmt[i] == '%' && fmt[i+1] == '%') {
            buf[j++] = '%'; i += 2;
        } else buf[j++] = fmt[i++];
    }
    buf[j] = '\0';
    return j;
}

// %s
int xv6_sprintf_s(char* buf, const char* fmt, const char* arg1) {
    int i = 0, j = 0;
    while (fmt[i]) {
        if (fmt[i] == '%' && fmt[i+1] == 's') {
            const char* s = arg1;
            while (*s) buf[j++] = *s++;
            i += 2;
        } else if (fmt[i] == '%' && fmt[i+1] == '%') {
            buf[j++] = '%'; i += 2;
        } else buf[j++] = fmt[i++];
    }
    buf[j] = '\0';
    return j;
}

// %f with precision (%.3f)
int xv6_sprintf(char* buf, const char* fmt, double arg1) {
    int i = 0, j = 0;
    while (fmt[i]) {
        if (fmt[i] == '%' && fmt[i+1] == '.' && xv6_isdigit(fmt[i+2]) && fmt[i+3] == 'f') {
            int precision = fmt[i+2] - '0';
            char temp[64];
            int len = xv6_ftoa(arg1, temp, precision);
            for (int k = 0; k < len; k++) buf[j++] = temp[k];
            i += 4;
        } else if (fmt[i] == '%' && fmt[i+1] == 'f') {
            char temp[64];
            int len = xv6_ftoa(arg1, temp, 6);
            for (int k = 0; k < len; k++) buf[j++] = temp[k];
            i += 2;
        } else if (fmt[i] == '%' && fmt[i+1] == '%') {
            buf[j++] = '%'; i += 2;
        } else buf[j++] = fmt[i++];
    }
    buf[j] = '\0';
    return j;
}

// %c
int xv6_sprintf_c(char* buf, const char* fmt, char arg1) {
    int i = 0, j = 0;
    while (fmt[i]) {
        if (fmt[i] == '%' && fmt[i+1] == 'c') { buf[j++] = arg1; i += 2; }
        else if (fmt[i] == '%' && fmt[i+1] == '%') { buf[j++] = '%'; i += 2; }
        else buf[j++] = fmt[i++];
    }
    buf[j] = '\0';
    return j;
}


// ============================================================
// sscanf Implementations
// ============================================================

static const char* skip_ws(const char* str) {
    while (*str && xv6_isspace(*str)) str++;
    return str;
}

static const char* parse_int(const char* str, int* result) {
    int sign = 1, val = 0;
    str = skip_ws(str);
    if (*str == '-') { sign = -1; str++; }
    else if (*str == '+') str++;
    if (!xv6_isdigit(*str)) return 0;
    while (xv6_isdigit(*str)) { val = val * 10 + (*str - '0'); str++; }
    *result = sign * val;
    return str;
}

static const char* parse_float(const char* str, double* result) {
    double sign = 1.0, val = 0.0, frac = 0.0, div = 1.0;  // No 'f' suffix
    str = skip_ws(str);
    if (*str == '-') { sign = -1.0; str++; }  // -1.0, not -1.0f
    else if (*str == '+') str++;
    while (xv6_isdigit(*str)) { val = val * 10.0 + (*str - '0'); str++; }
    if (*str == '.') {
        str++;
        while (xv6_isdigit(*str)) { 
            frac = frac * 10.0 + (*str - '0'); 
            div *= 10.0; 
            str++; 
        }
    }
    *result = sign * (val + frac / div);
    return str;
}

int xv6_sscanf_d(const char* buf, const char* fmt, int* arg1) {
    const char* b = buf; const char* f = fmt;
    while (*f && *b) {
        if (*f == '%' && *(f+1) == 'd') {
            b = parse_int(b, arg1); if (!b) return 0; f += 2;
        } else if (xv6_isspace(*f)) { b = skip_ws(b); f++; }
        else if (*f == *b) { f++; b++; }
        else return 0;
    }
    return 1;
}

int xv6_sscanf_f(const char* buf, const char* fmt, double* arg1) {
    const char* b = buf; const char* f = fmt;
    while (*f && *b) {
        if (*f == '%' && *(f+1) == 'f') {
            b = parse_float(b, arg1); if (!b) return 0; f += 2;
        } else if (xv6_isspace(*f)) { b = skip_ws(b); f++; }
        else if (*f == *b) { f++; b++; }
        else return 0;
    }
    return 1;
}

int xv6_sscanf_s(const char* buf, const char* fmt, char* arg1, int maxlen) {
    const char* b = buf; const char* f = fmt;
    while (*f && *b) {
        if (*f == '%' && *(f+1) == 's') {
            b = skip_ws(b);
            int i = 0;
            while (*b && !xv6_isspace(*b) && i < maxlen - 1) arg1[i++] = *b++;
            arg1[i] = '\0';
            f += 2;
        } else if (xv6_isspace(*f)) { b = skip_ws(b); f++; }
        else if (*f == *b) { f++; b++; }
        else return 0;
    }
    return 1;
}
