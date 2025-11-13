// xv6_stdlib.h - Standard library functions header for xv6
#ifndef XV6_STDLIB_H
#define XV6_STDLIB_H

// Memory allocation
void* xv6_calloc(int nmemb, int size);

// Searching and sorting
void* xv6_bsearch(const void* key, const void* base, int nmemb, int size,
                   int (*compar)(const void*, const void*));
void xv6_qsort(void* base, int nmemb, int size,
               int (*compar)(const void*, const void*));

// String to number conversion
int xv6_atoi(const char* str);
float xv6_atof(const char* str);

#endif // XV6_STDLIB_H
