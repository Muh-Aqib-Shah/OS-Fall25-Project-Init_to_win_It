// test_stdlib.c - Unit tests for standard library functions
#include "kernel/types.h"
#include "user/user.h"
#include "xv6_stdlib.h"
#include "test_mileSt2.h"

// Comparison functions for qsort and bsearch
int compare_ints(const void* a, const void* b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int compare_floats(const void* a, const void* b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

// Test result structure
typedef struct {
    int passed;
    int failed;
} TestResults;

// ============================================================================
// calloc Tests
// ============================================================================
void test_calloc(TestResults* results) {
    printf("\n=== Testing xv6_calloc ===\n");
    
    // Test 1: Normal allocation
    printf("Test 1: Normal allocation of 10 integers\n");
    int* arr = (int*)xv6_calloc(10, sizeof(int));
    if (arr != 0) {
        int all_zero = 1;
        for (int i = 0; i < 10; i++) {
            if (arr[i] != 0) {
                all_zero = 0;
                break;
            }
        }
        if (all_zero) {
            printf("  PASS: Memory allocated and initialized to zero\n");
            results->passed++;
        } else {
            printf("  FAIL: Memory not properly zeroed\n");
            results->failed++;
        }
        free(arr);
    } else {
        printf("  FAIL: Allocation failed\n");
        results->failed++;
    }
    
    // Test 2: Zero size allocation
    printf("Test 2: Zero size allocation\n");
    void* ptr = xv6_calloc(0, sizeof(int));
    if (ptr == 0) {
        printf("  PASS: Returns NULL for zero size\n");
        results->passed++;
    } else {
        printf("  FAIL: Should return NULL for zero size\n");
        results->failed++;
        free(ptr);
    }
    
    // Test 3: Large array allocation
    printf("Test 3: Allocate array of 100 floats\n");
    float* farr = (float*)xv6_calloc(100, sizeof(float));
    if (farr != 0) {
        int all_zero = 1;
        for (int i = 0; i < 100; i++) {
            if (farr[i] != 0.0f) {
                all_zero = 0;
                break;
            }
        }
        if (all_zero) {
            printf("  PASS: Large array allocated and zeroed\n");
            results->passed++;
        } else {
            printf("  FAIL: Large array not properly zeroed\n");
            results->failed++;
        }
        free(farr);
    } else {
        printf("  FAIL: Large allocation failed\n");
        results->failed++;
    }
}

// ============================================================================
// qsort Tests
// ============================================================================
void test_qsort(TestResults* results) {
    printf("\n=== Testing xv6_qsort ===\n");
    
    // Test 1: Sort integers
    printf("Test 1: Sort integer array\n");
    int int_arr[] = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    int expected[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = sizeof(int_arr) / sizeof(int_arr[0]);
    
    xv6_qsort(int_arr, n, sizeof(int), compare_ints);
    
    int sorted = 1;
    for (int i = 0; i < n; i++) {
        if (int_arr[i] != expected[i]) {
            sorted = 0;
            break;
        }
    }
    
    if (sorted) {
        printf("  PASS: Integer array sorted correctly\n");
        results->passed++;
    } else {
        printf("  FAIL: Integer array not sorted\n");
        printf("  Expected: ");
        for (int i = 0; i < n; i++) printf("%d ", expected[i]);
        printf("\n  Got: ");
        for (int i = 0; i < n; i++) printf("%d ", int_arr[i]);
        printf("\n");
        results->failed++;
    }
    
    // Test 2: Sort floats
    printf("Test 2: Sort float array\n");
    float float_arr[] = {3.14f, 1.5f, 2.7f, 0.5f, 4.2f};
    int fn = sizeof(float_arr) / sizeof(float_arr[0]);
    
    xv6_qsort(float_arr, fn, sizeof(float), compare_floats);
    
    int fsorted = 1;
    for (int i = 0; i < fn - 1; i++) {
        if (float_arr[i] > float_arr[i + 1]) {
            fsorted = 0;
            break;
        }
    }
    
    if (fsorted) {
        printf("  PASS: Float array sorted correctly\n");
        results->passed++;
    } else {
        printf("  FAIL: Float array not sorted\n");
        results->failed++;
    }
    
    // Test 3: Already sorted array
    printf("Test 3: Already sorted array\n");
    int sorted_arr[] = {1, 2, 3, 4, 5};
    int sn = sizeof(sorted_arr) / sizeof(sorted_arr[0]);
    
    xv6_qsort(sorted_arr, sn, sizeof(int), compare_ints);
    
    int still_sorted = 1;
    for (int i = 0; i < sn - 1; i++) {
        if (sorted_arr[i] > sorted_arr[i + 1]) {
            still_sorted = 0;
            break;
        }
    }
    
    if (still_sorted) {
        printf("  PASS: Already sorted array remains sorted\n");
        results->passed++;
    } else {
        printf("  FAIL: Already sorted array corrupted\n");
        results->failed++;
    }
}

// ============================================================================
// bsearch Tests
// ============================================================================
void test_bsearch(TestResults* results) {
    printf("\n=== Testing xv6_bsearch ===\n");
    
    int sorted_arr[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int n = sizeof(sorted_arr) / sizeof(sorted_arr[0]);
    
    // Test 1: Find existing element
    printf("Test 1: Find existing element (7)\n");
    int key1 = 7;
    int* result1 = (int*)xv6_bsearch(&key1, sorted_arr, n, sizeof(int), compare_ints);
    
    if (result1 != 0 && *result1 == 7) {
        printf("  PASS: Found element 7\n");
        results->passed++;
    } else {
        printf("  FAIL: Could not find element 7\n");
        results->failed++;
    }
    
    // Test 2: Search for non-existent element
    printf("Test 2: Search for non-existent element (8)\n");
    int key2 = 8;
    int* result2 = (int*)xv6_bsearch(&key2, sorted_arr, n, sizeof(int), compare_ints);
    
    if (result2 == 0) {
        printf("  PASS: Correctly returned NULL for non-existent element\n");
        results->passed++;
    } else {
        printf("  FAIL: Should return NULL for non-existent element\n");
        results->failed++;
    }
    
    // Test 3: Search in single-element array
    printf("Test 3: Search in single-element array\n");
    int single[] = {42};
    int key3 = 42;
    int* result3 = (int*)xv6_bsearch(&key3, single, 1, sizeof(int), compare_ints);
    
    if (result3 != 0 && *result3 == 42) {
        printf("  PASS: Found element in single-element array\n");
        results->passed++;
    } else {
        printf("  FAIL: Could not find element in single-element array\n");
        results->failed++;
    }
}

// ============================================================================
// atoi Tests
// ============================================================================
void test_atoi(TestResults* results) {
    printf("\n=== Testing xv6_atoi ===\n");
    
    // Test 1: Positive number
    printf("Test 1: Convert positive number \"123\"\n");
    int result1 = xv6_atoi("123");
    if (result1 == 123) {
        printf("  PASS: Got 123\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 123, got %d\n", result1);
        results->failed++;
    }
    
    // Test 2: Negative number
    printf("Test 2: Convert negative number \"-456\"\n");
    int result2 = xv6_atoi("-456");
    if (result2 == -456) {
        printf("  PASS: Got -456\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected -456, got %d\n", result2);
        results->failed++;
    }
    
    // Test 3: Leading whitespace
    printf("Test 3: Convert with leading whitespace \"  789\"\n");
    int result3 = xv6_atoi("  789");
    if (result3 == 789) {
        printf("  PASS: Got 789\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 789, got %d\n", result3);
        results->failed++;
    }
    
    // Test 4: Plus sign
    printf("Test 4: Convert with plus sign \"+100\"\n");
    int result4 = xv6_atoi("+100");
    if (result4 == 100) {
        printf("  PASS: Got 100\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 100, got %d\n", result4);
        results->failed++;
    }
}

// ============================================================================
// atof Tests
// ============================================================================
void test_atof(TestResults* results) {
    printf("\n=== Testing xv6_atof ===\n");
    
    // Test 1: Integer as float
    printf("Test 1: Convert integer \"42\"\n");
    float result1 = xv6_atof("42");
    if (result1 >= 41.9f && result1 <= 42.1f) {
        printf("  PASS: Got approximately 42.0\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 42.0, got %d\n", (int)result1);
        results->failed++;
    }
    
    // Test 2: Fractional number
    printf("Test 2: Convert fractional \"3.14159\"\n");
    float result2 = xv6_atof("3.14159");
    if (result2 >= 3.14f && result2 <= 3.15f) {
        printf("  PASS: Got approximately 3.14159\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 3.14159, got approximately %d.%d\n", 
               (int)result2, (int)((result2 - (int)result2) * 100));
        results->failed++;
    }
    
    // Test 3: Negative number
    printf("Test 3: Convert negative \"-2.5\"\n");
    float result3 = xv6_atof("-2.5");
    if (result3 >= -2.6f && result3 <= -2.4f) {
        printf("  PASS: Got approximately -2.5\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected -2.5\n");
        results->failed++;
    }
    
    // Test 4: Scientific notation
    printf("Test 4: Convert scientific notation \"1.5e2\"\n");
    float result4 = xv6_atof("1.5e2");
    if (result4 >= 149.0f && result4 <= 151.0f) {
        printf("  PASS: Got approximately 150.0\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 150.0, got %d\n", (int)result4);
        results->failed++;
    }
    
    // Test 5: Negative exponent
    printf("Test 5: Convert negative exponent \"5e-2\"\n");
    float result5 = xv6_atof("5e-2");
    if (result5 >= 0.04f && result5 <= 0.06f) {
        printf("  PASS: Got approximately 0.05\n");
        results->passed++;
    } else {
        printf("  FAIL: Expected 0.05\n");
        results->failed++;
    }
}

int test_stdlib(){
      printf("\n");
    printf("========================================\n");
    printf("  Standard Library Tests (xv6_stdlib)\n");
    printf("========================================\n");
    
    TestResults results = {0, 0};
    
    test_calloc(&results);
    test_qsort(&results);
    test_bsearch(&results);
    test_atoi(&results);
    test_atof(&results);
    
    printf("\n========================================\n");
    printf("Stdlib Test Summary\n");
    printf("========================================\n");
    printf("Total Tests:  %d\n", results.passed + results.failed);
    printf("Passed:       %d\n", results.passed);
    printf("Failed:       %d\n", results.failed);
    printf("Success Rate: %d%%\n", 
           (results.passed * 100) / (results.passed + results.failed));
    printf("========================================\n\n");
    
    return results.passed;
}
