#include "test_mileSt2.h"
#include "kernel/types.h"
#include "user/user.h"

int main(){

    int total_tests = 407;
    int passed = 0;
    
    passed += fputest();
    passed += stringtests();
    passed += test_maths();
    passed += test_stdlib();
    
    printf("\n========================================\n");
    printf("Final Test Report\n");
    printf("========================================\n");
    printf("Total Tests:  %d\n", total_tests);
    printf("Passed:       %d\n", passed);
    printf("Failed:       %d\n", total_tests - passed);
    printf("Success Rate: %d%%\n", 
           (passed * 100) / (total_tests));
    printf("========================================\n\n");
    
   
    exit(0);

}
