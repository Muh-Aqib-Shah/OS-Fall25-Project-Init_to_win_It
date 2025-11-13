#include "test_mileSt2.h"
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"


int fputest(){
    printf("\n\n======================Fpu Tests======================\n");

  double sum = 0;
  for (double i = 1.0; i <= 50.0; i += 0.5)
    sum += i * i;
  
  int pid = fork();
  if(pid < 0) { printf("Error in fputest.."); return 0; }
  if (pid == 0) {
    double sum2 = 0;
    for (double i = 1.0; i <= 50.0; i += 0.5)
      sum2 += i * i * i;
    printf("Child: sum of cubes = %f\n", sum2);
    exit(0);
  } else {
    wait(0);
    for (double i = 50.5; i <= 100.0; i += 0.5)
      sum += i * i;
    printf("Parent: sum of squares = %f\n", sum);
  }

  return 1;
}

