#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(void)
{
  printf("Starting FPU test...\n");

  double sum = 0;
  for (double i = 1.0; i <= 50.0; i += 0.5)
    sum += i * i;
  
  int pid = fork();
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

  exit(0);
}

