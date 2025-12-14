#include "kernel/types.h"
#include "user/user.h"

int counter = 0;
mutex_t lock;

void increment(void *arg) {
  for(int i = 0; i < 1000; i++){
    mutex_lock(&lock);
    counter++;
    mutex_unlock(&lock);
  }
  thread_exit();
}

int main(void) {
  mutex_init(&lock);
  
  int t1 = thread_create(increment, 0);
  int t2 = thread_create(increment, 0);
  
  thread_join(t1);
  thread_join(t2);
  
  printf("Counter: %d (should be 2000)\n", counter);
  exit(0);
}
