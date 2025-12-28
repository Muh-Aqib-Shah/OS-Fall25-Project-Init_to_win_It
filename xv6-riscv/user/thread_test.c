#include "kernel/types.h"
#include "user/user.h"

void thread_func(void *arg) {
  int id = (int)(uint64)arg;
  printf("Thread %d: Hello!\n", id);
  for(int i = 0; i < 3; i++){
    printf("Thread %d: iteration %d\n", id, i);
    pause(10);
  }
  printf("Thread %d: Goodbye!\n", id);
  thread_exit();
  printf("THREAD EXITED SUCCESSFULLY");
}

int main(void) {
  printf("\nCreating thread 1...\n");
  
  int t1 = thread_create(thread_func, (void*)1);
  printf("\nCreating thread 2...\n");
  int t2 = thread_create(thread_func, (void*)2);

  
  printf("Waiting for threads...\n");
  thread_join(t1);
  thread_join(t2);
  
  printf("All threads done!\n");
  exit(0);
}
