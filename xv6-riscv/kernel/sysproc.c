#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"
#include "vm.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  kexit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return kfork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return kwait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int t;
  int n;

  argint(0, &n);
  argint(1, &t);
  addr = myproc()->sz;

  if(t == SBRK_EAGER || n < 0) {
    if(growproc(n) < 0) {
      return -1;
    }
  } else {
    // Lazily allocate memory for this process: increase its memory
    // size but don't allocate memory. If the processes uses the
    // memory, vmfault() will allocate it.
    if(addr + n < addr)
      return -1;
    if(addr + n > TRAPFRAME)
      return -1;
    myproc()->sz += n;
  }
  return addr;
}

uint64
sys_pause(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  if(n < 0)
    n = 0;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kkill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

uint64
sys_getcycles(void){
  return rd_cycle();
}

uint64
sys_gettime(void){
  return rd_time();
}

uint64
sys_getinstret(void)
{
  return rd_instret();
}

// thread functions here 
uint64
sys_thread_create(void)
{
  struct proc *p = myproc();
  uint64 start_routine = p->trapframe->a0;
  uint64 arg = p->trapframe->a1;

  argaddr(0, &start_routine);
  argaddr(1, &arg);

  return thread_create(start_routine, arg);
}

uint64
sys_thread_join(void)
{
  int tid = myproc()->trapframe->a0;
  argint(0, &tid);
  if(tid < 0)
    return -1;
  
  return thread_join(tid);
}

uint64
sys_thread_exit(void)
{
  thread_exit();
  return 0; // never reached
}

// mutex functions down here :)

// Initialize a mutex
uint64
sys_mutex_init(void)
{
  struct proc *p = myproc();
  uint64 mutex_ptr = p->trapframe->a0;

  if(mutex_ptr == 0 || mutex_ptr >= p->sz)
    return -1;

  mutex_t m;
  m.locked = 0;
  m.owner_tid = -1;

  if(copyout(p->pagetable, mutex_ptr, (char*)&m, sizeof(m)) < 0)
    return -1;

  return 0;
}

// Lock a mutex (spinlock version)
uint64
sys_mutex_lock(void)
{
  struct proc *p = myproc();
  uint64 mutex_ptr = p->trapframe->a0;

  if(mutex_ptr == 0 || mutex_ptr >= p->sz)
    return -1;

  mutex_t m;
  for(;;){
    if(copyin(p->pagetable, (char*)&m, mutex_ptr, sizeof(m)) < 0)
      return -1;

    if(m.locked == 0){
      m.locked = 1;
      m.owner_tid = p->thread_id;

      if(copyout(p->pagetable, mutex_ptr, (char*)&m, sizeof(m)) < 0)
        return -1;

      if(copyin(p->pagetable, (char*)&m, mutex_ptr, sizeof(m)) < 0)
        return -1;

      if(m.locked == 1 && m.owner_tid == p->thread_id){
        return 0;
      }
    }
    yield();
  }
}

// Unlock a mutex
uint64
sys_mutex_unlock(void)
{
  struct proc *p = myproc();
  uint64 mutex_ptr = p->trapframe->a0;

  if(mutex_ptr == 0 || mutex_ptr >= p->sz)
    return -1;

  mutex_t m;
  m.locked = 0;
  m.owner_tid = -1;

  if(copyout(p->pagetable, mutex_ptr, (char*)&m, sizeof(m)) < 0)
    return -1;

  return 0;
}
