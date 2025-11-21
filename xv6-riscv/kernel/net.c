#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#include "fs.h"
#include "sleeplock.h"
#include "file.h"
#include "net.h"

#define MAX_QUEUED_PACKETS 16
#define MAX_PORTS 32

// xv6's ethernet and IP addresses
static uint8 local_mac[ETHADDR_LEN] = { 0x52, 0x54, 0x00, 0x12, 0x34, 0x56 };
static uint32 local_ip = MAKE_IP_ADDR(10, 0, 2, 15);

// qemu host's ethernet address.
static uint8 host_mac[ETHADDR_LEN] = { 0x52, 0x55, 0x0a, 0x00, 0x02, 0x02 };

static struct spinlock netlock;

struct packet_queue {
  char *buf;          // packet buffer
  int len;            // packet length
  uint32 src_ip;      // source IP address
  uint16 src_port;    // source UDP port
};

struct port_info {
  int bound;          // is this port bound?
  uint16 port_num;    // the actual port number
  int head;           // queue head
  int tail;           // queue tail
  int count;          // number of packets in queue
  struct packet_queue queue[MAX_QUEUED_PACKETS];
};

static struct port_info ports[MAX_PORTS];

void
netinit(void)
{
  initlock(&netlock, "netlock");
  
  // Initialize all port structures
  for(int i = 0; i < MAX_PORTS; i++) {
    ports[i].bound = 0;
    ports[i].port_num = 0;
    ports[i].head = 0;
    ports[i].tail = 0;
    ports[i].count = 0;
  }
}

// Find port info structure for a given port number
static struct port_info*
find_port(short port)
{
  for(int i = 0; i < MAX_PORTS; i++) {
    if(ports[i].bound && ports[i].port_num == port) {
      return &ports[i];
    }
  }
  return 0;
}

// Allocate a port info structure for a given port number
static struct port_info*
alloc_port(short port)
{
  // Find a free slot
  for(int i = 0; i < MAX_PORTS; i++) {
    if(!ports[i].bound) {
      ports[i].bound = 1;
      ports[i].port_num = port;
      ports[i].head = 0;
      ports[i].tail = 0;
      ports[i].count = 0;
      return &ports[i];
    }
  }
  return 0;  // no free slots
}


//
// bind(int port)
// prepare to receive UDP packets address to the port,
// i.e. allocate any queues &c needed.
// You can bind the same port multiple times
uint64
sys_bind(void)
{
  int port;
  argint(0, &port);

  acquire(&netlock);

  // First, check if this port is already bound
  if (find_port((short)port) != 0) {
    release(&netlock);
    return -1;   // port already bound
  }

  // Now allocate a slot for this new bound port
  struct port_info *pi = alloc_port((short)port);
  if (pi == 0) {
    // no free slot for a new bound port
    release(&netlock);
    return -1;
  }

  release(&netlock);
  return 0;
}



//
// unbind(int port)
// release any resources previously created by bind(port);
// from now on UDP packets addressed to port should be dropped.
//
uint64
sys_unbind(void)
{
  int port;
  
  argint(0, &port);
  
  acquire(&netlock);
  
  struct port_info *pi = find_port(port);
  if(pi) {
    // Free any queued packets
    while(pi->count > 0) {
      kfree(pi->queue[pi->head].buf);
      pi->head = (pi->head + 1) % MAX_QUEUED_PACKETS;
      pi->count--;
    }
    pi->bound = 0;
    pi->port_num = 0;
  }
  
  release(&netlock);
  return 0;
}

//
// recv(int dport, int *src, short *sport, char *buf, int maxlen)
// if there's a received UDP packet already queued that was
// addressed to dport, then return it.
// otherwise wait for such a packet.
//
// sets *src to the IP source address.
// sets *sport to the UDP source port.
// copies up to maxlen bytes of UDP payload to buf.
// returns the number of bytes copied,
// and -1 if there was an error.
//
// dport, *src, and *sport are host byte order.
// bind(dport) must previously have been called.
//


uint64
sys_recv(void)
{
  int dport;
  uint64 src_addr, sport_addr, buf_addr;
  int maxlen;
  
  argint(0, &dport);
  argaddr(1, &src_addr);
  argaddr(2, &sport_addr);
  argaddr(3, &buf_addr);
  argint(4, &maxlen);
  
  struct proc *p = myproc();
  
  acquire(&netlock);
  
  struct port_info *pi = find_port(dport);
  if(pi == 0) {
    release(&netlock);
    return -1;  // port not bound
  }
  
  // Wait until a packet is available
  while(pi->count == 0) {
    sleep(pi, &netlock);
  }
  
  // Get the packet from the queue
  struct packet_queue *pq = &pi->queue[pi->head];
  
  // Store values before updating queue
  uint32 src_ip = pq->src_ip;
  uint16 src_port = pq->src_port;
  int payload_len = pq->len;
  char *payload_buf = pq->buf;
  
  // Update queue pointers BEFORE copyout (while still holding lock)
  pi->head = (pi->head + 1) % MAX_QUEUED_PACKETS;
  pi->count--;
  
  release(&netlock);  // RELEASE LOCK BEFORE COPYOUT!
  
  // Now do copyout WITHOUT holding the lock (copyout can sleep)
  // Copy metadata to user space
  if(copyout(p->pagetable, src_addr, (char*)&src_ip, sizeof(uint32)) < 0) {
    kfree(payload_buf);
    return -1;
  }
  if(copyout(p->pagetable, sport_addr, (char*)&src_port, sizeof(uint16)) < 0) {
    kfree(payload_buf);
    return -1;
  }
  
  // Copy payload to user space
  int copy_len = payload_len < maxlen ? payload_len : maxlen;
  if(copyout(p->pagetable, buf_addr, payload_buf, copy_len) < 0) {
    kfree(payload_buf);
    return -1;
  }
  
  // Free the packet buffer
  kfree(payload_buf);
  
  return copy_len;
}
// This code is lifted from FreeBSD's ping.c, and is copyright by the Regents
// of the University of California.
static unsigned short
in_cksum(const unsigned char *addr, int len)
{
  int nleft = len;
  const unsigned short *w = (const unsigned short *)addr;
  unsigned int sum = 0;
  unsigned short answer = 0;

  /*
   * Our algorithm is simple, using a 32 bit accumulator (sum), we add
   * sequential 16 bit words to it, and at the end, fold back all the
   * carry bits from the top 16 bits into the lower 16 bits.
   */
  while (nleft > 1)  {
    sum += *w++;
    nleft -= 2;
  }

  /* mop up an odd byte, if necessary */
  if (nleft == 1) {
    *(unsigned char *)(&answer) = *(const unsigned char *)w;
    sum += answer;
  }

  /* add back carry outs from top 16 bits to low 16 bits */
  sum = (sum & 0xffff) + (sum >> 16);
  sum += (sum >> 16);
  /* guaranteed now that the lower 16 bits of sum are correct */

  answer = ~sum; /* truncate to 16 bits */
  return answer;
}

//
// send(int sport, int dst, int dport, char *buf, int len)
//
uint64
sys_send(void)
{
  struct proc *p = myproc();
  int sport;
  int dst;
  int dport;
  uint64 bufaddr;
  int len;

  argint(0, &sport);
  argint(1, &dst);
  argint(2, &dport);
  argaddr(3, &bufaddr);
  argint(4, &len);

  int total = len + sizeof(struct eth) + sizeof(struct ip) + sizeof(struct udp);
  if(total > PGSIZE)
    return -1;

  char *buf = kalloc();
  if(buf == 0){
    printf("sys_send: kalloc failed\n");
    return -1;
  }
  memset(buf, 0, PGSIZE);

  struct eth *eth = (struct eth *) buf;
  memmove(eth->dhost, host_mac, ETHADDR_LEN);
  memmove(eth->shost, local_mac, ETHADDR_LEN);
  eth->type = htons(ETHTYPE_IP);

  struct ip *ip = (struct ip *)(eth + 1);
  ip->ip_vhl = 0x45; // version 4, header length 4*5
  ip->ip_tos = 0;
  ip->ip_len = htons(sizeof(struct ip) + sizeof(struct udp) + len);
  ip->ip_id = 0;
  ip->ip_off = 0;
  ip->ip_ttl = 100;
  ip->ip_p = IPPROTO_UDP;
  ip->ip_src = htonl(local_ip);
  ip->ip_dst = htonl(dst);
  ip->ip_sum = in_cksum((unsigned char *)ip, sizeof(*ip));

  struct udp *udp = (struct udp *)(ip + 1);
  udp->sport = htons(sport);
  udp->dport = htons(dport);
  udp->ulen = htons(len + sizeof(struct udp));

  char *payload = (char *)(udp + 1);
  if(copyin(p->pagetable, payload, bufaddr, len) < 0){
    kfree(buf);
    printf("send: copyin failed\n");
    return -1;
  }

  e1000_transmit(buf, total);

  return 0;
}

void
ip_rx(char *buf, int len)
{
  // don't delete this printf; make grade depends on it.
  static int seen_ip = 0;
  if(seen_ip == 0)
    printf("ip_rx: received an IP packet\n");
  seen_ip = 1;

    // Parse the IP header
  struct eth *eth = (struct eth *)buf;
  struct ip *ip = (struct ip *)(eth + 1);
  
  // Check if it's a UDP packet
  if(ip->ip_p != IPPROTO_UDP) {
    kfree(buf);
    return;
  }
  
  // Parse UDP header
  struct udp *udp = (struct udp *)(ip + 1);
  uint16 dport = ntohs(udp->dport);
  uint16 sport = ntohs(udp->sport);
  uint32 src_ip = ntohl(ip->ip_src);
  uint16 udp_len = ntohs(udp->ulen);
  
  // Calculate payload length (UDP length includes UDP header)
  int payload_len = udp_len - sizeof(struct udp);
  
  acquire(&netlock);
  
  // Find the port structure
  struct port_info *pi = find_port(dport);
  if(pi == 0) {
    // Port not bound, drop the packet
    release(&netlock);
    kfree(buf);
    return;
  }
  
  // Check if queue is full (16 packets max)
  if(pi->count >= MAX_QUEUED_PACKETS) {
    // Queue full, drop the packet
    release(&netlock);
    kfree(buf);
    return;
  }
  
  // Allocate a new buffer for the payload
  char *payload_buf = kalloc();
  if(payload_buf == 0) {
    release(&netlock);
    kfree(buf);
    return;
  }
  
  // Copy the payload
  char *payload = (char *)(udp + 1);
  memmove(payload_buf, payload, payload_len);
  
  // Add to queue
  struct packet_queue *pq = &pi->queue[pi->tail];
  pq->buf = payload_buf;
  pq->len = payload_len;
  pq->src_ip = src_ip;
  pq->src_port = sport;
  
  pi->tail = (pi->tail + 1) % MAX_QUEUED_PACKETS;
  pi->count++;
  
  // Wake up any process waiting for this port
  wakeup(pi);
  
  release(&netlock);
  
  // Free the original buffer
  kfree(buf);
}

//
// send an ARP reply packet to tell qemu to map
// xv6's ip address to its ethernet address.
// this is the bare minimum needed to persuade
// qemu to send IP packets to xv6; the real ARP
// protocol is more complex.
//
void
arp_rx(char *inbuf)
{
  static int seen_arp = 0;

  if(seen_arp){
    kfree(inbuf);
    return;
  }
  printf("arp_rx: received an ARP packet\n");
  seen_arp = 1;

  struct eth *ineth = (struct eth *) inbuf;
  struct arp *inarp = (struct arp *) (ineth + 1);

  char *buf = kalloc();
  if(buf == 0)
    panic("send_arp_reply");
  
  struct eth *eth = (struct eth *) buf;
  memmove(eth->dhost, ineth->shost, ETHADDR_LEN); // ethernet destination = query source
  memmove(eth->shost, local_mac, ETHADDR_LEN); // ethernet source = xv6's ethernet address
  eth->type = htons(ETHTYPE_ARP);

  struct arp *arp = (struct arp *)(eth + 1);
  arp->hrd = htons(ARP_HRD_ETHER);
  arp->pro = htons(ETHTYPE_IP);
  arp->hln = ETHADDR_LEN;
  arp->pln = sizeof(uint32);
  arp->op = htons(ARP_OP_REPLY);

  memmove(arp->sha, local_mac, ETHADDR_LEN);
  arp->sip = htonl(local_ip);
  memmove(arp->tha, ineth->shost, ETHADDR_LEN);
  arp->tip = inarp->sip;

  e1000_transmit(buf, sizeof(*eth) + sizeof(*arp));

  kfree(inbuf);
}

void
net_rx(char *buf, int len)
{
  struct eth *eth = (struct eth *) buf;

  if(len >= sizeof(struct eth) + sizeof(struct arp) &&
     ntohs(eth->type) == ETHTYPE_ARP){
    arp_rx(buf);
  } else if(len >= sizeof(struct eth) + sizeof(struct ip) &&
     ntohs(eth->type) == ETHTYPE_IP){
    ip_rx(buf, len);
  } else {
    kfree(buf);
  }
}
