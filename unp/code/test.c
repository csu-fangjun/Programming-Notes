#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>

static void test_htons() {
  // short means uint16_t
  // long means uint32_t
  //
  // Note that network byte order is big endian.
  assert(htons(0x1234) == 0x3412);
  assert(htonl(0x12345678) == 0x78563412);

  assert(ntohs(0x3412) == 0x1234);
  assert(ntohl(0x78563412) == 0x12345678);
}

static void test_inet_pton() {
  uint32_t addr;
  int ret = inet_pton(AF_INET, "1.2.3.4", &addr);
  // note that addr is in network byte order, e.g., big endian
  assert(addr == 0x04030201);
  assert(ret == 1);  // return 1 on success
}

static void test_inet_aton() {
  struct in_addr inp;
  int ret = inet_aton("1.2.3.4", &inp);
  assert(inp.s_addr == 0x04030201);
  assert(ret == 1);  // 1 on success; 0 on failure
}

int main() {
  test_htons();
  test_inet_pton();
  test_inet_aton();
  return 0;
}
