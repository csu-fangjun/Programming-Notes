#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define MAXLINE 1000

static void err_exit(const char* msg) {
  perror(msg);
  exit(-1);
}

int main(int argc, char* argv[]) {
  int sockfd, n;
  char recvline[MAXLINE + 1];

  struct sockaddr_in server_addr;
  if (argc != 2) {
    fprintf(stderr, "usage: a.out <server_ip>\n");
    return -1;
  }
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;  // Use IP protoal
  server_addr.sin_port = htons(13);  // server port is 13
  // server IP address will be set later using `inet_pton()`

  // int socket(int domain, int type, int protocol);
  // create a socket; later we will connect it to the server
  // so that we can use read/write
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    err_exit("socket");
  }
  // struct in_addr {uint32_t s_addr};
  // get server IP from the command line
  if (inet_pton(AF_INET, argv[1], &server_addr.sin_addr) <= 0) {
    err_exit("inet_pton");
  }

  if (connect(sockfd, (const struct sockaddr*)&server_addr,
              sizeof(server_addr)) < 0) {
    err_exit("connect");
  }

  while ((n = read(sockfd, recvline, MAXLINE) > 0)) {
    recvline[n] = 0;  // null terminate
    if (fputs(recvline, stdout) == EOF) {
      err_exit("fputs");
    }
  }
  if (n < 0) {
    err_exit("read");
  }

  return 0;
}
