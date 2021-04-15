#include <assert.h>
#include <stdio.h>  // for perror()
#include <unistd.h> // for pipe()

/** int pipe(int fildes[2]); */

// this test shows the basic usage of pipe.
// It is useless and it is not used in practice.
//
// pipe() is usually for interprocess communication.
void test1() {
  int fd[2];
  int ret = pipe(fd);
  if (ret == -1) {
    perror("test1");
    return;
  }
#if 0
  // TODO(fangjun): The program hangs!
  // fd[0] for reading
  // fd[1] for writing
  FILE *fp0 = fdopen(fd[0], "r");
  FILE *fp1 = fdopen(fd[1], "w");

  // write to the pipe
  int a = 10;
  // n is the number of elements read/written
  int n = fwrite(&a, sizeof(int), 1, fp1);
  assert(n == 1);

  // read from the pipe
  int b;
  n = fread(&a, sizeof(int), 1, fp0);
  assert(n == 1);
  assert(a == 10);
#endif
  // there were only 3 descriptors: 0, 1, and 2
  // so fd[0] == 3, fd[1] == 4
  assert(fd[0] == 3);
  assert(fd[1] == 4);

  off_t t = lseek(fd[0], 0, SEEK_CUR);
  assert(t == -1);
  if (t == -1) {
    // perror("lseek"); // lseek: Illegal seek
  }

  int a = 10;
  ssize_t n = write(fd[1], &a, sizeof(a));
  assert(n == 4);

  int b[3] = {0};

  // there is not enough data in the pipe,
  // but it still returns
  n = read(fd[0], &b, sizeof(b));
  assert(n == 4);
  assert(b[0] == 10);
}

void test2() {
  int fd[2];
  int ret = pipe(fd);
  if (ret == -1) {
    perror("test2 popen");
    return;
  }
  pid_t pid = fork();
  if (pid == -1) {
    perror("test2 fork");
    return;
  } else if (pid == 0) {
    // child
    // child will read from fd[0]
    //
    // fd[1] is still opened in the parent
    close(fd[1]);
    int a = 10;
    // read will hang if the pipe is empty
    ssize_t n = read(fd[0], &a, sizeof(a));
    assert(n == 4);
    assert(a == 100);
  } else {
    // parent
    // parent will write to fd[1]
    //
    // fd[0] is still opened in the child
    close(fd[0]);
    int a = 100;
    ssize_t n = write(fd[1], &a, sizeof(a));
    assert(n == 4);
  }
  return;
}

int main() {
  test1();
  test2();
  return 0;
}
