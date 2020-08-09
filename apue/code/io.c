#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void test_read_write() {
  const char* filename = "test.txt";
  // this test shows that if we open a file for read and write,
  // it shares the same offset for reading and writing.
  int fd = open(filename, O_CREAT | O_RDWR, S_IRWXU);
  if (fd == -1) {
    int saved_errno = errno;
    // if filename is "/test.txt"
    perror("io");                   // io: Permission denied
    perror(strerror(saved_errno));  // Permision denied: Permission denied
    printf("errno: %d\n", saved_errno);
    printf("%s:%d %s\n", __FUNCTION__, __LINE__, strerror(saved_errno));
    exit(-1);
  }

  remove(filename);  // it calls unlink.

  off_t offset = lseek(fd, 0, SEEK_CUR);
  assert(offset == 0);

  ssize_t n = write(fd, "123", 3);
  assert(n == 3);

  offset = lseek(fd, 0, SEEK_CUR);
  assert(offset == 3);

  // note that read/write share the same offset!
  char buf[10];
  n = read(fd, buf, sizeof(buf));
  assert(n == 0);

  offset = lseek(fd, -1, SEEK_CUR);
  assert(offset == 2);

  n = read(fd, buf, sizeof(buf));
  printf("n: %ld\n", (long)n);
  assert(n == 1);
  assert(buf[0] == '3');

  offset = lseek(fd, 0, SEEK_CUR);
  assert(offset == 3);

  int ret = close(fd);
  // ret = close(fd);  // close it the second time
  if (ret == -1) {
    perror("io");  // io: Bad file descriptor
    exit(-1);
  }
}

static void test_fileno() {
  assert(STDIN_FILENO == 0);
  assert(STDOUT_FILENO == 1);
  assert(STDERR_FILENO == 2);

  char s[] = "hello";
  char w[] = " world";
  // it is printed to the console immediately
  write(STDOUT_FILENO, s, sizeof(s));
  sleep(1);
  write(STDOUT_FILENO, w, sizeof(w));
  write(1, "\n", 1);
}

static void test_fork() {
  // this shows that after fork(),
  // the "user file descriptor table" is copied
  // from the parent to the child.
  // Since the file offset is saved in the kernel
  // data structure "file table", when the child
  // changes the offset, the parent will see it.
  //
  // We have to close the file both in the parent and in the child.
  //
  // The file descriptor returned by "open()" is the index
  // into the "user file descriptor table", which is a per process
  // data structure.
  //
  // The file offset is saved in the kernel data structure "file table";
  // this structure is shared by processes.
  const char* filename = "a.txt";
  int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
  if (fd == -1) {
    perror("test_fork");
    exit(0);
  }

  const char s[] = "hello";

  pid_t pid = fork();
  if (pid == 0) {
    // child
    write(fd, s, sizeof(s) - 1);
    off_t offset = lseek(fd, 0, SEEK_CUR);
    printf("child: %ld\n", offset);
    int ret = close(fd);
    if (ret == -1) {
      perror("close");
    }
    return;
  } else {
    sleep(2);
    off_t offset = lseek(fd, 0, SEEK_CUR);
    if (offset == -1) {
      perror("lseek");
      close(fd);
      return;
    }
    printf("parent offset: %ld\n", offset);
    lseek(fd, -2, SEEK_CUR);
    char buf[10];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n == -1) {
      perror("read error");
      close(fd);
      return;
    }

    buf[n] = '\0';
    printf("parent data: %s\n", buf);
  }

  int ret = close(fd);
  if (ret == -1) {
    perror("parent close");
  }
}

int main() {
  // test_read_write();
  // test_fileno();
  test_fork();
  return 0;
}
