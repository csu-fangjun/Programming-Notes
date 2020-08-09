#include <iostream>
#include <sstream>
enum {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  FATAL = 4,
};

const char* kName = "DIWEF";
int kThreshold = 1;

class MessageLogger {
 public:
  MessageLogger(const char* file, int line_no, int severity)
      : severity_(severity) {
    os_ << "[" << kName[severity_] << "] " << file << ":" << line_no << " ";
  }

  std::ostream& stream() { return os_; }
  ~MessageLogger() {
    std::cerr << os_.str() << "\n";
    if (severity_ == FATAL) {
      abort();
    }
  }

 private:
  int severity_;
  std::ostringstream os_;
};

class Voidfier {
 public:
  Voidfier() = default;
  void operator&(const std::ostream& os) const {}
};

#define LOG(n) \
  if ((n) >= kThreshold) MessageLogger(__FILE__, __LINE__, n).stream()

#define LOG_IF(n, cond) \
  if ((n) >= kThreshold && (cond)) MessageLogger(__FILE__, __LINE__, n).stream()

#define FATAL_IF(cond) \
  (cond) ? (void)0     \
         : Voidfier() & MessageLogger(__FILE__, __LINE__, FATAL).stream()

#define CHECK(cond) FATAL_IF(cond) << "Check failed: " #cond " "

#ifndef NDBUG
#define DCHECK(cond) CHECK(cond)
#else
#define DCHECK(cond) \
  while (false) CHECK(cond)
#endif

#define CHECK_OP(val1, val2, op)                                              \
  FATAL_IF(((val1)op(val2))) << "Check failed: " #val1 " " #op " " #val2 " (" \
                             << val1 << " vs " << val2 << ") "

#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)

#ifndef NDBUG
#define DCHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define DCHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#define DCHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#else
#define DCHECK_EQ(val1, val2) \
  while (false) CHECK_OP(val1, val2, ==)
#define DCHECK_NE(val1, val2) \
  while (false) CHECK_OP(val1, val2, !=)
#define DCHECK_GE(val1, val2) \
  while (false) CHECK_OP(val1, val2, >=)
#define DCHECK_GT(val1, val2) \
  while (false) CHECK_OP(val1, val2, >)
#define DCHECK_LE(val1, val2) \
  while (false) CHECK_OP(val1, val2, <=)
#define DCHECK_LT(val1, val2) \
  while (false) CHECK_OP(val1, val2, <)
#endif

int main() {
  LOG(INFO) << "hello world";
  int a = 2;
  int b = 3;
  CHECK_EQ(a, b) << "here";
  LOG(WARNING) << "warning";
}
