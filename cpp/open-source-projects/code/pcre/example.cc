#include "pcre.h"
#include "pcrecpp.h"
#include <iostream>

void test_cpp();
void test_c();

int main() {
  // test_c();
  test_cpp();
}

void test_cpp() {
  {
    // http://host:port/abc
    pcrecpp::RE re(R"(http://(.*):(\d+).*)");
    std::string s = "http://www.google.com:1234/hello";
    s = "http://www.google";

    std::string host;
    int port;

    std::cerr << "number of capturing groups: " << re.NumberOfCapturingGroups()
              << "\n";
    if (re.FullMatch(s, &host, &port)) {
      std::cerr << "host: " << host << "\n";
      std::cerr << "port: " << port << "\n";
    } else {
      std::cerr << "no match\n";
    }
  }
  {
    // parse numbers
    pcrecpp::RE re(R"((\d+))"); // note that it MUST be a group
    pcrecpp::StringPiece s = "12 100 a bc d 34 56";
    int n;
    while (re.Consume(&s, &n)) {
      std::cerr << "n: " << n << "\n";
      // it prints only
      // 12
    }
  }

  {
    // parse numbers
    pcrecpp::RE re(R"((\d+)\s+)"); // note that it MUST be a group
    pcrecpp::StringPiece s = "12 100 a bc d 34 56";
    int m;
    while (re.Consume(&s, &m)) {
      std::cerr << "m: " << m << "\n";
      // it prints only
      // 12
      // 100
    }
  }

  {
    // parse numbers
    pcrecpp::RE re(R"((\d+))"); // note that it MUST be a group
    pcrecpp::StringPiece s = "12 100 a bc d 34 56";
    int k;
    while (re.FindAndConsume(&s, &k)) {
      std::cerr << "k: " << k << "\n";
      // it prints:
      // 12
      // 100
      // 34
      // 56
    }
  }
}

void test_c() {
  const char *test_strings[] = {
      "This should match... hello", "This could match... hello!",
      "More than one hello... hello", "No chance of a match...", nullptr};

  const char *str_regex = "((.*)(hello))+";

  const char *err_str;
  int err_offset;
  pcre *compiled = pcre_compile(str_regex, 0, &err_str, &err_offset, nullptr);
  if (compiled == nullptr) {
    std::cerr << "could not compile: " << str_regex << "\t: " << err_str
              << "\n";
    std::cerr << "                   ";
    for (int i = 0; i != err_offset; ++i) {
      std::cerr << " ";
    }
    std::cerr << "^"
              << "\n";
    exit(-1);
  }

  pcre_extra *extra = pcre_study(compiled, 0, &err_str);
  if (extra == nullptr) {
    std::cerr << "Could not study '" << str_regex << "': " << err_str << "\n";
    exit(-1);
  }

  int sub_str_vec[30];
  for (auto s : test_strings) {
    if (!s) {
      break;
    }

    int ret =
        pcre_exec(compiled, extra, s, strlen(s),
                  0, // start from this position
                  0, // options
                  sub_str_vec,
                  sizeof(sub_str_vec) / sizeof(int) // length of sub_str_vec
        );
    if (ret < 0) {
      switch (ret) {
      case PCRE_ERROR_NOMATCH:
        std::cerr << "No match: " << s << "\n";
        break;
      case PCRE_ERROR_NULL:
        std::cerr << "null\n";
        break;
      case PCRE_ERROR_BADOPTION:
        std::cerr << "a bad option was passed\n";
        break;
      case PCRE_ERROR_BADMAGIC:
        std::cerr << "bad magic number (compiled re corrupted?)\n";
        break;
      case PCRE_ERROR_UNKNOWN_NODE:
        std::cerr << "some unknown node in compiled re)\n";
        break;
      case PCRE_ERROR_NOMEMORY:
        std::cerr << "out of memory\n";
        break;
      default:
        std::cerr << "unknown error\n";
        break;
      }
    } else {
      std::cerr << "Got a match: " << s << "\n";
    }
  }

  pcre_free(compiled);
  if (extra) {
#if defined(PCRE_CONFIG_JIT)
    pcre_free_study(extra);
#else
    pcre_free(extra);
#endif
  }
}
