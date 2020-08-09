#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

// bounded buffer
class Buffer {
 public:
  void push(int v) {
    std::unique_lock<std::mutex> lock(m_);
    while (is_full()) {
      std::cout << "push wait, size: " << size() << "\n";
      cond_full_.wait(lock);
    }
    buf_[front_] = v;
    front_ = (front_ + 1) % kN_;

    cond_empty_.notify_one();
  }
  int pop() {
    std::unique_lock<std::mutex> lock(m_);
    while (is_empty()) {
      std::cout << "pop wait, size: " << size() << "\n";
      cond_empty_.wait(lock);
    }

    int v = buf_[end_];
    end_ = (end_ + 1) % kN_;
    cond_full_.notify_one();
    return v;
  }
  bool is_empty() const { return front_ == end_; }
  bool is_full() const { return (front_ + 1) % kN_ == end_; }
  int size() const { return (front_ + kN_ - end_) % kN_; }

 private:
  int front_ = 0;
  int end_ = 0;
  enum { kN_ = 3 };
  int buf_[kN_];

  std::mutex m_;
  std::condition_variable cond_empty_;
  std::condition_variable cond_full_;
};

class Buffer g;
bool done = false;
static void producer() {
  for (int i = 0; i < 30; ++i) {
    g.push(i);
  }
  done = true;
}

static void consumer() {
  while (!done) {
    int v = g.pop();
    std::cout << v << " ";
  }
  std::cout << "\n";
}

int main() {
  std::thread t1(producer);
  std::thread t2(consumer);
  std::thread t3(consumer);
  t1.join();
  t2.join();
  t3.join();
  return 0;
}
