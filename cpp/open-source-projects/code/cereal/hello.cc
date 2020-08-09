// Copyright 2020. All Rights Reserved.
// Author: fangjun.kuang@gmail.com (Fangjun Kuang)
// Date: Sat Mar 28 14:39:27 CST 2020
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/archives/xml.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

namespace {

struct Student {
  int id;
  std::string name;
  std::vector<int> courses;

  template <typename Archive>
  void serialize(Archive& ar) {
    ar(id, name, courses);
  }
};

struct Class {
  std::vector<Student> students;
  int id;
#if 0
  template <typename Archive>
  void save(Archive& ar) const {
    ar(id, students);
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(id);
    ar(students);
  }
#else
  template <typename Archive>
  void serialize(Archive& ar) {
    ar(id);
    ar(students);
  }
#endif
};

}  // namespace

int main() {
  {
    std::ofstream os("test.bin", std::ios::binary);
    cereal::BinaryOutputArchive archive(os);

    Class c;
    c.id = 2020;

    Student s1;
    s1.id = 1;
    s1.name = "s1";
    s1.courses = {1, 2, 3};

    Student s2;
    s2.id = 2;
    s2.name = "s2";
    s2.courses = {10, 20};

    c.students = {s1, s2};

    archive(c);

    {
      cereal::JSONOutputArchive output(std::cout);
      output(c);
    }

    {
      cereal::XMLOutputArchive xml(std::cout);
      xml(c);
    }
  }
  {
    std::ifstream fs("test.bin", std::ios::binary);
    cereal::BinaryInputArchive iarchive(fs);
    Class c;
    iarchive(c);
    assert(c.students.size() == 2);
  }

  return 0;
}
