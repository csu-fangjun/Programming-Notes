#include "tree/event-map.h"
/*
TableEventMap

There is a key, i.e., position for the table.
A table can be used for a pdf, the 0th position, the 1st position, etc.
We can consider the key as the table type.

A table has a vector of event map. The index of the map is the
value corresponding to the `key` in the given event type.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

SplitEventMap

It has a yes set and a key. For a given event, it first get the
value of the key, if the value is in the yes set, choose the yes_map,
else choose the no_map.

*/

using namespace kaldi;

void test_const_event_map() {
  ConstantEventMap m(10);     // 10 is the answer
  m.Write(std::cout, false);  // CE 10
  std::cout << "\n";
  EventType e = {{-1, 10}, {1, 2}, {2, 100}};
  EventAnswerType ans;
  m.Map(e, &ans);
  std::cout << ans << "\n";  // 10
}

void test_table_event_map() {
  ConstantEventMap* m1 = new ConstantEventMap(10);
  ConstantEventMap* m2 = new ConstantEventMap(20);
  ConstantEventMap* m3 = new ConstantEventMap(30);

  std::map<EventValueType, EventMap*> map;
  map[1] = m1;
  map[2] = m2;
  map[3] = m3;
  EventKeyType key = 0;
  TableEventMap m(key, map);
  m.Write(std::cout, false);  // TE 0 4 ( NULL CE 10 CE 20 CE 30)

  // TE: table event
  // 0: the key is 0
  // 4: there are 4 elements
  // map for value 0, e.g., index 0, is NULL
  // map for value 1, e.g., index 1, is 10
  // map for value 2, e.g., index 2, is 20
  // map for value 3, e.g., index 3, is 30
}

void test_table_event_map2() {
  std::map<EventValueType, EventAnswerType> map{{1, 10}, {2, 20}, {3, 30}};
  EventKeyType key = 0;
  TableEventMap m(key, map);
  m.Write(std::cout, false);
  // TE 0 4 ( NULL CE 10 CE 20 CE 30 )
}

int main() {
  // test_const_event_map();
  // test_table_event_map();
  test_table_event_map2();
  return 0;
}
