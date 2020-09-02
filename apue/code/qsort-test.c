#include <stdio.h>
#include <stdlib.h>

int cmp_int(const void *first, const void *second) {
  const int *p1 = (const int *)first;
  const int *p2 = (const int *)second;
  if (*p1 < *p2)
    return -1;
  else if (*p1 == *p2)
    return 0;
  else
    return 1;
}

void test_int() {
  int arr[] = {10, 3, -1, 2, 19};
  int n = sizeof(arr) / sizeof(arr[0]);
  printf("before sort:\n");
  for (int i = 0; i != n; ++i) {
    printf("%d ", arr[i]);
  }
  printf("\n");

  qsort(arr, n, sizeof(arr[0]), cmp_int);

  printf("\nafter sort:\n");
  for (int i = 0; i != n; ++i) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

struct People {
  int age;
  const char *name;
};

int cmp_people(const void *first, const void *second) {
  const struct People *p1 = (const struct People *)first;
  const struct People *p2 = (const struct People *)second;
  if (p1->age < p2->age)
    return -1;
  else if (p1->age == p2->age)
    return 0;
  else
    return 1;
}

void test_struct() {
  struct People people[] = {
      {.age = 30, .name = "30 name"},
      {.age = 40, .name = "40 name"},
      {.age = 10, .name = "10 name"},
      {.age = 3, .name = "3 name"},
  };

  int n = sizeof(people) / sizeof(people[0]);
  printf("n: %d\n", n);

  printf("before sort:\n");
  for (int i = 0; i != n; ++i) {
    printf("(%d, %s) ", people[i].age, people[i].name);
  }
  printf("\n");

  qsort(people, n, sizeof(people[0]), cmp_people);

  printf("after sort:\n");
  for (int i = 0; i != n; ++i) {
    printf("(%d, %s) ", people[i].age, people[i].name);
  }
  printf("\n");
}

int main() {
  test_int();
  test_struct();
  return 0;
}
