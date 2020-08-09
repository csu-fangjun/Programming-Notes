
message(STATUS "run it with cmake -P ./macro_test.cmake")

set(foo bar)
#[=====[
#  Before running the macro, cmake replaces all occurrences
#  of `${var1}` with the passed argument. In our case,
#  we invoke the macro using `my_macro(foo)`, so
#  the value of `${var1}` is foo.
#
#  It is called a macro because cmake does string substitution.
#  CMake substitues only `${var1}`, NOT `var1`.
#
#  Another point is that ``set(var1 hello)`` leaks the variable.
#  There is no separate stack for the macro.
#
#  NOTE(fangjun): Never use ``set(some_var "xxx")`` in a macro.
#  Use ``set(${some_var} "xxx")`` instead, where ``some_var``
#  is an argument.
#
# ]=====]
macro(my_macro var1)
  message(STATUS "value of var1 is ${var1}")
  set(var1 hello)
  message(STATUS "value of var1 is ${var1}")
endmacro()

my_macro(foo)
message(STATUS "after my_macro")
message(STATUS "value of foo is: ${foo}")
message(STATUS "value of var1 is: ${var1}")

message(STATUS)

my_macro(${foo})

#[===[
-- run it with cmake -P ./macro_test.cmake
-- value of var1 is foo
-- value of var1 is foo
-- after my_macro
-- value of foo is: bar
-- value of var1 is: hello
--
-- value of var1 is bar
-- value of var1 is bar
#]===]
