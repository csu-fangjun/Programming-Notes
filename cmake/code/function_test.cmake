
message(STATUS "run it with cmake -P ./function_test.cmake")

set(foo bar)

#[==[
Function has a separate stack, so ``set(var1 hello)``
does not leaky to the caller. That is, ``var1`` is
not visible to the caller.
#]==]
function(my_function var1)
  message(STATUS "value of var1 is ${var1}")
  set(var1 hello)
  message(STATUS "value of var1 is ${var1}")
endfunction()


my_function(foo) # var1 is foo
message(STATUS "after my_macro")
message(STATUS "value of foo is: ${foo}")
message(STATUS "value of var1 is: ${var1}")

message(STATUS)

my_function(${foo}) $ var1 is bar.
#[==[
-- run it with cmake -P ./function_test.cmake
-- value of var1 is foo
-- value of var1 is hello
-- after my_macro
-- value of foo is: bar
-- value of var1 is:
--
-- value of var1 is bar
-- value of var1 is hello
#]==]

