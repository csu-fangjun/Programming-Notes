; Usage:
; lli global_variable.ll
; echo $?
@gv = global i32 10

define i32 @main() {
  %1 = load i32, i32* @gv
  %2 = add i32 %1, 2
  ret i32 %2
}
