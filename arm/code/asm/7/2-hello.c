
#include <stdio.h>
#include <unistd.h>

int main() {
  printf("1234567!\n");
  write(2, "123\n", 5);
  return 0;
}

#if 0
aarch64-linux-gnu-objdump -d 2-hello | awk -v RS= '/^[[:xdigit:]]+ <main>/'
0000000000400670 <main>:
  400670:       a9bf7bfd        stp     x29, x30, [sp,#-16]!
  400674:       910003fd        mov     x29, sp
  400678:       90000000        adrp    x0, 400000 <_init-0x4b8>
  40067c:       911ca000        add     x0, x0, #0x728
  400680:       97ffffa8        bl      400520 <puts@plt>
  400684:       52800000        mov     w0, #0x0                        // #0
  400688:       a8c17bfd        ldp     x29, x30, [sp],#16
  40068c:       d65f03c0        ret
#endif
