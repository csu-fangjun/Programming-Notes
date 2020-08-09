#!/bin/bash

function disassemble() {
  local name="1-print-hello"
  aarch64-linux-gnu-readelf -x .rodata $name

  aarch64-linux-gnu-objdump -d $name | awk -v RS= '/^[[:xdigit:]]+ <main>/'

  : <<'EOF'
0000000000400640 <main>:
  400640:       a9bf7bfd        stp     x29, x30, [sp,#-16]!
  400644:       10000560        adr     x0, 4006f0 <hello>
  400648:       97ffffaa        bl      4004f0 <printf@plt>
  40064c:       30000540        adr     x0, 4006f5 <new_line>
  400650:       97ffffa8        bl      4004f0 <printf@plt>
  400654:       d2800000        mov     x0, #0x0                        // #0
  400658:       a8c17bfd        ldp     x29, x30, [sp],#16
  40065c:       d65f03c0        ret
EOF

  aarch64-linux-gnu-objdump -d $name | awk -v RS= '/^[[:xdigit:]]+ <hello>/'

  : <<'EOF'
00000000004006f0 <hello>:
  4006f0:       34333231        .word   0x34333231
    ...
EOF
}

function run() {
  make run-$1
}

if [ $# -ne 1 ]; then
  echo "please provide 1 argument"
  exit 0
fi

# disassemble


name=$(basename -s .S $1)
name=$(basename -s .c $name)
name=$(basename -s .cc $name)
run $name

