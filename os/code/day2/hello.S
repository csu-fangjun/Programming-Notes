
  .code32
  .data
  .global d

d:
  .word 0xdead

  .global d1
d1:
  .word 0xbeaf

s0:
  .string "s0"

s1:
 .string "s1"

  .balign 16
  .comm c1, 8

  .balign 64
  .comm c2, 4

  .balign 32
  .lcomm c3, 8
  .lcomm c4, 2

  .text
  .global _start

_start:
  lea c1, %eax
  lea c2, %eax
  lea c3, %eax
  call foo

  .global foo
foo:
  mov $d1, %ebx # d1 is defined in this file
  mov $s0, %esi
  mov $s1, %edi

bar:
  call foo

#if 0
$ gcc -m32 -c -o hello.o hello.S
$ ld -m elf_i386 -o hello hello.o

$ readelf -S hello.o

There are 8 section headers, starting at offset 0xf0:

Section Headers:
  [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            00000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        00000000 000034 00002b 00  AX  0   0  1
  [ 2] .rel.text         REL             00000000 000348 000040 08      6   1  4
  [ 3] .data             PROGBITS        00000000 000080 000040 00  WA  0   0 64
  [ 4] .bss              NOBITS          00000000 0000c0 00000a 00  WA  0   0  8
  [ 5] .shstrtab         STRTAB          00000000 0000c0 000030 00      0   0  1
  [ 6] .symtab           SYMTAB          00000000 000230 0000f0 10      7   9  4
  [ 7] .strtab           STRTAB          00000000 000320 000027 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings)
  I (info), L (link order), G (group), T (TLS), E (exclude), x (unknown)
  O (extra OS processing required) o (OS specific), p (processor specific)

$ readelf -s hello.o

Symbol table '.symtab' contains 15 entries:
   Num:    Value  Size Type    Bind   Vis      Ndx Name
     0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 00000000     0 SECTION LOCAL  DEFAULT    1 
     2: 00000000     0 SECTION LOCAL  DEFAULT    3 
     3: 00000000     0 SECTION LOCAL  DEFAULT    4 
     4: 00000004     0 NOTYPE  LOCAL  DEFAULT    3 s0
     5: 00000007     0 NOTYPE  LOCAL  DEFAULT    3 s1
     6: 00000000     8 OBJECT  LOCAL  DEFAULT    4 c3
     7: 00000008     2 OBJECT  LOCAL  DEFAULT    4 c4
     8: 00000026     0 NOTYPE  LOCAL  DEFAULT    1 bar
     9: 00000000     0 NOTYPE  GLOBAL DEFAULT    3 d
    10: 00000002     0 NOTYPE  GLOBAL DEFAULT    3 d1
    11: 00000008     8 OBJECT  GLOBAL DEFAULT  COM c1
    12: 00000004     4 OBJECT  GLOBAL DEFAULT  COM c2
    13: 00000000     0 NOTYPE  GLOBAL DEFAULT    1 _start
    14: 00000017     0 NOTYPE  GLOBAL DEFAULT    1 foo

$ objdump -d hello.o

code/day2/hello.o:     file format elf32-i386

Disassembly of section .text:

00000000 <_start>:
   0:	8d 05 00 00 00 00    	lea    0x0,%eax
   6:	8d 05 00 00 00 00    	lea    0x0,%eax
   c:	8d 05 00 00 00 00    	lea    0x0,%eax
  12:	e8 fc ff ff ff       	call   13 <_start+0x13>

00000017 <foo>:
  17:	bb 00 00 00 00       	mov    $0x0,%ebx
  1c:	be 04 00 00 00       	mov    $0x4,%esi
  21:	bf 07 00 00 00       	mov    $0x7,%edi

00000026 <bar>:
  26:	e8 fc ff ff ff       	call   27 <bar+0x1>

$ readelf -r hello.o

Relocation section '.rel.text' at offset 0x348 contains 8 entries:
 Offset     Info    Type            Sym.Value  Sym. Name
00000002  00000b01 R_386_32          00000008   c1
00000008  00000c01 R_386_32          00000004   c2
0000000e  00000301 R_386_32          00000000   .bss
00000013  00000e02 R_386_PC32        00000017   foo
00000018  00000a01 R_386_32          00000002   d1
0000001d  00000201 R_386_32          00000000   .data
00000022  00000201 R_386_32          00000000   .data
00000027  00000e02 R_386_PC32        00000017   foo

$ readelf -S hello

There are 7 section headers, starting at offset 0x12c:

Section Headers:
  [Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            00000000 000000 000000 00      0   0  0
  [ 1] .text             PROGBITS        08048074 000074 00002b 00  AX  0   0  1
  [ 2] .data             PROGBITS        080490c0 0000c0 000040 00  WA  0   0 64
  [ 3] .bss              NOBITS          08049100 000100 000020 00  WA  0   0  8
  [ 4] .shstrtab         STRTAB          00000000 000100 00002c 00      0   0  1
  [ 5] .symtab           SYMTAB          00000000 000244 000140 10      6  11  4
  [ 6] .strtab           STRTAB          00000000 000384 000047 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings)
  I (info), L (link order), G (group), T (TLS), E (exclude), x (unknown)
  O (extra OS processing required) o (OS specific), p (processor specific)

$ readelf -s hello

Symbol table '.symtab' contains 20 entries:
   Num:    Value  Size Type    Bind   Vis      Ndx Name
     0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 08048074     0 SECTION LOCAL  DEFAULT    1 
     2: 080490c0     0 SECTION LOCAL  DEFAULT    2 
     3: 08049100     0 SECTION LOCAL  DEFAULT    3 
     4: 00000000     0 FILE    LOCAL  DEFAULT  ABS hello.o
     5: 080490c4     0 NOTYPE  LOCAL  DEFAULT    2 s0
     6: 080490c7     0 NOTYPE  LOCAL  DEFAULT    2 s1
     7: 08049100     8 OBJECT  LOCAL  DEFAULT    3 c3
     8: 08049108     2 OBJECT  LOCAL  DEFAULT    3 c4
     9: 0804809a     0 NOTYPE  LOCAL  DEFAULT    1 bar
    10: 00000000     0 FILE    LOCAL  DEFAULT  ABS 
    11: 080490c2     0 NOTYPE  GLOBAL DEFAULT    2 d1
    12: 08049110     4 OBJECT  GLOBAL DEFAULT    3 c2
    13: 08048074     0 NOTYPE  GLOBAL DEFAULT    1 _start
    14: 08049118     8 OBJECT  GLOBAL DEFAULT    3 c1
    15: 08049100     0 NOTYPE  GLOBAL DEFAULT    3 __bss_start
    16: 080490c0     0 NOTYPE  GLOBAL DEFAULT    2 d
    17: 0804808b     0 NOTYPE  GLOBAL DEFAULT    1 foo
    18: 08049100     0 NOTYPE  GLOBAL DEFAULT    2 _edata
    19: 08049120     0 NOTYPE  GLOBAL DEFAULT    3 _end

$ objdump -d hello


code/day2/hello:     file format elf32-i386


Disassembly of section .text:

08048074 <_start>:
 8048074:	8d 05 18 91 04 08    	lea    0x8049118,%eax
 804807a:	8d 05 10 91 04 08    	lea    0x8049110,%eax
 8048080:	8d 05 00 91 04 08    	lea    0x8049100,%eax
 8048086:	e8 00 00 00 00       	call   804808b <foo>

0804808b <foo>:
 804808b:	bb c2 90 04 08       	mov    $0x80490c2,%ebx
 8048090:	be c4 90 04 08       	mov    $0x80490c4,%esi
 8048095:	bf c7 90 04 08       	mov    $0x80490c7,%edi

0804809a <bar>:
 804809a:	e8 ec ff ff ff       	call   804808b <foo>

$ readelf -r hello

There are no relocations in this file.

$ nm hello.o

00000026 t bar
00000008 C c1
00000004 C c2
00000000 b c3
00000008 b c4
00000000 D d
00000002 D d1
00000017 T foo
00000004 d s0
00000007 d s1
00000000 T _start

$ nm hello

0804809a t bar
08049100 B __bss_start
08049118 B c1
08049110 B c2
08049100 b c3
08049108 b c4
080490c0 D d
080490c2 D d1
08049100 D _edata
08049120 B _end
0804808b T foo
080490c4 d s0
080490c7 d s1
08048074 T _start

#endif
