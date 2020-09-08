ENTRY(main)

SECTIONS {
  . = 0x10000;
  .text : { *(.foo) *(.text) }
  /DISCARD/ : {  *(.eh_frame) }

  . = 0x8000000;
  .data : { *(.data) }
  .bss : { *(.bss) }
}
