OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
    . = 0x80000000;
    .text : ALIGN(4) {
        KEEP(*(.text.entry))
        *(.text .text.*)
        . = ALIGN(4);
    }
    .rodata : ALIGN(8) {
        *(.rodata .rodata.* .srodata .srodata.*)
        . = ALIGN(8);
    }
    .data : ALIGN(8) {
        *(.data .data.* .sdata .sdata.*)
        . = ALIGN(8);
    }
    .bss (NOLOAD) : ALIGN(8) {
        *(.bss .bss.* .sbss .sbss.*)
        . = ALIGN(8);
    }
    __stack_top = ALIGN(. + 4096, 16);
    /DISCARD/ : { *(.eh_frame .eh_frame_hdr) }
}
