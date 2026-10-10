use core::arch::{asm, global_asm};
use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

const INPUT: usize = 0x7fff_ffe0;
const OUTPUT: usize = 0x7fff_ffe8;
const PANIC: usize = 0x7fff_fff0;
const TERMINATION: usize = 0x7fff_fff8;

global_asm!(
    ".option norvc",
    ".option norelax",
    ".section .text.entry,\"ax\",@progbits",
    ".globl _start",
    "_start:",
    "la sp, __stack_top",
    "call guest_main",
    "2: jal zero, 2b",
);

pub fn input() -> u64 {
    // SAFETY: guests.rs supplies an aligned eight-byte public input region here.
    unsafe { read_volatile(INPUT as *const u64) }
}

pub fn finish(value: u64) -> ! {
    // SAFETY: guests.rs reserves these aligned output and write-once status words.
    unsafe {
        write_volatile(OUTPUT as *mut u64, value);
        write_volatile(TERMINATION as *mut u64, 1);
        asm!("2: jal zero, 2b", options(noreturn));
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    // SAFETY: guests.rs reserves the panic word; the trap-free loop never returns.
    unsafe {
        write_volatile(PANIC as *mut u64, 1);
        asm!("2: jal zero, 2b", options(noreturn));
    }
}
