#![no_std]
#![no_main]

use core::ptr::{read_volatile, write_volatile};

#[path = "../common.rs"]
mod common;

#[no_mangle]
extern "C" fn guest_main() -> ! {
    let count = common::input();
    let mut index = 0_u64;
    let mut sum = 0_u64;
    while index < count {
        // SAFETY: sum is a live, aligned stack word throughout this single-hart loop.
        // Volatile access prevents closed-form lowering to a multiplication routine.
        unsafe { write_volatile(&mut sum, read_volatile(&sum).wrapping_add(index)) };
        index = index.wrapping_add(1);
    }
    common::finish(sum)
}
