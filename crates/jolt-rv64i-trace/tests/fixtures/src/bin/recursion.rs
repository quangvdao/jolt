#![no_std]
#![no_main]

use core::ptr::{read_volatile, write_volatile};

#[path = "../common.rs"]
mod common;

#[inline(never)]
fn recurse(depth: u64) -> u64 {
    if depth == 0 {
        return 1;
    }
    let previous = recurse(depth - 1);
    let mut result = previous.wrapping_add(depth);
    // SAFETY: result is a live stack word. The post-call volatile access keeps
    // this call non-tail and makes each recursion frame observable.
    unsafe {
        write_volatile(&mut result, previous.wrapping_add(depth));
        read_volatile(&result)
    }
}

#[no_mangle]
extern "C" fn guest_main() -> ! {
    common::finish(recurse(common::input()))
}
