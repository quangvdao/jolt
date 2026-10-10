#![no_std]
#![no_main]

#[path = "../common.rs"]
mod common;

#[no_mangle]
extern "C" fn guest_main() -> ! {
    let count = common::input();
    let mut index = 0_u64;
    let mut sum = 0_u64;
    while index < count {
        sum = sum.wrapping_add(index);
        index = index.wrapping_add(1);
    }
    common::finish(sum)
}
