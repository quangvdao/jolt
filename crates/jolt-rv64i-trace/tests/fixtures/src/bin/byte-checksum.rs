#![no_std]
#![no_main]

use core::ptr::{addr_of_mut, read_volatile, write_volatile};

#[path = "../common.rs"]
mod common;

static mut BUFFER: [u8; 64] = [0; 64];

#[no_mangle]
extern "C" fn guest_main() -> ! {
    let buffer = addr_of_mut!(BUFFER).cast::<u8>();
    let offset = common::input() as u8;
    let mut index = 0_usize;
    while index < 64 {
        // SAFETY: index is below the allocated 64-byte buffer; execution is single-hart.
        unsafe { write_volatile(buffer.add(index), (index as u8).wrapping_add(offset)) };
        index += 1;
    }
    index = 0;
    let mut checksum = 0_u64;
    while index < 64 {
        // SAFETY: each indexed byte was initialized by the bounded store loop.
        let byte = unsafe { read_volatile(buffer.add(index)) };
        checksum = checksum.wrapping_add(u64::from(byte));
        index += 1;
    }
    common::finish(checksum)
}
