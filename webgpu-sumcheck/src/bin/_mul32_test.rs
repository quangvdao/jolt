use rand::Rng;

fn mul32_emulated(a: u32, b: u32) -> (u32, u32) {
    let a0 = a & 0xFFFF;
    let a1 = a >> 16;
    let b0 = b & 0xFFFF;
    let b1 = b >> 16;

    let c0 = a0.wrapping_mul(b0);

    let c1a = a0.wrapping_mul(b1);
    let c1b = a1.wrapping_mul(b0);
    let (c1_low, carry1) = c1a.overflowing_add(c1b);
    let c1_high_extra = if carry1 { 1u32 } else { 0u32 };

    let c2 = a1.wrapping_mul(b1);

    let m1_low16 = c1_low & 0xFFFF;
    let m1_high_total = (c1_low >> 16) + (c1_high_extra << 16);

    let low_part = m1_low16 << 16;
    let (low_tmp, carry_low0) = c0.overflowing_add(low_part);
    let carry0 = if carry_low0 { 1u32 } else { 0u32 };

    let lo = low_tmp;
    let hi = c2.wrapping_add(m1_high_total).wrapping_add(carry0);

    (lo, hi)
}

fn main() {
    let mut rng = rand::thread_rng();
    for _ in 0..1_000_000 {
        let a: u32 = rng.gen();
        let b: u32 = rng.gen();
        let (lo, hi) = mul32_emulated(a, b);
        let prod = (a as u64) * (b as u64);
        let lo_ref = prod as u32;
        let hi_ref = (prod >> 32) as u32;
        if lo != lo_ref || hi != hi_ref {
            eprintln!(
                "mismatch: a={:#x}, b={:#x}, got=({:#x},{:#x}), expected=({:#x},{:#x})",
                a, b, lo, hi, lo_ref, hi_ref
            );
            std::process::exit(1);
        }
    }
    println!("mul32_emulated passed 1e6 random tests");
}
