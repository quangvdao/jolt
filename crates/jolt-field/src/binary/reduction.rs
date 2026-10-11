pub(super) const MODULUS64: u64 = 0x1b;

// SHORT requires the first fold's high half to be zero. The butterfly wrappers
// enforce twiddle degree <=60: product degree <=123, high-half degree <=59,
// and folding by the degree-4 modulus then fits in 64 bits.
#[inline(always)]
pub(super) fn reduce64<P: Copy, R, const SHORT: bool>(
    product: P,
    mut fold_high: impl FnMut(P) -> P,
    finish_short: impl FnOnce(P, P) -> R,
    finish_full: impl FnOnce(P, P, P) -> R,
) -> R {
    let first = fold_high(product);
    if SHORT {
        finish_short(product, first)
    } else {
        let second = fold_high(first);
        finish_full(product, first, second)
    }
}
