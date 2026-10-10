//! Live subset coefficients of packed outer monomials.

#[derive(Clone, Copy)]
pub struct WordInput {
    pub a: u64,
    pub b: u64,
    pub left: u32,
    pub right: u32,
}

/// One coefficient product of the outer monomial form, with live subset offsets.
/// Offsets are prepared below eight; both transforms affect the gathered result.
#[inline]
pub fn word_monomial(input: WordInput) -> u64 {
    let mut a = input.a;
    let mut b = input.b;
    for (shift, mask) in [
        (1, 0xaaaa_aaaa_aaaa_aaaa),
        (2, 0xcccc_cccc_cccc_cccc),
        (4, 0xf0f0_f0f0_f0f0_f0f0),
    ] {
        a ^= (a << shift) & mask;
        b ^= (b << shift) & mask;
    }
    let mut value = ((a >> input.left) & (b >> input.right)) & 0x0101_0101_0101_0101;
    value = (value | (value >> 7)) & 0x0003_0003_0003_0003;
    value |= value >> 14;
    (value | (value >> 28)) & 0xff
}
