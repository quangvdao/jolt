use super::{
    accumulator::{F128Accumulator, F192Accumulator, F64Accumulator},
    portable, F128, F192, F64,
};
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
))]
use super::{arch::Word, kernels};
use crate::{Accumulator, ExtField, WithAccumulator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

#[test]
fn kernel_matches_portable() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6269_6e61_7279_0012);
    let boundaries64 = [0, 1, 1 << 63, u64::MAX];
    let mut acc64 = F64Accumulator::default();
    let mut expected64 = 0;
    for (i, (a, b)) in boundaries64
        .into_iter()
        .flat_map(|a| boundaries64.map(|b| (a, b)))
        .chain((0..10_000).map(|_| (rng.next_u64(), rng.next_u64())))
        .enumerate()
    {
        if i < 1000 {
            acc64.fmadd(F64::from_raw(a), F64::from_raw(b));
            expected64 ^= portable::multiply64(a, b);
        }
        assert_eq!(portable::reduce64(portable::embed64(a)), a);
        let product = portable::product64(a, b);
        assert_eq!(portable::reduce64(product), portable::multiply64(a, b));
        assert_eq!(
            portable::reduce_accumulator64(product),
            portable::multiply64(a, b)
        );
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(
                kernels::canonical64(kernels::embed64(a)),
                portable::embed64(a)
            );
            assert_eq!(kernels::canonical64(kernels::product64(a, b)), product);
            assert_eq!(
                kernels::reduce64(kernels::product64(a, b)),
                portable::multiply64(a, b)
            );
            assert_eq!(
                kernels::reduce_accumulator64(kernels::product64(a, b)),
                portable::multiply64(a, b)
            );
            assert_eq!(kernels::multiply64(a, b), portable::multiply64(a, b));
            assert_eq!(kernels::square64(a), portable::square64(a));
        }
    }

    assert_eq!(acc64.reduce().to_raw(), expected64);

    let boundaries128 = [0, 1, 1 << 127, u128::MAX];
    let mut acc128 = F128Accumulator::default();
    let mut expected128 = 0;
    for (i, (a, b)) in boundaries128
        .into_iter()
        .flat_map(|a| boundaries128.map(|b| (a, b)))
        .chain((0..10_000).map(|_| {
            let [a0, a1, b0, b1] = std::array::from_fn(|_| rng.next_u64());
            (
                u128::from(a0) | (u128::from(a1) << 64),
                u128::from(b0) | (u128::from(b1) << 64),
            )
        }))
        .enumerate()
    {
        if i < 1000 {
            acc128.fmadd(F128::from_raw(a), F128::from_raw(b));
            expected128 ^= portable::multiply128(a, b);
        }
        assert_eq!(portable::reduce128(portable::embed128(a)), a);
        let product = portable::product128(a, b);
        assert_eq!(portable::accumulate128([0; 2], a, b), product);
        assert_eq!(portable::reduce128(product), portable::multiply128(a, b));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(
                kernels::canonical128(kernels::embed128(a)),
                portable::embed128(a)
            );
            assert_eq!(kernels::canonical128(kernels::product128(a, b)), product);
            assert_eq!(
                kernels::canonical128(kernels::accumulate128(Default::default(), a, b)),
                product
            );
            for (raw, reduced) in kernels::variants128(a, b) {
                assert_eq!(raw, product);
                for value in reduced {
                    assert_eq!(value, portable::multiply128(a, b));
                }
            }
            assert_eq!(
                kernels::reduce128(kernels::product128(a, b)),
                portable::multiply128(a, b)
            );
            assert_eq!(kernels::multiply128(a, b), portable::multiply128(a, b));
            assert_eq!(kernels::square128(a), portable::square128(a));
        }
    }

    assert_eq!(acc128.reduce().to_raw(), expected128);

    let boundaries192 = [[0; 3], [1, 0, 0], [0, 0, 1 << 63], [u64::MAX; 3]];
    let mut acc192 = F192Accumulator::default();
    let mut expected192 = [0; 3];
    for (i, (a, b)) in boundaries192
        .into_iter()
        .flat_map(|a| boundaries192.map(|b| (a, b)))
        .chain((0..10_000).map(|_| {
            (
                std::array::from_fn(|_| rng.next_u64()),
                std::array::from_fn(|_| rng.next_u64()),
            )
        }))
        .enumerate()
    {
        if i < 1000 {
            acc192.fmadd(
                F192::from_base_fn(|i| F64::from_raw(a[i])),
                F192::from_base_fn(|i| F64::from_raw(b[i])),
            );
            for (word, product) in expected192.iter_mut().zip(portable::multiply192(a, b)) {
                *word ^= product;
            }
        }
        assert_eq!(portable::reduce192(portable::embed192(a)), a);
        let product = portable::product192(a, b);
        assert_eq!(portable::reduce192(product), portable::multiply192(a, b));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            assert_eq!(
                kernels::canonical192(kernels::embed192(a)),
                portable::embed192(a)
            );
            assert_eq!(kernels::canonical192(kernels::product192(a, b)), product);
            assert_eq!(
                kernels::reduce192(kernels::product192(a, b)),
                portable::multiply192(a, b)
            );
            assert_eq!(kernels::multiply192(a, b), portable::multiply192(a, b));
            assert_eq!(kernels::square192(a), portable::square192(a));
        }
    }
    assert_eq!(
        acc192.reduce(),
        F192::from_base_fn(|i| F64::from_raw(expected192[i]))
    );

    // (2^64-1)*0x1b = (0x9<<64)^0x9 and 0x9*0x1b = 0xc3 in GF(2)[x].
    let all64 = 0xffff_ffff_ffff_ff35;
    // (2^128-1)*0x87 = (0x7d<<128)^0x7d and 0x7d*0x87 = 0x3ff3.
    let all128 = 0xffff_ffff_ffff_ffff_ffff_ffff_ffff_c071;
    assert_eq!(portable::reduce64(u128::MAX), all64);
    assert_eq!(portable::reduce_accumulator64(u128::MAX), all64);
    assert_eq!(portable::reduce128([u128::MAX; 2]), all128);
    assert_eq!(portable::reduce192([u128::MAX; 3]), [all64; 3]);
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    {
        let ones = Word::from_u128(u128::MAX);
        assert_eq!(kernels::reduce64(ones.into_unreduced64()), all64);
        assert_eq!(
            kernels::reduce_accumulator64(ones.into_unreduced64()),
            all64
        );
        for value in kernels::reductions128([ones, Word::from_u64(0), ones]) {
            assert_eq!(value, all128);
        }
        assert_eq!(kernels::reduce192([ones; 3]), [all64; 3]);
        let storage_ones = [ones; 3];
        // The overlapping middle 128 bits cancel when all three words are ones.
        let outer_ones = [
            0x0000_0000_0000_0000_ffff_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff_0000_0000_0000_0000,
        ];
        assert_eq!(kernels::canonical128(storage_ones), outer_ones);
        for value in kernels::reductions128(storage_ones) {
            assert_eq!(value, portable::reduce128(outer_ones));
        }
    }
}

#[test]
fn concrete_accumulator_types() {
    fn assert_types<F, A>()
    where
        F: WithAccumulator<
            Accumulator = A,
            SmallScalarAccumulator = A,
            SignedProductAccumulator = A,
        >,
        A: Accumulator<Element = F>,
    {
    }

    assert_types::<F64, F64Accumulator>();
    assert_types::<F128, F128Accumulator>();
    assert_types::<F192, F192Accumulator>();
}

#[test]
fn f128_word_products_match_shift_xor() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x776f_7264_0012);
    let values: Vec<_> = [0, u128::MAX]
        .into_iter()
        .chain((0..128).map(|bit| 1u128 << bit))
        .chain((0..10_000).map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64)))
        .collect();
    assert_eq!(F128::from_raw(1 << 127).mul_x(), F128::from_raw(0x87));
    for a in values {
        let field = F128::from_raw(a);
        assert_eq!(field.mul_x(), field * F128::from_raw(2));
        for word in [0, 1, 2, 1 << 63, u64::MAX, rng.next_u64()] {
            let expected = portable::multiply128(a, u128::from(word));
            assert_eq!(
                field.mul_word(word),
                field * F128::from_raw(u128::from(word))
            );
            assert_eq!(field.mul_word(word).to_raw(), expected);
            assert_eq!(portable::multiply128_word(a, word), expected);
            assert_eq!(
                portable::reduce128(portable::accumulate128_word([0; 2], a, word)),
                expected
            );
            #[cfg(any(
                all(target_arch = "aarch64", target_feature = "aes"),
                all(target_arch = "x86_64", target_feature = "pclmulqdq")
            ))]
            {
                assert_eq!(kernels::multiply128_word(a, word), expected);
                assert_eq!(
                    kernels::canonical128(kernels::accumulate128_word(Default::default(), a, word)),
                    portable::product128(a, u128::from(word))
                );
            }
        }
    }
}

#[test]
fn f128_word_accumulator_mixed_merges() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6d69_7865_645f_776f);
    for count in [1, 2, 20] {
        let mut accumulators = [F128Accumulator::default(); 2];
        let mut portable_accumulators = [[0; 2]; 2];
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        let mut kernel_accumulators = [Default::default(); 2];
        let mut expected = F128::from_raw(0);
        for list in 0..2 {
            for term in 0..count {
                let a =
                    F128::from_raw(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64));
                let b =
                    F128::from_raw(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64));
                let word = rng.next_u64();
                let add = F128::from_raw(u128::from(rng.next_u64()));
                match (term + list) % 3 {
                    0 => {
                        accumulators[list].fmadd_word(a, word);
                        portable_accumulators[list] = portable::accumulate128_word(
                            portable_accumulators[list],
                            a.to_raw(),
                            word,
                        );
                        expected += a * F128::from_raw(u128::from(word));
                    }
                    1 => {
                        accumulators[list].fmadd(a, b);
                        portable_accumulators[list] = portable::accumulate128(
                            portable_accumulators[list],
                            a.to_raw(),
                            b.to_raw(),
                        );
                        expected += a * b;
                    }
                    _ => {
                        accumulators[list].add(add);
                        for (lane, value) in portable_accumulators[list]
                            .iter_mut()
                            .zip(portable::embed128(add.to_raw()))
                        {
                            *lane ^= value;
                        }
                        expected += add;
                    }
                }
                #[cfg(any(
                    all(target_arch = "aarch64", target_feature = "aes"),
                    all(target_arch = "x86_64", target_feature = "pclmulqdq")
                ))]
                {
                    let acc = &mut kernel_accumulators[list];
                    match (term + list) % 3 {
                        0 => *acc = kernels::accumulate128_word(*acc, a.to_raw(), word),
                        1 => *acc = kernels::accumulate128(*acc, a.to_raw(), b.to_raw()),
                        _ => {
                            for (lane, value) in acc.iter_mut().zip(kernels::embed128(add.to_raw()))
                            {
                                *lane ^= value;
                            }
                        }
                    }
                }
            }
        }
        assert_ne!(expected, F128::from_raw(0));
        let [mut left, right] = accumulators;
        left.merge(right);
        assert_eq!(left.reduce(), expected);
        let [left, right] = portable_accumulators;
        assert_eq!(
            portable::reduce128(std::array::from_fn(|i| left[i] ^ right[i])),
            expected.to_raw()
        );
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            let [left, right] = kernel_accumulators;
            assert_eq!(
                kernels::reduce128(std::array::from_fn(|i| left[i] ^ right[i])),
                expected.to_raw()
            );
        }
    }
}

// Polynomial long multiplication and division, independent of the field kernels.
fn schoolbook_base(mut a: u64, mut b: u64, bits: u32, modulus: u64) -> u64 {
    let top = 1u64 << (bits - 1);
    let mask = u64::MAX >> (64 - bits);
    let mut result = 0;
    for _ in 0..bits {
        if b & 1 != 0 {
            result ^= a;
        }
        let carry = a & top != 0;
        a = (a << 1) & mask;
        if carry {
            a ^= modulus;
        }
        b >>= 1;
    }
    result
}

fn schoolbook_cubic(a: [u64; 3], b: [u64; 3], bits: u32, modulus: u64) -> [u64; 3] {
    let mut coefficients = [0; 5];
    for (i, a) in a.into_iter().enumerate() {
        for (j, b) in b.into_iter().enumerate() {
            coefficients[i + j] ^= schoolbook_base(a, b, bits, modulus);
        }
    }
    for degree in (3..=4).rev() {
        coefficients[degree - 3] ^= coefficients[degree];
        coefficients[degree - 2] ^= coefficients[degree];
    }
    [coefficients[0], coefficients[1], coefficients[2]]
}

#[test]
fn f192_base_accumulator_schoolbook() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6261_7365_0019);
    let edges = [0, 1, u64::MAX, 1 << 63];
    let supported = (0..3).flat_map(|position| {
        edges.into_iter().flat_map(move |a| {
            edges.map(|b| {
                let mut coefficients = [0; 3];
                coefficients[position] = a;
                (coefficients, b)
            })
        })
    });
    let mut acc = F192Accumulator::default();
    let mut expected_sum = [0; 3];
    let dense_edges = [[0; 3], [1, 0, 0], [u64::MAX; 3], [0, 0, 1 << 63]]
        .into_iter()
        .flat_map(|a| edges.map(|b| (a, b)));
    for (a, b) in supported
        .chain(dense_edges)
        .chain((0..1024).map(|_| (std::array::from_fn(|_| rng.next_u64()), rng.next_u64())))
    {
        let e = F192::from_base_fn(|i| F64::from_raw(a[i]));
        let base = F64::from_raw(b);
        let expected = schoolbook_cubic(a, [b, 0, 0], 64, 0x1b);
        let field_expected = F192::from_base_fn(|i| F64::from_raw(expected[i]));
        let mut single = F192Accumulator::default();
        single.fmadd_base(e, base);
        assert_eq!(single.reduce(), field_expected);
        assert_eq!(e.mul_base(base), field_expected);
        acc.fmadd_base(e, base);
        for (sum, product) in expected_sum.iter_mut().zip(expected) {
            *sum ^= product;
        }
        let portable_product = portable::product192_base(a, b);
        assert_eq!(portable::reduce192(portable_product), expected);
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            let product = kernels::product192_base(a, b);
            assert_eq!(kernels::canonical192(product), portable_product);
            assert_eq!(kernels::reduce192(product), expected);
        }
    }
    assert_eq!(
        acc.reduce(),
        F192::from_base_fn(|i| F64::from_raw(expected_sum[i]))
    );
}

#[test]
fn f192_base_pair_schoolbook() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7061_6972_0019);
    let edges = [0, 1, u64::MAX, 1 << 63];
    let mut cases = Vec::new();
    for i in 0..3 {
        for j in 0..2 {
            for a in edges {
                for b in edges {
                    let mut e = [0; 3];
                    let mut v = [0; 2];
                    e[i] = a;
                    v[j] = b;
                    cases.push((e, v));
                }
            }
        }
    }
    for a in [[0; 3], [1, 0, 0], [u64::MAX; 3], [0, 0, 1 << 63]] {
        for b in [[0; 2], [1, 0], [u64::MAX; 2], [0, 1 << 63]] {
            cases.push((a, b));
        }
    }
    cases.extend((0..1024).map(|_| {
        (
            std::array::from_fn(|_| rng.next_u64()),
            std::array::from_fn(|_| rng.next_u64()),
        )
    }));
    let mut acc = F192Accumulator::default();
    let mut expected_sum = [0; 3];
    for (a, b) in cases {
        let e = F192::from_base_fn(|i| F64::from_raw(a[i]));
        let v = b.map(F64::from_raw);
        let expected = schoolbook_cubic(a, [b[0], b[1], 0], 64, 0x1b);
        let field_expected = F192::from_base_fn(|i| F64::from_raw(expected[i]));
        assert_eq!(e.mul_base_pair(v), field_expected);
        let mut single = F192Accumulator::default();
        single.fmadd_base_pair(e, v);
        assert_eq!(single.reduce(), field_expected);
        acc.fmadd_base_pair(e, v);
        for (sum, product) in expected_sum.iter_mut().zip(expected) {
            *sum ^= product;
        }
        let portable_product = portable::product192_base_pair(a, b);
        assert_eq!(portable_product, portable::product192(a, [b[0], b[1], 0]));
        assert_eq!(portable::multiply192_base_pair(a, b), expected);
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            let product = kernels::product192_base_pair(a, b);
            assert_eq!(kernels::canonical192(product), portable_product);
            assert_eq!(kernels::multiply192_base_pair(a, b), expected);
        }
    }
    assert_eq!(
        acc.reduce(),
        F192::from_base_fn(|i| F64::from_raw(expected_sum[i]))
    );
}

#[test]
fn f192_base_pair_exhaustive_toy() {
    for e in 0..4096 {
        let a = [e & 15, (e >> 4) & 15, e >> 8];
        for v in 0..256 {
            let b = [v & 15, v >> 4];
            let d0 = schoolbook_base(a[0], b[0], 4, 0x3);
            let d1 = schoolbook_base(a[1], b[1], 4, 0x3);
            let c01 = schoolbook_base(a[0] ^ a[1], b[0] ^ b[1], 4, 0x3) ^ d0 ^ d1;
            let c02 = schoolbook_base(a[2], b[0], 4, 0x3);
            let c12 = schoolbook_base(a[2], b[1], 4, 0x3);
            assert_eq!(
                [d0 ^ c12, c01 ^ c12, d1 ^ c02],
                schoolbook_cubic(a, [b[0], b[1], 0], 4, 0x3),
                "e={e:x}, v={v:x}"
            );
        }
    }
}

#[test]
fn f192_mul_y_and_base_pair_composition() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6d75_6c79_0019);
    let edges = [[0; 3], [1, 0, 0], [u64::MAX; 3], [0, 0, 1 << 63]]
        .into_iter()
        .flat_map(|a| [[0; 2], [1, 0], [u64::MAX; 2], [0, 1 << 63]].map(|b| (a, b)));
    for (a, b) in edges.chain((0..1024).map(|_| {
        (
            std::array::from_fn(|_| rng.next_u64()),
            std::array::from_fn(|_| rng.next_u64()),
        )
    })) {
        let e = F192::from_base_fn(|i| F64::from_raw(a[i]));
        let b = b.map(F64::from_raw);
        let expected_y = [a[2], a[0] ^ a[2], a[1]];
        assert_eq!(
            e.mul_y(),
            F192::from_base_fn(|i| F64::from_raw(expected_y[i]))
        );
        assert_eq!(expected_y, schoolbook_cubic(a, [0, 1, 0], 64, 0x1b));
        let mut acc = F192Accumulator::default();
        acc.fmadd_base(e, b[0]);
        acc.fmadd_base(e.mul_y(), b[1]);
        assert_eq!(acc.reduce(), e.mul_base_pair(b));
    }
}

#[test]
fn f192_specialized_accumulator_mixed_merges() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6d69_7865_6419);
    let mut accumulators = [F192Accumulator::default(); 2];
    let mut expected = [0; 3];
    for acc in &mut accumulators {
        for term in 0..1024 {
            let a = std::array::from_fn(|_| rng.next_u64());
            let b = std::array::from_fn(|_| rng.next_u64());
            let e = F192::from_base_fn(|i| F64::from_raw(a[i]));
            let v = b.map(F64::from_raw);
            let product = match term % 4 {
                0 => {
                    acc.fmadd_base(e, v[0]);
                    schoolbook_cubic(a, [b[0], 0, 0], 64, 0x1b)
                }
                1 => {
                    acc.fmadd_base_pair(e, [v[0], v[1]]);
                    schoolbook_cubic(a, [b[0], b[1], 0], 64, 0x1b)
                }
                2 => {
                    acc.fmadd(e, F192::from_base_fn(|i| v[i]));
                    schoolbook_cubic(a, b, 64, 0x1b)
                }
                _ => {
                    acc.add(e);
                    a
                }
            };
            for (sum, product) in expected.iter_mut().zip(product) {
                *sum ^= product;
            }
        }
    }
    let [mut left, right] = accumulators;
    left.merge(right);
    assert_eq!(
        left.reduce(),
        F192::from_base_fn(|i| F64::from_raw(expected[i]))
    );
}
