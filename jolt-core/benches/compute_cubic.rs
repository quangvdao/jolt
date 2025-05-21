use ark_bn254::Fr;
use ark_std::{rand::Rng, test_rng};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use jolt_core::poly::sparse_interleaved_poly::{SparseCoefficient, SparseInterleavedPolynomial};
use jolt_core::poly::split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial};
use jolt_core::subprotocols::sumcheck::BatchedCubicSumcheck;
use jolt_core::utils::math::Math;
use jolt_core::utils::transcript::KeccakTranscript;

fn random_dense_coeffs<F: JoltField>(rng: &mut impl Rng, num_vars: usize) -> Vec<F> {
    std::iter::repeat_with(|| F::random(rng))
        .take(1 << num_vars)
        .collect()
}

fn random_sparse_coeffs<F: JoltField>(
    rng: &mut impl Rng,
    batch_size: usize,
    num_vars: usize,
    density: f64,
) -> Vec<Vec<SparseCoefficient<F>>> {
    (0..batch_size)
        .map(|batch_index| {
            let mut coeffs: Vec<SparseCoefficient<F>> = vec![];
            for i in 0..(1 << num_vars) {
                if rng.gen_bool(density) {
                    coeffs.push((batch_index * (1 << num_vars) + i, F::random(rng)).into())
                }
            }
            coeffs
        })
        .collect()
}

fn benchmark_dense_interleaved<F: JoltField>(c: &mut Criterion, num_vars: usize) {
    c.bench_function(
        &format!("DenseInterleavedPolynomial::compute_cubic {num_vars} variables"),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_dense_coeffs(&mut rng, num_vars);
                    let poly = DenseInterleavedPolynomial::new(coeffs);
                    let r_eq: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    let eq_poly = SplitEqPolynomial::new(&r_eq);
                    let claim = F::random(&mut rng);
                    (poly, eq_poly, claim)
                },
                |(poly, eq_poly, claim)| {
                    criterion::black_box(
                        BatchedCubicSumcheck::<F, KeccakTranscript>::compute_cubic(
                            &poly, &eq_poly, claim,
                        ),
                    );
                },
            );
        },
    );
}

fn benchmark_sparse_interleaved<F: JoltField>(
    c: &mut Criterion,
    batch_size: usize,
    num_vars: usize,
    density: f64,
) {
    c.bench_function(
        &format!(
            "SparseInterleavedPolynomial::compute_cubic {} x {} variables, {}% ones",
            batch_size,
            num_vars,
            (1.0 - density) * 100.0
        ),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_sparse_coeffs(&mut rng, batch_size, num_vars, density);
                    let poly = SparseInterleavedPolynomial::new(coeffs, batch_size << num_vars);
                    let r_eq: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take((batch_size << num_vars).next_power_of_two().log_2())
                        .collect();
                    let eq_poly = SplitEqPolynomial::new(&r_eq);
                    let claim = F::random(&mut rng);
                    (poly, eq_poly, claim)
                },
                |(poly, eq_poly, claim)| {
                    criterion::black_box(
                        BatchedCubicSumcheck::<F, KeccakTranscript>::compute_cubic(
                            &poly, &eq_poly, claim,
                        ),
                    );
                },
            );
        },
    );
}

fn benchmark_dense_comparison_variants<F: JoltField>(c: &mut Criterion, num_vars: usize) {
    let mut group = c.benchmark_group(&format!("Dense_Compare_Cubic_V{num_vars}"));

    // Setup shared data for this group
    let small_value_lookup_tables: <F as JoltField>::SmallValueLookupTables =
        F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables.clone());

    let mut rng = test_rng();
    let coeffs = random_dense_coeffs(&mut rng, num_vars);
    let poly = DenseInterleavedPolynomial::new(coeffs);
    let r_eq: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
        .take(num_vars)
        .collect();

    let eq_poly_orig = SplitEqPolynomial::new(&r_eq);
    let eq_poly_new = GruenSplitEqPolynomial::new(&r_eq);
    let claim = F::random(&mut rng);

    group.bench_function("compute_cubic_original_BatchedSumcheck", |b| {
        b.iter(|| {
            criterion::black_box(BatchedCubicSumcheck::<F, KeccakTranscript>::compute_cubic(
                &poly,
                &eq_poly_orig,
                claim,
            ));
        });
    });

    group.bench_function("compute_cubic_new_DensePoly", |b| {
        b.iter(|| {
            criterion::black_box(poly.compute_cubic_new(&eq_poly_new, claim));
        });
    });
    group.finish();
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(5));

    // benchmark_dense_interleaved::<Fr>(&mut criterion, 20);
    // benchmark_dense_interleaved::<Fr>(&mut criterion, 22);
    // benchmark_dense_interleaved::<Fr>(&mut criterion, 24);

    // benchmark_sparse_interleaved::<Fr>(&mut criterion, 64, 20, 0.1);
    // benchmark_sparse_interleaved::<Fr>(&mut criterion, 128, 20, 0.1);
    // benchmark_sparse_interleaved::<Fr>(&mut criterion, 64, 21, 0.1);
    // benchmark_sparse_interleaved::<Fr>(&mut criterion, 128, 21, 0.1);

    benchmark_dense_comparison_variants::<Fr>(&mut criterion, 20);
    benchmark_dense_comparison_variants::<Fr>(&mut criterion, 22);

    criterion.final_summary();
}
