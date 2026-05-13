use criterion::Criterion;
use jolt_core::poly::commitment::dory::{DoryContext, DoryGlobals};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::math::Math;
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_openings::{BatchCommitmentSource, CommitmentScheme, CommitmentSource, SourceRow};
use jolt_poly::MultilinearPoly;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

struct BenchPolySource<'a>(&'a MultilinearPolynomial<Fr>);

impl CommitmentSource<Fr> for BenchPolySource<'_> {
    fn num_vars(&self) -> usize {
        MultilinearPoly::num_vars(self.0)
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        MultilinearPoly::evaluate(self.0, point)
    }

    fn for_each_row<V>(&self, sigma: usize, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
    {
        MultilinearPoly::for_each_row(self.0, sigma, &mut |row_index, row| {
            visit(row_index, SourceRow::FieldElements(row));
        });
    }

    fn is_one_hot(&self) -> bool {
        MultilinearPoly::is_one_hot(self.0)
    }

    fn for_each_one<V>(&self, mut visit: V)
    where
        V: FnMut(usize),
    {
        MultilinearPoly::for_each_one(self.0, &mut visit);
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        MultilinearPoly::fold_rows(self.0, left, sigma)
    }
}

struct BenchPolyBatch {
    ids: Vec<usize>,
    polys: Vec<MultilinearPolynomial<Fr>>,
}

impl BenchPolyBatch {
    fn new(polys: Vec<MultilinearPolynomial<Fr>>) -> Self {
        Self {
            ids: (0..polys.len()).collect(),
            polys,
        }
    }
}

impl BatchCommitmentSource<Fr> for BenchPolyBatch {
    type Id = usize;

    type Source<'a> = BenchPolySource<'a>;

    fn source_ids(&self) -> &[Self::Id] {
        &self.ids
    }

    fn num_vars(&self, id: Self::Id) -> usize {
        self.source(id).num_vars()
    }

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        BenchPolySource(&self.polys[id])
    }

    fn map_rows<R, V>(&self, sigma: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, Fr>) -> R + Send + Sync,
    {
        let mut rows = Vec::new();
        for (id_index, &id) in ids.iter().enumerate() {
            let mut seen_rows = 0;
            self.source(id).for_each_row(sigma, |row_index, row| {
                let mapped = visit(id, row);
                if id_index == 0 {
                    rows.push(vec![mapped]);
                } else {
                    rows.get_mut(row_index)
                        .expect("bench batch sources must have identical row shapes")
                        .push(mapped);
                }
                seen_rows += 1;
            });

            if id_index != 0 {
                assert_eq!(
                    seen_rows,
                    rows.len(),
                    "bench batch sources must have identical row counts",
                );
            }
        }
        rows
    }
}

fn benchmark_dory_dense(c: &mut Criterion, name: &str, k: usize, t: usize) {
    let globals = DoryGlobals::initialize_context(k, t, DoryContext::Main, None);
    let (setup, _) = DoryScheme::setup(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    // Generate leaves with percentage of ones
    let coeffs: Vec<u64> = (0..t).map(|_| rng.next_u64()).collect();
    let poly: MultilinearPolynomial<Fr> = MultilinearPolynomial::from(coeffs);

    c.bench_function(&format!("{name} Dory commit_rows"), |b| {
        b.iter(|| {
            let _ = globals;
            DoryScheme::commit(&poly, &setup);
        });
    });
}

fn benchmark_dory_one_hot_batch(c: &mut Criterion, name: &str, k: usize, t: usize) {
    let globals = DoryGlobals::initialize_context(k, t, DoryContext::Main, None);
    let (setup, _) = DoryScheme::setup(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let num_polys = 30;
    let batch = BenchPolyBatch::new(
        (0..num_polys)
            .map(|_| {
                let mut one_hot_coeffs = vec![0u64; t];
                let one_idx: usize = rng.gen_range(0..t);
                one_hot_coeffs[one_idx] = 1;
                MultilinearPolynomial::from(one_hot_coeffs)
            })
            .collect::<Vec<_>>(),
    );

    c.bench_function(&format!("{name} Dory one-hot commit"), |b| {
        b.iter(|| {
            let _ = globals;
            DoryScheme::commit_batch(&batch, batch.source_ids(), &setup);
        });
    });
}

fn benchmark_dory_mixed_batch(c: &mut Criterion, name: &str, k: usize, t: usize) {
    let globals = DoryGlobals::initialize_context(k, t, DoryContext::Main, None);
    let (setup, _) = DoryScheme::setup(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let num_polys = 30;
    let batch = BenchPolyBatch::new(
        (0..num_polys)
            .map(|_| {
                let one_hot = rng.gen_ratio(4, 5);
                if one_hot {
                    let mut one_hot_coeffs = vec![0u64; t];
                    let one_idx: usize = rng.gen_range(0..t);
                    one_hot_coeffs[one_idx] = 1;
                    MultilinearPolynomial::from(one_hot_coeffs)
                } else {
                    let coeffs: Vec<u64> = (0..t).map(|_| rng.next_u64()).collect();
                    MultilinearPolynomial::from(coeffs)
                }
            })
            .collect::<Vec<_>>(),
    );

    c.bench_function(&format!("{name} Dory mixed batch commit"), |b| {
        b.iter(|| {
            let _ = globals;
            DoryScheme::commit_batch(&batch, batch.source_ids(), &setup);
        });
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5))
        .sample_size(10);

    benchmark_dory_dense(&mut criterion, "Dory T = 2^25", 1 << 8, 1 << 25);
    benchmark_dory_one_hot_batch(&mut criterion, "Dory T = 2^25", 1 << 8, 1 << 25);
    benchmark_dory_mixed_batch(&mut criterion, "Dory T = 2^25", 1 << 8, 1 << 25);

    criterion.final_summary();
}
