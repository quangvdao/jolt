mod support;

use jolt_field::F128;
use jolt_rv64i_kernels::source::CycleSource;
use jolt_rv64i_kernels::synth::SynthProfile;
use rayon::prelude::*;

use support::example::{DenseProductCore, ExampleError};
use support::{run_core, RunnerError};

fn main() -> Result<(), RunnerError> {
    run_core(
        "example",
        &[SynthProfile::Local, SynthProfile::AllRows],
        |source| {
            let (a, b) = (0..source.cycles())
                .into_par_iter()
                .map(|cycle| {
                    (
                        F128::from_raw(u128::from(source.trace_word(0, cycle))),
                        F128::from_raw(u128::from(source.trace_word(1, cycle))),
                    )
                })
                .unzip();
            let core = DenseProductCore::new(a, b)?;
            let claim = core.initial_claim();
            Ok::<_, ExampleError>((core, claim))
        },
        |core, _point| core.final_values(),
    )
}
