pub mod binding;
pub mod field_mul;
pub mod naive_sort;
pub mod prover_time;
pub mod rv64i_trace_adapt;
pub mod source_trace_gen;
pub mod trace_gen;

use rv64i_trace_adapt::{Rv64iTraceAdaptMeasurement, THREAD_POOLS};
use serde_json::Value;
use source_trace_gen::PROGRAMS;
use std::path::{Path, PathBuf};

/// Criterion's output directory, honoring absolute or work-directory-relative
/// `CARGO_TARGET_DIR`, with `target/criterion` as the default.
pub fn criterion_output_dir(work_dir: &Path) -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || work_dir.join("target"),
        |target| {
            let target = PathBuf::from(target);
            if target.is_absolute() {
                target
            } else {
                work_dir.join(target)
            }
        },
    );
    target.join("criterion")
}

/// Reads seconds from Criterion means, or the sum of source-trace medians.
/// `rv64i_trace_adapt` uses the three one-thread medians divided by their total
/// padded cycles, in nanoseconds per cycle. Counts come from benchmark metadata.
pub fn read_criterion_estimate(work_dir: &Path, bench_name: &str, baseline: &str) -> Option<f64> {
    read_criterion_estimate_at(&criterion_output_dir(work_dir), bench_name, baseline)
}

fn read_criterion_estimate_at(directory: &Path, bench_name: &str, baseline: &str) -> Option<f64> {
    if bench_name == "source_trace_gen" {
        return PROGRAMS.iter().try_fold(0.0, |total, (label, _, _)| {
            let path = directory
                .join(format!("source_trace_gen_{label}"))
                .join("source")
                .join(baseline)
                .join("estimates.json");
            let data = std::fs::read_to_string(path).ok()?;
            let json: Value = serde_json::from_str(&data).ok()?;
            let nanos = json.get("median")?.get("point_estimate")?.as_f64()?;
            Some(total + nanos / 1e9)
        });
    }
    if bench_name == "rv64i_trace_adapt" {
        let (nanos, cycles) =
            PROGRAMS
                .iter()
                .try_fold((0.0, 0_u64), |(nanos, cycles), (label, _, _)| {
                    let group = directory.join(format!("rv64i_trace_adapt_{label}"));
                    let data = std::fs::read_to_string(group.join("counts.json")).ok()?;
                    let counts: Value = serde_json::from_str(&data).ok()?;
                    let padded = counts.get("padded_cycles")?.as_u64()?;
                    if padded == 0 {
                        return None;
                    }
                    let data = std::fs::read_to_string(
                        group
                            .join("one_thread")
                            .join(baseline)
                            .join("estimates.json"),
                    )
                    .ok()?;
                    let json: Value = serde_json::from_str(&data).ok()?;
                    let estimate = json.get("median")?.get("point_estimate")?.as_f64()?;
                    Some((nanos + estimate, cycles.checked_add(padded)?))
                })?;
        return Some(nanos / cycles as f64);
    }
    let path = directory
        .join(bench_name)
        .join(baseline)
        .join("estimates.json");
    let data = std::fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&data).ok()?;
    let nanos = json.get("mean")?.get("point_estimate")?.as_f64()?;
    Some(nanos / 1e9)
}

fn read_adapter_measurements_at(
    directory: &Path,
    baseline: &str,
) -> Option<Vec<Rv64iTraceAdaptMeasurement>> {
    let mut measurements = Vec::new();
    for (program, _, _) in PROGRAMS {
        let group = directory.join(format!("rv64i_trace_adapt_{program}"));
        let data = std::fs::read_to_string(group.join("counts.json")).ok()?;
        let counts: Value = serde_json::from_str(&data).ok()?;
        let executed = counts.get("executed_rows")?.as_u64()?;
        let padded = counts.get("padded_cycles")?.as_u64()?;
        let facts_buffer_bytes = counts.get("facts_buffer_bytes")?.as_u64()?;
        if executed == 0 || padded == 0 {
            return None;
        }
        for pool in THREAD_POOLS {
            let data =
                std::fs::read_to_string(group.join(pool).join(baseline).join("estimates.json"))
                    .ok()?;
            let json: Value = serde_json::from_str(&data).ok()?;
            let nanos = json.get("median")?.get("point_estimate")?.as_f64()?;
            measurements.push(Rv64iTraceAdaptMeasurement {
                program,
                pool,
                ns_per_padded_cycle: nanos / padded as f64,
                ns_per_executed_row: nanos / executed as f64,
                facts_buffer_bytes,
                peak_incremental_allocated_bytes: counts
                    .get("peak_incremental_allocated_bytes")?
                    .get(pool)?
                    .as_u64()?,
            });
        }
    }
    Some(measurements)
}

#[cfg(test)]
mod tests {
    use super::{read_adapter_measurements_at, read_criterion_estimate_at};

    #[test]
    fn source_trace_objective_sums_source_medians() {
        let dir = tempfile::tempdir().unwrap();
        for (label, nanos) in [
            ("alu", 2_000_000),
            ("memory", 3_000_000),
            ("call_frame", 5_000_000),
        ] {
            let path = dir.path().join(format!(
                "target/criterion/source_trace_gen_{label}/source/new"
            ));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join("estimates.json"),
                format!(
                    r#"{{"median":{{"point_estimate":{nanos}}},"mean":{{"point_estimate":999}}}}"#
                ),
            )
            .unwrap();
        }
        assert_eq!(
            read_criterion_estimate_at(
                &dir.path().join("target/criterion"),
                "source_trace_gen",
                "new"
            ),
            Some(0.01)
        );
        assert_eq!(
            read_criterion_estimate_at(
                &dir.path().join("target/criterion"),
                "source_trace_gen",
                "missing"
            ),
            None
        );
    }

    #[test]
    fn rv64i_adapter_objective_normalizes_one_thread_medians_by_recorded_cycles() {
        let directory = tempfile::tempdir().unwrap();
        for (label, nanos, cycles) in [("alu", 40, 4), ("memory", 80, 8), ("call_frame", 120, 16)] {
            let group = directory.path().join(format!("rv64i_trace_adapt_{label}"));
            std::fs::create_dir_all(&group).unwrap();
            std::fs::write(
                group.join("counts.json"),
                format!(r#"{{"executed_rows":2,"padded_cycles":{cycles},"facts_buffer_bytes":1024,"peak_incremental_allocated_bytes":{{"one_thread":1100,"default_pool":1200}}}}"#),
            )
            .unwrap();
            for (pool, median) in [("one_thread", nanos), ("default_pool", nanos / 2)] {
                let path = group.join(pool).join("new");
                std::fs::create_dir_all(&path).unwrap();
                std::fs::write(
                    path.join("estimates.json"),
                    format!(r#"{{"median":{{"point_estimate":{median}}},"mean":{{"point_estimate":999}}}}"#),
                )
                .unwrap();
            }
        }
        let report = read_adapter_measurements_at(directory.path(), "new").unwrap();
        assert_eq!(report.len(), 6);
        assert_eq!(report[0].program, "alu");
        assert_eq!(report[0].pool, "one_thread");
        assert_eq!(report[0].ns_per_padded_cycle, 10.0);
        assert_eq!(report[0].ns_per_executed_row, 20.0);
        assert_eq!(report[0].facts_buffer_bytes, 1024);
        assert_eq!(report[0].peak_incremental_allocated_bytes, 1100);
        assert_eq!(report[5].program, "call_frame");
        assert_eq!(report[5].pool, "default_pool");
        assert_eq!(report[5].ns_per_padded_cycle, 3.75);
        assert_eq!(report[5].ns_per_executed_row, 30.0);
        assert_eq!(report[5].peak_incremental_allocated_bytes, 1200);
        assert!(read_adapter_measurements_at(directory.path(), "missing").is_none());
        assert_eq!(
            read_criterion_estimate_at(directory.path(), "rv64i_trace_adapt", "new"),
            Some(240.0 / 28.0)
        );
        assert_eq!(
            read_criterion_estimate_at(directory.path(), "rv64i_trace_adapt", "missing"),
            None
        );
        std::fs::write(
            directory.path().join("rv64i_trace_adapt_alu/counts.json"),
            r#"{"padded_cycles":0}"#,
        )
        .unwrap();
        assert_eq!(
            read_criterion_estimate_at(directory.path(), "rv64i_trace_adapt", "new"),
            None
        );
        assert!(read_adapter_measurements_at(directory.path(), "new").is_none());
    }
}
