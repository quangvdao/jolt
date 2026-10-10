//! Benchmark target for tail.

use std::io::{self, Write};

fn main() -> io::Result<()> {
    writeln!(
        io::stdout().lock(),
        "tail: benchmark is not yet implemented"
    )
}
