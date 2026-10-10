//! Benchmark target for routers.

use std::io::{self, Write};

fn main() -> io::Result<()> {
    writeln!(
        io::stdout().lock(),
        "routers: benchmark is not yet implemented"
    )
}
