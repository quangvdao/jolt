These bare Rust programs use assembly startup, their own linker script and
volatile I/O, with no SDK runtime. `guests.rs` builds all three with Rust 1.95,
opt-level 2, LTO and aborting panics, then checks each complete ELF under Strict:

```sh
CARGO_TARGET_DIR=target/rv64i-fixtures RUSTFLAGS='-C target-feature=-m,-a,-c,-zmmul,-zca,-zaamo,-zalrsc' cargo build --manifest-path crates/jolt-rv64i-trace/tests/fixtures/Cargo.toml --release --bins --target riscv64imac-unknown-none-elf
```

`-m,-a,-c` disable multiplication/division, atomics and compressed instructions;
their independently enabled subsets need `-zmmul,-zca,-zaamo,-zalrsc` as well.
The source avoids arithmetic/runtime calls from precompiled M/C libraries;
volatile accumulation prevents multiplication lowering, and terminal Rust loops
avoid the unreachable trap following `asm!(..., options(noreturn))`.

```sh
cargo nextest run -p jolt-rv64i-trace --features emulator --cargo-quiet -E 'binary(guests)'
```
