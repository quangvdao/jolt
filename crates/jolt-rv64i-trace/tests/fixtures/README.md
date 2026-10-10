These three bare Rust programs use assembly startup, their own linker script,
volatile output and termination stores, and no SDK runtime. `guests.rs` builds
the package once with Rust 1.95:

```sh
CARGO_TARGET_DIR=/Users/quangdao/Documents/SNARKs/jolt-wt/trace-target/rv64i-fixtures RUSTFLAGS='-C target-feature=-m,-a,-c' cargo build --manifest-path crates/jolt-rv64i-trace/tests/fixtures/Cargo.toml --release --bins --target riscv64imac-unknown-none-elf
```

The complete linked `input-loop` ELF fails strict decoding with
`IllegalCompressedInstruction { address: 0x80000018 }`; its text also contains
M instructions. The required C fallback compiler, `riscv-none-elf-gcc`, is absent
on this machine. The test remains ignored with this reason, preserving its
strict gate and all subsequent row, output and memory checks. No fixture has
been accepted through a more permissive decode mode.

To rerun the gate after resolving the toolchain limitation:

```sh
CARGO_TARGET_DIR=/Users/quangdao/Documents/SNARKs/jolt-wt/trace-target cargo nextest run -p jolt-rv64i-trace --features emulator --run-ignored only --cargo-quiet -E 'binary(guests)'
```
