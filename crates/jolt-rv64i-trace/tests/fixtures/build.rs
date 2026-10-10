use std::path::PathBuf;

fn main() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("linker.ld");
    println!("cargo:rustc-link-arg=-T{}", script.display());
    println!("cargo:rustc-link-arg=--no-relax");
    println!("cargo:rerun-if-changed=linker.ld");
}
