//! Adapter layer gate; native flags match the kernels bench.
pub mod support;

#[expect(clippy::print_stdout, reason = "dynamic mix is benchmark output")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::witness::WitnessFixture::new(20)?;
    let _ = fixture.checked()?;
    println!("{}", fixture.mix);
    Ok(())
}
