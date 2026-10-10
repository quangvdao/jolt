//! Dense test oracle for the binary-field RV64I protocol (§12), exempt from the per-cycle budget.
//! Usable at approximately `t <= 10`, `a <= 10`, `b <= 10`; larger dense cubes are impractical.
pub mod bits_reduction;
pub mod bytecode;
pub mod ra_product;
pub mod read_checking;
pub mod routers;
pub mod spartan;
pub mod val_evaluation;
pub mod views;
