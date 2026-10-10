//! Binary-field RV64I stage wrappers and kernel registries.
pub mod stage1;
pub mod stage2;
pub mod stage3a;
pub mod stage3b;
pub mod stage4;
pub mod stage5;
pub mod stage6a;
pub mod stage6b;

#[cfg(feature = "test-utils")]
pub mod fixture;
