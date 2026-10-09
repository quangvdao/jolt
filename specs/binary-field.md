# Spec: Binary Fields for the Jolt Trait Spine

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

A binary-field Jolt proves RV64 execution with sum-checks over a field of characteristic 2, where XOR, AND and shifts of machine words are low-degree. Jolt's protocol crates (`jolt-poly`, `jolt-sumcheck`, `jolt-transcript`, `jolt-claims`) are generic over `JoltField`, and no type in the workspace has characteristic 2. This spec adds a third backend to `jolt-field`, behind a `binary` feature, with three binary fields that implement `JoltField`, so that those crates can be instantiated over them without a fork. It is the first of nine milestones; the contract layer gains one capability trait and no existing backend changes.

## Intent

### Goal

Add a `binary` backend module to `jolt-field` with three element types that implement `JoltField`, and one capability contract at the crate root for what is specific to characteristic 2.

| Type | Field | Representation | Fixed by |
|---|---|---|---|
| `F64` | $\mathbb F_2[x]/(x^{64}+x^4+x^3+x+1)$ | one `u64`, bit $i$ is the coefficient of $x^i$ | leanVM's PCS, which is written over this concrete field |
| `F192` | `F64`$[y]/(y^3+y+1)$ | `[F64; 3]`, index $i$ is the coefficient of $y^i$ | leanVM's PCS challenge field |
| `F128` | $\mathbb F_2[x]/(x^{128}+x^7+x^2+x+1)$ | one `u128`, bit $i$ is the coefficient of $x^i$ | Akita's `BinaryField128` |

`F192` implements `ExtField<F64>` with `DEGREE = 3`. `F128` is not an extension of `F64` in this representation and implements no `ExtField`.

The capability trait `BinaryField: JoltField` carries:

- `const DEGREE: u32`, the degree over $\mathbb F_2$;
- `fn basis(i: u32) -> Self`, the $i$-th element of the fixed $\mathbb F_2$-basis $1,x,\dots$ (for `F192`, $x^ay^b$ at index $a+64b$);
- `fn from_bits(bits: u64) -> Self`, the $\mathbb F_2$-linear embedding of a 64-bit word against the first 64 basis elements. This is how a machine word enters the field, and it is distinct from `Ring::from_u64`.

### Invariants

1. **Field axioms.** Each type is a field of the stated order: addition is XOR of representations, multiplication is polynomial multiplication reduced by the stated modulus, `inverse` returns `None` exactly for zero and otherwise `a * a.inverse() == 1`.
2. **Agreement with the back ends.** `F64` and `F192` multiplication, squaring and inversion equal leanVM's `F64` and `F192` at revision `48a90420`; `F128` equals Akita's `BinaryField128`. The byte encodings below equal theirs, so the commitment adapters convert without arithmetic.
3. **`Ring` integer maps are the ring homomorphism.** `from_u64(v)`, `from_i64`, `from_u128`, `from_i128` return `one()` if `v` is odd and `zero()` otherwise. Hence `from_u64(2) == zero()`, `pow2(k) == zero()` for `k > 0`, and `mul_pow_2(k)` is zero for `k > 0`. `jolt-sumcheck`'s `BooleanHypercube::round_sum_coefficients` relies on `from_u64(2)` being the ring image of 2.
4. **`two_inv` and `half` panic.** They are unreachable for a correct caller in characteristic 2 and keep the default `expect("field has characteristic two")`. The sum-check batch prelude that calls them is replaced in milestone 2.
5. **Canonical encoding.** `NUM_BYTES` is 8, 24 and 16; `MODULUS_BITS` is 65, 193 and 129 (bit length of the order). `to_bytes_le` writes the representation little-endian (`F192`: the three `F64` coefficients in index order). Every byte string of length `NUM_BYTES` is canonical, so `from_bytes_le_checked` fails only on a wrong length. `from_bytes_le_reduced` reads the first `NUM_BYTES` bytes, zero-padding a short input and ignoring the rest; it is a truncation, not a reduction of an integer.
6. **Bit-pattern conversions.** `to_u128_checked` returns the representation when it fits (always for `F64` and `F128`; for `F192` when the $y^2$ coefficient is zero). `from_u128_checked` and `from_u128_reduced` are the inverse bit-pattern maps (`F64` rejects, respectively truncates, above 64 bits). These are not integer conversions and no caller may treat them as such; `num_bits` is the bit length of the representation.
7. **Challenges.** `from_challenge_bytes` and `from_scalar_challenge_bytes` are `from_bytes_le_reduced`. A challenge is uniform on the field only if the transcript supplies at least `NUM_BYTES` bytes; with Jolt's 16-byte squeeze that holds for `F64` and `F128` and not for `F192`. The 24-byte transcript is a later milestone and out of scope here.
8. **Sampling.** `random` reads exactly `NUM_BYTES` bytes from the RNG and decodes them; no rejection is needed.
9. **Serde.** Wire serialization uses `impl_serde_bytes!` over the canonical bytes, as the other backends do.

No `jolt-eval` invariant changes: the crate has no caller yet.

### Non-Goals

- Carry-less multiply intrinsics, packed backends and deferred-reduction accumulators. `WithAccumulator` uses `NaiveAccumulator` for all three associated types; a portable shift-and-XOR or table multiply is sufficient here. Performance kernels come with milestone 7.
- A degree-2 extension of `F64` as the sum-check field of the hash build. That choice is open (see Alternatives) and is added, if taken, with its first caller.
- Changes to the existing spine traits or to the `bn254` and `solinas` backends, and changes to `jolt-sumcheck` or `jolt-transcript`. Splitting the integer embedding out of `Ring` (see Alternatives) is not part of this spec.
- The fixed one-hot generator and any protocol constant; they are added with the relation that uses them.
- Conversions to leanVM and Akita types. They live in the commitment adapters, which own those dependencies.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-field` gains a `binary` feature, off by default, that enables `mod binary` and re-exports `F64`, `F128`, `F192`. The backend references no other backend and contains no `unsafe`. `BinaryField` is defined in the contract layer, unconditionally.
- [ ] `cargo clippy -p jolt-field --no-default-features --features binary --all-targets -- -D warnings` passes, as does the same with `bn254,solinas,binary`.
- [ ] `F64`, `F128` and `F192` satisfy `JoltField`, checked by a compile-time bound assertion.
- [ ] Fixed test vectors (at least 16 per type and operation, including zero, one, the top basis element and all-ones) for multiplication, squaring and inversion match values produced by leanVM `48a90420` and Akita's `BinaryField128`; the generating commit is recorded next to the vectors.
- [ ] A bitwise reference multiply written from the definition (schoolbook product, then reduction by the modulus one bit at a time) agrees with the production multiply on 10,000 seeded random pairs per type.
- [ ] Algebraic properties on seeded random inputs: associativity, commutativity, distributivity, `a + a == 0`, `a.square() == a * a`, `a * a.inverse() == 1` for nonzero `a`, and the Frobenius identity $a^{2^n}=a$ with $n$ = 64, 128, 192.
- [ ] `from_u64(2)`, `pow2(1)` and `one().mul_pow_2(1)` are zero; `from_u64(3) == one()`; `from_i64(-1) == one()`.
- [ ] Byte and serde round-trips for random elements; `from_bytes_le_checked` rejects every wrong length.
- [ ] `basis` and `from_bits` agree: `from_bits(w)` equals the sum of `basis(i)` over the set bits of `w`, and `F192::basis(64 * b + a)` has `basis` image only in coefficient `b`.
- [ ] `cargo fmt --check` passes.

### Testing Strategy

New tests only, in `crates/jolt-field/tests/binary_*.rs`, run with `cargo nextest run -p jolt-field --features binary --cargo-quiet`. Existing `jolt-field` tests must pass unchanged with and without the feature. There is no host or ZK mode distinction at this layer. The fixed vectors are the independent ground truth; the bitwise reference is the definition of the field and not a copy of the production code path.

### Performance

None claimed. This milestone is the reference arithmetic; throughput is measured when the optimised kernels land. No existing `jolt-eval` objective moves.

## Design

### Architecture

`JoltField` is a blanket bundle of `Field`, `CanonicalEncoding`, `WithAccumulator`, serde and `MaybeAllocative` (`crates/jolt-field/src/algebra.rs:518–532`). None of the supertraits asserts a prime order, and `CanonicalEncoding` already documents extension fields, so the binary fields implement the existing spine and the protocol crates need no second trait hierarchy. What the spine cannot express, the embedding of bits and words, goes in `BinaryField`, a capability contract at the crate root beside `PseudoMersenne` and `ExtField`.

The backend lives in `jolt-field` because that is the crate's stated architecture: a contract layer at the root and feature-gated backend modules that never reference each other (`crates/jolt-field/src/lib.rs`, "Architecture: contracts and backends"). `bn254` and `solinas` are the two existing backends; `binary` is the third. The feature is off by default, so no existing build compiles it, and the crate's byte-compatibility invariants, which are stated per backend, are untouched.

The representations are fixed by the two commitment back ends. leanVM's PCS is written over its concrete `F64` and `F192`; Akita's field switch is sealed over `BinaryField128`. Using the same moduli and bit order makes the adapter a reinterpretation of bytes.

The places where the trait surface assumes odd characteristic, and what this crate does about each:

| Method | Behaviour here | Consumer to fix later |
|---|---|---|
| `two_inv`, `half` | panic (default) | `prove_batch` in `jolt-sumcheck`, milestone 2 |
| `from_u64` and the other integer maps | parity | none: this is what `BooleanHypercube` needs; words use `from_bits` |
| `pow2`, `mul_pow_2` | zero for a positive exponent | batch padding, milestone 2 |
| `to_u128_checked`, `from_u128_*` | bit-pattern maps | any caller treating them as integer conversions |
| `from_challenge_bytes` | truncating decode | 24-byte transcript for `F192`, later milestone |

### Alternatives Considered

- **A separate trait hierarchy for binary fields.** Rejected: it forks `jolt-poly`, `jolt-sumcheck`, `jolt-transcript` and `jolt-claims`, all of which are bounded by `Field`, `Ring` or `JoltField` and are otherwise reusable.
- **A separate crate `jolt-binary-field`.** Rejected: `jolt-field` already separates contracts from backends by feature, and a second crate would split the contract layer (`BinaryField` in one crate, the spine in another) for no build-time gain. Plonky3 does use one crate per field, but its `p3-field` holds only traits; `jolt-field` holds its backends, as Binius64's `field` crate and leanVM's `primitives::field` do.
- **Depending on leanVM's and Akita's field types directly.** Rejected: it would put two git dependencies under `jolt-field`, which is in the verifier closure, and Akita's type is behind a sealed trait. The adapters own the conversion.
- **Reshaping the spine first, as Plonky3 does.** Plonky3's base trait is `PrimeCharacteristicRing`, where integers enter through the prime subfield and integer-valued semantics sit on `PrimeField`. `Ring::from_u64`, `pow2`, `mul_pow_2`, `two_inv` and `half` are where Jolt's spine assumes odd characteristic. Moving them to a prime-only trait is the right long-term shape, but it touches every generic caller in the workspace; it is deferred until the binary prover shows which callers actually need which semantics.
- **Making `from_u64` the word embedding.** Rejected: `Ring::from_u64` is used as the ring map by existing generic code, and a word embedding there would make `from_u64(2)` nonzero and silently break round-sum reconstruction.
- **One 128-bit field for both builds, as a quadratic extension of `F64`.** Open, not rejected. It would let the hash build run its sum-checks in 128 bits and pay for 192 bits only inside the PCS. It is a third representation of $\mathbb F_{2^{128}}$ (neither Akita's polynomial basis nor a subfield of `F192`), so it needs its own type and a map into `F192` that does not exist as a field embedding; the decision belongs to the sum-check milestone.

## Documentation

None. The crate has no user-facing surface until the binary prover exists; the book gains a page with the end-to-end milestone.

## Execution

`src/binary/{mod,f64,f128,f192}.rs` for the backend and `src/binary_field.rs` for the `BinaryField` contract. `F64` multiply: 64-step shift-and-XOR with reduction by `0x1B`. `F128` multiply: the same over `u128` with reduction by `0x87`. `F192`: schoolbook over `F64` with $y^3=y+1$, $y^4=y^2+y$. Inversion by exponentiation to $2^n-2$ (or the tower formula for `F192`). Stamp operator impls with `impl_ring_ops!` and `impl_group_ops!`, serde with `impl_serde_bytes!`. Test vectors are generated outside the repository from the two reference implementations and checked in as constants.

## References

- leanVM, `crates/primitives/src/field/mod.rs` at `48a90420` (MIT OR Apache-2.0): definitions of `F64` and `F192`.
- Akita, `crates/akita-algebra/src/binary/host.rs`: `BinaryField128`.
- `specs/consolidate-field-traits.md`, `specs/jolt-field-rebuild.md`: the trait spine this crate implements.
- `specs/verifier-closure-lints.md`: the lint set the backend must satisfy.
