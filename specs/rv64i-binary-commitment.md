# Spec: The Commitment Scheme for the Bit Table of RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The front end of the binary-field RV64I experiment (`specs/rv64i-binary-protocol.md`) commits to one table of bits, `Bits`, with 256 columns and `T = 2^t` rows, and at the end asks for one multilinear evaluation of it at a point with coordinates in `F128`. Until now the only implementation of that contract is `TransparentBits`, a test stand-in that sends the table. This spec fixes the production scheme behind the same two traits, `BitsCommitmentScheme` and `BitsCommitmentProver`, with no change to either.

The scheme is hash-based. The bits are packed 64 to a symbol of `F64`, the `2^(t+2)` symbols are encoded with an interleaved Reed-Solomon code on an additive domain of `F64` and committed with a Merkle tree, and the commitment carries one out-of-domain evaluation so that after the commit phase one table is bound and not a list of tables. The opening is a WHIR recursion over the cubic extension `F192` of `F64`, analysed in the list-decoding regime up to the Johnson bound with proven proximity gaps and no conjecture. The evaluation claim lives in `F128`, which is not a subfield of `F192`; a tensor reduction (ring switching) of shape 64 by 128 turns the 64 partial evaluations that the front end already publishes into one inner-product claim over `F192`. No element of `F128` is ever multiplied inside `F192`, so the missing embedding costs nothing.

At the reference size `t = 22` the table has `2^30` bits and `2^24` symbols. The parameter set is rate 1/2, fold schedule 6, 4, 4, 4, 4, query counts 260, 65, 37, 26, 20 and no grinding. Every error term of the scheme is at most `2^-128`; the weakest is the query term of the first level at `2^-128.00`, followed by its proximity-gap term at `2^-129.33`. The expected opening proof is about 326 KiB-scale (326,252 bytes expected, 430,024 at most) and the commitment is 56 bytes. The existing public implementation of the same recursion, at this geometry and for a claim in `F192`, measures 311 ms to commit and 730 ms to open on one thread and 67.8 ms and 142.6 ms on twelve, on a loaded host. The model of this spec for the scheme with the `F128` bridge and the commit-time sample is 198 ns per cycle on one thread, which gives a threshold of 247 ns per cycle, 12% of the prover's 2,100.

The spec compares two opening fields side by side, `F192` with the bridge and a field that contains `F128`, and recommends `F192`. That recommendation is for the owner to confirm.

## Intent

### Goal

Fix, completely enough to implement and to audit, the commitment scheme of the bit table: the committed object and the claim, every message of the commit phase and of the opening, every challenge and the bytes it is drawn from, the wire format, the parameter set and its soundness ledger, the prover's operation counts and memory, and the items of work.

Notation used throughout. `K = F64`, `H = F128` and `E = F192` are the types of `jolt_field::binary`; `K ⊂ E` and `H` is unrelated to `E`. `t = log_T`, and `μ = t + 2` is the number of variables of the packed table. Points are low-variable-first and sumchecks bind the lowest variable first, as in the front end. `eq_F(r, z)` is the equality polynomial over the field `F`, with bit `l` of the index `z` paired with `r[l]`. `β_i = x^i`, `i < 64`, is the basis of `K` over `F_2` in which `F64::from_raw` reads a word. `bit_b(e)` is bit `b` of the raw representation of an element of `K` or `H`. "Level" means one committed oracle of the recursion; level 0 is the commitment itself.

### Invariants

1. **The committed object.** The commitment binds one function `p : {0,1}^μ → K`. For an honest prover `p[h + 4·j] = F64::from_raw(row_j[h])` for `h < 4`, `j < T`, where `row_j : [u64; 4]` is row `j` of the table, so that `bit_i(p[h + 4·j]) = Bits[i + 64·h, j]`. Index and bit order are those of §3 of the protocol spec and are frozen here.
2. **Unique binding at commit time.** After `verify_commit` returns, and except with the probability of the ledger rows "commit sample" and "Merkle", there is at most one table of bits for which any later opening can be accepted. The mechanism is the out-of-domain evaluation carried in the commitment and checked in the opening (§2, §4 of Architecture).
3. **The claim proved.** `verify_opening` accepts only if, for the bound table, `Σ_{z} eq_H(r, z)·bit_i(p[z]) = s_i` for all `i < 64`, where `r = (rho[6], rho[7], r_6)` and `s_i = Σ_{h<4} eq_H((rho[6], rho[7]), h)·C[i + 64·h]`, except with the probability of the ledger. This implies `Bits~(rho, r_6) = BitsOpening::value()`, which is the contract, and is strictly stronger. `rho[0..6)` is not used by the scheme.
4. **Fields.** Code symbols of level 0 are in `K`. Every challenge of the scheme, every later symbol and every sumcheck message is in `E`. No value of `H` is converted to `E` other than through the map `Φ_α` of §3, which reads its bits.
5. **Challenges.** Every challenge in `E` is 24 bytes from `squeeze_bytes`, that is two consecutive 16-byte draws of which the first 24 bytes are kept, read by `F192::from_bytes_le_checked`. It is uniform in `E`. No challenge of the scheme is sampled from a subset of `E`, and none is derived from another challenge except the powers of a batching challenge.
6. **Transcript.** Every message of the scheme is absorbed before the challenge that depends on it is drawn, in the order of §6. The scheme uses six labels, none of which is one of the nine of the front end.
7. **Canonical wire.** `BitsWire::read` accepts exactly the strings that `write` produces for the geometry. Lengths are checked against bounds computed from the geometry before any allocation. A commitment or an opening has one encoding.
8. **Parameters are constants.** The level schedule and the query counts are functions of `t` alone, stored as a table of integers (§5). No floating-point computation runs in the verifier or in the prover.
9. **The prover copies no table.** `commit` reads the rows through the `Arc` it is given and writes the level-0 codeword, which is a new buffer of twice the table's size. It makes no packed copy of the table. `open` reads the rows through the same `Arc`.
10. **Verifier safety.** The verifier's code is in a crate with `#![forbid(unsafe_code)]`, does not panic on any input, and allocates in proportion to validated lengths only.

### Non-Goals

- Zero knowledge. The scheme hides nothing.
- A change to the front end, to its proof, or to the two traits. The scheme is one more implementation of them.
- A general polynomial commitment scheme for `jolt-openings`. The generalisation of those traits is a separate later item (Execution, item 9).
- Recursion-friendliness of the verifier, batching of several tables or several points, and streaming commitment.
- A proof of the proximity-gap theorem or of the round-by-round theorem. The spec states which published results it rests on and with what constants.
