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

## Design

### Architecture

#### 1. The committed object and the claim

The table has `256·T` bits. Row `j` is a `BitsRow = [u64; 4]`, and bit `y` of row `j`, at flat index `y + 256·j`, is bit `y mod 64` of word `⌊y/64⌋`. The packed table is `p[z] = F64::from_raw(row_j[h])` at `z = h + 4·j`. It has `2^μ` symbols, `μ = t + 2`, and it is the row buffer read as `u64` words in memory order, so the honest prover never builds it. The six low column variables are inside a symbol; the two high column variables and the `t` cycle variables are the variables of `p`, in that order.

The front end calls the opening with the column point `rho ∈ H^8`, the cycle point `r_6 ∈ H^t` and the 256 column values `C`, after it has absorbed `C` and drawn `rho` (§11 of the protocol spec). The scheme derives

```text
r    = (rho[6], rho[7], r_6[0], …, r_6[t−1])                    ∈ H^μ
s_i  = Σ_{h<4} eq_H((rho[6], rho[7]), h) · C[i + 64·h]            i < 64, in H
```

and proves `s_i = Σ_z eq_H(r, z)·bit_i(p[z])` for every `i`. For any `C`, `value() = Σ_{i<64} eq_H(rho[0..6), i)·s_i`, so the 64 equalities give `value() = Bits~(rho, r_6)`. The scheme does not need the bound `8/2^128` of the front end for `rho`: it proves the 64 partial evaluations and not one random combination of them.

#### 2. The code and the commit phase

**Domain and basis.** The evaluation domain of dimension `d` is `S_d = { F64::from_raw(x) : x < 2^d }`, the span of `β_0, …, β_{d−1}`. Its subspace polynomials are `s_0(X) = X` and `s_{l+1}(X) = s_l(X)·(s_l(X) + s_l(β_l))`; `s_l` vanishes on `S_l` and `s_l(β_l) ≠ 0` because `β_l ∉ S_l`. With `Ŵ_l(X) = s_l(X)/s_l(β_l)`, the basis polynomial of index `w < 2^c` is `X_w(X) = Π_{l<c} Ŵ_l(X)^{w_l}`, of degree `w`. The code of dimension `2^c` on `S_d` is

```text
Enc_{c,d}(f)[x] = Σ_{w<2^c} f[w] · X_w(F64::from_raw(x)),        x < 2^d,
```

for `f` with values in `K` or in `E`. It is the Reed-Solomon code of the polynomials of degree below `2^c` on `S_d`, of rate `2^(c−d)`. For a fixed position `x` the map `f ↦ Enc(f)[x]` is the inner product with the vector `W_x[w] = X_w(x)`, whose multilinear extension is a product of `c` factors:

```text
W~_x(q) = Π_{l<c} (1 + q_l + q_l·Ŵ_l(F64::from_raw(x))).
```

**Levels.** Level `i` commits a function `f_i` of `m_i = k_i + c_i` variables: the low `k_i` variables index `2^(k_i)` lanes and the high `c_i` index the message of a lane. Its oracle is the matrix

```text
O_i[x][u] = Enc_{c_i, d_i}(f_i(u, ·))[x],        f_i(u, ·)[w] = f_i[u + 2^(k_i)·w],
```

with `2^(d_i)` positions `x` and `2^(k_i)` lanes `u`. Leaf `x` of its Merkle tree is the concatenation of `O_i[x][u]` for ascending `u`, each symbol in its canonical little-endian bytes: 8 bytes at level 0, where `f_0 = p`, and 24 bytes at every later level. A leaf digest is BLAKE2s-256 of the leaf bytes, a node is BLAKE2s-256 of its two children's 64 bytes, and the tree has depth `d_i`. The lane index is the low part of the symbol index. At level 0 the message matrix `p[u + 2^(k_0)·w]`, read position by position, is therefore the row buffer itself, and the encoder reads it without a transposition.

**Commit phase.** These are the messages of `commit` and `verify_commit`, step 6 of the preamble.

```text
1  P → V   root_0, the root of the tree of O_0                    absorbed
2  V       z_0 ∈ E^μ                                              μ challenges
3  P → V   y_0 = Σ_z eq_E(z_0, z)·p[z]  ∈ E                       absorbed
```

The commitment is `root_0` and `y_0`, 56 bytes. `verify_commit` checks nothing: it absorbs, draws, and keeps `(t, root_0, z_0, y_0)` as its state. `y_0` is a claim, and it is checked as the second claim of level 0 of the opening (§4). What it buys is invariant 2. The word in the leaves is within the decoding radius of at most `L_0` codewords (§8). Two different messages agree at a uniform point of `E^μ`, drawn after the root, with probability at most `μ/|E|`, so after step 3 at most one member of the list is consistent with `(z_0, y_0)`, except with probability `C(L_0, 2)·μ/|E|`. A commitment that is a root alone binds the list, and every error term of the front end would then be paid once per member of the list; at the reference size that is a loss of `log2 187 = 7.55` bits on terms that have no margin (§8).

#### 3. The bridge

The claims `s_i` are in `H`, the symbols in `K`, and the recursion runs in `E ⊃ K`. The bridge replaces the 64 claims by one inner-product claim over `E`, using only that `K ⊂ E` and that elements of `H` are vectors of 128 bits.

*Transposition.* Put `t_b = Σ_{i<64} β_i·bit_b(s_i) ∈ K` for `b < 128`: the 64 by 128 matrix of the bits of the `s_i`, read by columns. If the `s_i` are the partial evaluations of `p`, then, because `bit_b` is `F_2`-linear and `bit_i(p[z]) ∈ F_2`,

```text
t_b = Σ_i β_i · bit_b( Σ_z eq_H(r, z)·bit_i(p[z]) ) = Σ_z bit_b(eq_H(r, z)) · p[z].
```

*Batching.* The verifier draws `α ∈ E` and both sides are combined with the powers of `α`. Let `Φ_α : H → E` be the `F_2`-linear map `Φ_α(e) = Σ_{b<128} bit_b(e)·α^b`. Then

```text
τ = Σ_{b<128} t_b·α^b,        w[z] = Φ_α(eq_H(r, z)),        τ = Σ_z p[z]·w[z].        (bridge)
```

The verifier computes `τ` from `C`, `rho[6]`, `rho[7]` and `α`. The prover sends nothing for the bridge. The identity `(bridge)` is the first claim of level 0.

*Soundness, in outline.* Fix the function `p` bound by the commitment, with values in `E` in general (§8 explains why the argument is run for `E`-valued `p`), and write `p = p^(0) + p^(1)·y + p^(2)·y^2` with `p^(k)` valued in `K`. For claimed values `s_i` let `c_b = t_b + Σ_z bit_b(eq_H(r, z))·p[z] ∈ E`. The two sides of `(bridge)` differ by `Σ_b c_b·α^b`, a polynomial of degree at most 127 in `α`, and `p`, `C`, `rho`, `r_6` are all fixed before `α` is drawn. If some `c_b ≠ 0` the identity holds with probability at most `127/|E|`. If every `c_b = 0`, the component of `c_b` in `K` gives `t_b = Σ_z bit_b(eq_H(r,z))·p^(0)[z]` for all `b`, and since the `β_i` are independent over `F_2` this is `s_i = Σ_z eq_H(r,z)·bit_i(p^(0)[z])` for all `i`: the claims are the partial evaluations of the table of bits of `p^(0)`, which is fixed at commit time. The bound uses no relation between `H` and `E`.

*The weight's extension.* The final check of the recursion needs `w~(q) = Σ_z eq_E(q, z)·w[z]` at a point `q ∈ E^μ`. Work in the ring `E ⊗_{F_2} H`, whose elements are written `Σ_b V[b] ⊗ x^b` with `V ∈ E^128`. Since `(1+q_l)⊗(1+r_l) + q_l⊗r_l = (1+q_l)⊗1 + 1⊗r_l`,

```text
Σ_z eq_E(q, z) ⊗ eq_H(r, z) = Π_{l<μ} ( (1 + q_l)⊗1 + 1⊗r_l ),
```

and `w~(q)` is the image of this product under `V ↦ Σ_b V[b]·α^b`. The verifier evaluates it by a recurrence on `V`, starting from `V = (1, 0, …, 0)`:

```text
for l in 0..μ:    V ← (1 + q_l)·V + M_{r_l}·V,        (M_r·V)[b] = Σ_{b'} bit_b(r·x^(b')) · V[b'];
w~(q) = Σ_b V[b]·α^b.
```

`M_r` is the 128 by 128 matrix over `F_2` of multiplication by `r` in `H`; its column `b'` is `r·x^(b')`, obtained from the previous column by `mul_x`. One step is 128 multiplications in `E` and at most `128·128` additions in `E`, half of that for a uniform `r`. At `μ = 24` the recurrence is 3,072 multiplications and at most 393,216 additions, and the final combination 127 multiplications by Horner's rule.

The prover evaluates `Φ_α` with 16 tables of 256 entries of `E`, one per byte of the argument: `16·256·24 = 98,304` bytes, built from the powers of `α` with 16·255 additions each of one entry to another.

#### 4. The opening

Let `R` be the number of levels, `o_i = k_0 + … + k_{i−1}` the number of variables bound before level `i`, and `res = c_{R−1}` the number of variables of the final message; `m_i = μ − o_i` and `m_{i+1} = c_i`. The opening is one sumcheck of `μ` rounds for a claim `σ = Σ_z f(z)·ω(z)` whose weight `ω` grows by new terms at the start of each level. The round polynomial has degree 2.

```text
0   V       s_i, t_b from C and rho; α ∈ E; τ                       1 challenge
for i = 0 .. R−1:
a   V       λ_i ∈ E                                                 1 challenge
            σ ← σ + Σ_{j≥1} λ_i^j·v_j over the new claims (v_j, ω_j) of level i, in order
            (at i = 0:  σ ← τ + λ_0·y_0)
b   for each of k_i rounds:
    P → V   u_0, u_2 ∈ E                                            absorbed
    V       a ∈ E;  σ ← u_0 + (σ + u_2)·a + u_2·a^2                 1 challenge
            (the k_i challenges are a^(i) ∈ E^(k_i))
c   if i < R−1:
    P → V   root_{i+1}, the root of the tree of O_{i+1}             absorbed
    V       z_{i+1} ∈ E^(c_i)                                       c_i challenges
    P → V   y_{i+1} = f~_{i+1}(z_{i+1})                             absorbed
    else:
    P → V   f_R, 2^res elements of E                                absorbed
d   V       Q_i positions x of level i                              4·Q_i bytes
    P → V   the leaves O_i[x][·] at the distinct positions, and a Merkle multiproof
e   V       checks the multiproof against root_i;
            c_x = Σ_u eq_E(a^(i), u)·O_i[x][u] for each distinct x;
            if i < R−1: the new claims of level i+1 are (y_{i+1}, eq_E(z_{i+1}, ·)),
                        then (c_x, W_x) for the distinct x in ascending order;
            else:       checks c_x = Σ_w f_R[w]·W_x[w] for each distinct x
closing:
    for each of res rounds:  P → V  u_0, u_2;  V  a ∈ E;  σ as in b
final:
    V       checks σ = f~_R(q[μ−res..μ)) · Ω(q),   q ∈ E^μ the μ round challenges in order
```

`f_{i+1}[w] = Σ_u eq_E(a^(i), u)·f_i[u + 2^(k_i)·w]` is the fold of `f_i` over its lane variables, an honest `f_{i+1}` has `c_i` variables, and the encoding is linear, so `c_x = Enc(f_{i+1})[x] = Σ_w f_{i+1}[w]·W_x[w]`: a query of level `i` is an inner-product claim on the next message, and it joins the sumcheck with the others. In the round message `u_0` and `u_2` are the constant and the quadratic coefficient; the linear one is `σ + u_2`, because the two evaluations at 0 and 1 sum to `σ`. A position is drawn with replacement; repeated positions are opened once and give one claim. When `Q_i` of §5 is "all", no position is drawn and every position is opened, in ascending order.

The total weight at the end is

```text
Ω(q) = w~(q) + λ_0·eq_E(z_0, q)
     + Σ_{i=1}^{R−1} ( λ_i·eq_E(z_i, q[o_i..μ)) + Σ_{j≥2} λ_i^j·W~_{x_j}(q[o_i..μ)) ),
```

where `x_2 < x_3 < …` are the distinct positions queried at level `i−1` and `W~_x` is taken with `c = c_{i−1}` on the domain of dimension `d_{i−1}`. A weight that enters at level `i` is a function of the variables still free at that level, so it is evaluated at the suffix of `q` and carries no factor for the earlier coordinates. The verifier's work is the recurrence of §3, one equality polynomial per level, `c_{i−1}` multiplications of an element of `E` by an element of `K` per query claim after the `Ŵ_l(x)` have been computed in `K`, the lane combinations `c_x`, and the hashes of §9.

The opened leaves are not absorbed. They are fixed by a root that is absorbed before the positions are drawn, up to a collision of the hash, which the ledger prices separately.

#### 5. Parameters

The schedule is a function of `μ = t + 2`, for every `t` that the front end admits, `1 ≤ t ≤ 32`:

```text
k_0 = min(6, μ − 2),   c_0 = μ − k_0,   d_0 = c_0 + 1;
while c_i ≥ 6:   k_{i+1} = 4,   c_{i+1} = c_i − 4,   d_{i+1} = d_i − 1;
R = number of levels,   res = c_{R−1} ∈ {2, 3, 4, 5}.
```

Level 0 has rate 1/2 and each later level has a rate 8 times lower, since the message shrinks by 16 and the domain by 2. At `t = 22`: `μ = 24`, `R = 5`, `res = 2`.

| Level | `k_i` | `c_i` | `d_i` | Rate | Symbol | Leaf bytes | `Q_i` |
|---|---:|---:|---:|---|---|---:|---:|
| 0 | 6 | 18 | 19 | 1/2 | `K` | 512 | 260 |
| 1 | 4 | 14 | 18 | 1/16 | `E` | 384 | 65 |
| 2 | 4 | 10 | 17 | 1/128 | `E` | 384 | 37 |
| 3 | 4 | 6 | 16 | 1/1,024 | `E` | 384 | 26 |
| 4 | 4 | 2 | 15 | 1/8,192 | `E` | 384 | 20 |

There is no grinding at any level. The query counts are constants of the verifier, one row per `t`:

| `t` | `Q_0, Q_1, …` | | `t` | `Q_0, Q_1, …` |
|---:|---|---|---:|---|
| 1 to 9 | all | | 21 | 259, 65, 37, 26 |
| 10 | all, 59 | | 22 | 260, 65, 37, 26, 20 |
| 11 | 254, 62 | | 23 | 260, 65, 37, 26, 20 |
| 12 | 256, 63 | | 24 | 261, 65, 37, 26, 20 |
| 13 | 257, 64 | | 25 | 262, 65, 37, 26, 20 |
| 14 | 257, 64, 35 | | 26 | 262, 65, 37, 26, 20, 16 |
| 15 | 258, 64, 36 | | 27 | 263, 65, 37, 26, 20, 17 |
| 16 | 258, 65, 37 | | 28 | 264, 65, 37, 26, 20, 17 |
| 17 | 258, 65, 37 | | 29 | 266, 65, 37, 26, 20, 17 |
| 18 | 258, 65, 37, 25 | | 30 | 267, 65, 37, 26, 21, 17, 14 |
| 19 | 259, 65, 37, 26 | | 31 | 268, 65, 38, 26, 21, 17, 14 |
| 20 | 259, 65, 37, 26 | | 32 | 271, 66, 38, 27, 21, 17, 15 |

The table is derived by the rule of §8: for each level, among the admissible slacks `η = √ϱ/m`, `m ≥ 3` an integer, that keep every algebraic term of the level at most `2^-128`, the one with the fewest queries `Q = ⌈128 / log2(1/(√ϱ + η))⌉`, and "all" where that count reaches the number of positions. The derivation ran outside the repository, in double precision; item 1 of Execution re-derives every entry with exact rational bounds and freezes the integers. The verifier contains the integers and not the rule (invariant 8).

#### 6. Transcript, challenges and wire format

**Absorption.** In the notation of §13 of the protocol spec, `L(l)` is a label and `B(s)` one `append_bytes` call. The scheme's six labels are `whir_commit`, `whir_ood`, `whir_open`, `whir_round`, `whir_root` and `whir_final`.

```text
commit    L("whir_commit") B(root_0);   then z_0;   L("whir_ood") B(y_0)
open      L("whir_open");               then α
level i   λ_i
          per round:  L("whir_round") B(u_0 ‖ u_2);   then a
          i < R−1:    L("whir_root") B(root_{i+1});   then z_{i+1};   L("whir_ood") B(y_{i+1})
          i = R−1:    L("whir_final") B(f_R[0] ‖ … ‖ f_R[2^res − 1])
          then the positions of level i
closing   per round:  L("whir_round") B(u_0 ‖ u_2);   then a
```

An element of `E` is absorbed as its 24 canonical bytes and a root as its 32 bytes. Nothing of the opening request is absorbed again: the front end has absorbed `C` and has drawn `rho` and `r_6` from the same transcript, so its state binds them when `whir_open` is absorbed. The scheme absorbs nothing after the last closing round.

**Challenges.** An element of `E` is one call of `squeeze_bytes` for 24 bytes: two draws of 16 bytes, their little-endian encodings concatenated, the first 24 bytes kept and read as three little-endian `u64` coefficients. Vectors are drawn coordinate by coordinate in index order. The positions of level `i` are one call of `squeeze_bytes` for `4·Q_i` bytes; position `j` is the little-endian `u32` at bytes `4j..4j+4`, reduced to its low `d_i` bits. Since `2^(d_i)` divides `2^32` the positions are uniform and independent. `d_i ≤ 29` for every admitted `t`.

**Wire.** The commitment is `root_0 ‖ y_0`, 56 bytes. The opening proof is the concatenation, in protocol order, of:

```text
for each level i:
    k_i rounds, each u_0 ‖ u_2                                    48 bytes per round
    i < R−1:  root_{i+1} ‖ y_{i+1}                                56 bytes
    i = R−1:  f_R                                                 24·2^res bytes
    n_i   as u32 little-endian, the number of distinct positions
    n_i leaves, in ascending position order                       2^(k_i)·8 or 2^(k_i)·24 bytes each
    g_i   as u32 little-endian, the number of sibling digests
    g_i digests                                                   32 bytes each
res closing rounds, each u_0 ‖ u_2                                48 bytes per round
```

The multiproof is the standard one. With the distinct positions as the known nodes of the leaf layer, each layer is processed from the leaves up and, within a layer, from left to right: a known node whose sibling is not known takes the next digest of the list as that sibling. The digests are therefore in the order in which the verifier consumes them, and `g_i` is a function of the positions. `read` checks `1 ≤ n_i ≤ min(Q_i, 2^(d_i))`, `g_i ≤ n_i·d_i` and that the length of the string is exactly the sum of the parts, before it allocates; `verify_opening` checks that `n_i` and `g_i` are the counts that the drawn positions determine and rejects otherwise. No position, no claim value `c_x`, no bridge value and no nonce is on the wire.

At `t = 22` the fixed part is `24·48 + 4·56 + 4·24 + 5·8 = 1,512` bytes. The query part is at most `Σ Q_i·(leaf + 32·d_i) = 291,200 + 62,400 + 34,336 + 23,296 + 17,280 = 428,512` bytes, when no two paths share a node, and its expectation over the positions is 324,740 bytes (counted: a node at height `h` of a tree of depth `d` is on a queried path with probability `π_h = 1 − (1 − 2^(h−d))^Q`, and the expected number of digests is `Σ_{h<d} 2^(d−h)·π_h·(1 − π_h)`, which gives 2,623, 721, 403, 271 and 196 digests for the five levels). The opening proof is thus 326,252 bytes in expectation and at most 430,024, and the whole proof of the experiment is the front end's 15,024 bytes, its 26-byte envelope, the 56-byte commitment and the opening.
