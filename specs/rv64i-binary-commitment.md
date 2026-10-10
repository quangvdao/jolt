# Spec: The Commitment Scheme for the Bit Table of RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The front end of the binary-field RV64I experiment (`specs/rv64i-binary-protocol.md`) commits to one table of bits, `Bits`, with 256 columns and `T = 2^t` rows, and at the end asks for one multilinear evaluation of it at a point with coordinates in `F128`. Until now the only implementation of that contract is `TransparentBits`, a test stand-in that sends the table. This spec fixes the production scheme behind the same two traits, `BitsCommitmentScheme` and `BitsCommitmentProver`, with no change to either.

The scheme is hash-based. The bits are packed 64 to a symbol of `F64`, the `2^(t+2)` symbols are encoded with an interleaved Reed-Solomon code on an additive domain of `F64` and committed with a Merkle tree, and the commitment carries one out-of-domain evaluation of every lane of the code so that after the commit phase one table is bound and not a list of tables. The opening is a WHIR recursion over the cubic extension `F192` of `F64`, analysed in the list-decoding regime up to the Johnson bound with proven proximity gaps and no conjecture. The evaluation claim lives in `F128`, which is not a subfield of `F192`; a tensor reduction (ring switching) of shape 64 by 128 turns the 64 partial evaluations that the front end already publishes into one inner-product claim over `F192`. No element of `F128` is ever multiplied inside `F192`, so the missing embedding costs nothing.

At the reference size `t = 22` the table has `2^30` bits and `2^24` symbols. The parameter set is rate 1/2, fold schedule 6, 4, 4, 4, 4, query counts 260, 65, 37, 26, 20 and no grinding. Every error term of the scheme is at most `2^-128`; the weakest is the query term of the first level at `2^-128.00`, followed by its proximity-gap term at `2^-129.33`. The opening proof is 326,252 bytes in expectation and 430,024 at most, and the commitment is 1,568 bytes. The existing public implementation of the same recursion, at this geometry and for a claim in `F192`, measures 311 ms to commit and 730 ms to open on one thread and 67.8 ms and 142.6 ms on twelve, on a loaded host. The model of this spec for the scheme with the `F128` bridge and the commit-time sample is 198 ns per cycle on one thread, which gives a threshold of 247 ns per cycle, 12% of the prover's 2,100.

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
1  P → V   root_0, the root of the tree of O_0                              absorbed
2  V       z_0 ∈ E^(c_0)                                                    c_0 challenges
3  P → V   y_0[u] = Σ_w eq_E(z_0, w)·p[u + 2^(k_0)·w]  ∈ E,  u < 2^(k_0)    absorbed
```

The commitment is `root_0` and the `2^(k_0)` values `y_0`: `32 + 24·64 = 1,568` bytes at `t = 22`. The sample is one evaluation of every lane's message at a common point, and not one evaluation of `p` at a point of `E^μ`, for a reason of cost: a claim about `p` would join the sumcheck at level 0 with a weight of `2^μ` elements of `E`, while the lane values combine, after the lane variables are folded, into one claim about `f_1`, whose weight has `2^(c_0)` elements (§4, §9). `verify_commit` checks nothing: it absorbs, draws, and keeps `(t, root_0, z_0, y_0)` as its state. What the sample buys is invariant 2. The word in the leaves is within the decoding radius of at most `L_0` codewords of the interleaved code (§8). Two different members differ in some lane, and two different lane messages agree at a uniform point of `E^(c_0)`, drawn after the root, with probability at most `c_0/|E|`, so after step 3 at most one member of the list is consistent with `(z_0, y_0)`, except with probability `C(L_0, 2)·c_0/|E|`. A commitment that is a root alone binds the list, and every error term of the front end would then be paid once per member of the list; at the reference size that is a loss of `log2 187 = 7.55` bits on terms that have no margin (§8).

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
a   V       λ_i ∈ E, for i ≥ 1                                      1 challenge
            σ ← σ + Σ_{j≥1} λ_i^j·v_j over the new claims (v_j, ω_j) of level i, in order
            (at i = 0 there is no λ_0 and no new claim:  σ ← τ)
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
                        then (c_x, W_x) for the distinct x in ascending order,
                        then, for i = 0, the commit sample (Σ_u eq_E(a^(0), u)·y_0[u], eq_E(z_0, ·));
            else:       checks c_x = Σ_w f_R[w]·W_x[w] for each distinct x,
                        and, for i = 0, checks f~_R(z_0) = Σ_u eq_E(a^(0), u)·y_0[u]
closing:
    for each of res rounds:  P → V  u_0, u_2;  V  a ∈ E;  σ as in b
final:
    V       checks σ = f~_R(q[μ−res..μ)) · Ω(q),   q ∈ E^μ the μ round challenges in order
```

`f_{i+1}[w] = Σ_u eq_E(a^(i), u)·f_i[u + 2^(k_i)·w]` is the fold of `f_i` over its lane variables, an honest `f_{i+1}` has `c_i` variables, and the encoding is linear, so `c_x = Enc(f_{i+1})[x] = Σ_w f_{i+1}[w]·W_x[w]`: a query of level `i` is an inner-product claim on the next message, and it joins the sumcheck with the others. In the round message `u_0` and `u_2` are the constant and the quadratic coefficient; the linear one is `σ + u_2`, because the two evaluations at 0 and 1 sum to `σ`. A position is drawn with replacement; repeated positions are opened once and give one claim. When `Q_i` of §5 is "all", no position is drawn and every position is opened, in ascending order.

The total weight at the end is

```text
Ω(q) = w~(q)
     + Σ_{i=1}^{R−1} ( λ_i·eq_E(z_i, q[o_i..μ)) + Σ_{j=2}^{n_{i−1}+1} λ_i^j·W~_{x_j}(q[o_i..μ)) )
     + λ_1^(n_0 + 2)·eq_E(z_0, q[k_0..μ)),                          the last term only if R ≥ 2,
```

where `x_2 < x_3 < …` are the `n_{i−1}` distinct positions queried at level `i−1` and `W~_x` is taken with `c = c_{i−1}` on the domain of dimension `d_{i−1}`. The commit sample is a claim about `f_1`: the lane values `y_0[u]` are evaluations of the lanes of `f_0` at `z_0`, and `f_1` is their combination with `eq_E(a^(0), ·)`, so `f~_1(z_0) = Σ_u eq_E(a^(0), u)·y_0[u]`. It is the last claim of level 1, or, when level 0 is the only level, an equation on the final message. A weight that enters at level `i` is a function of the variables still free at that level, so it is evaluated at the suffix of `q` and carries no factor for the earlier coordinates. The verifier's work is the recurrence of §3, one equality polynomial per level, `c_{i−1}` multiplications of an element of `E` by an element of `K` per query claim after the `Ŵ_l(x)` have been computed in `K`, the lane combinations `c_x`, and the hashes of §9.

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

An element of `E` is absorbed as its 24 canonical bytes and a root as its 32 bytes. Nothing of the opening request is absorbed again: the front end has absorbed `C` and has drawn `rho` and `r_6` from the same transcript, so its state binds them when `whir_open` is absorbed. `y_0` is absorbed as the concatenation of its `2^(k_0)` elements in lane order, in one call. The scheme absorbs nothing after the last closing round.

**Challenges.** An element of `E` is one call of `squeeze_bytes` for 24 bytes: two draws of 16 bytes, their little-endian encodings concatenated, the first 24 bytes kept and read as three little-endian `u64` coefficients. Vectors are drawn coordinate by coordinate in index order. The positions of level `i` are one call of `squeeze_bytes` for `4·Q_i` bytes; position `j` is the little-endian `u32` at bytes `4j..4j+4`, reduced to its low `d_i` bits. Since `2^(d_i)` divides `2^32` the positions are uniform and independent. `d_i ≤ 29` for every admitted `t`.

**Wire.** The commitment is `root_0 ‖ y_0[0] ‖ … ‖ y_0[2^(k_0) − 1]`, `32 + 24·2^(k_0)` bytes. The opening proof is the concatenation, in protocol order, of:

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

At `t = 22` the fixed part is `24·48 + 4·56 + 4·24 + 5·8 = 1,512` bytes. The query part is at most `Σ Q_i·(leaf + 32·d_i) = 291,200 + 62,400 + 34,336 + 23,296 + 17,280 = 428,512` bytes, when no two paths share a node, and its expectation over the positions is 324,740 bytes (counted: a node at height `h` of a tree of depth `d` is on a queried path with probability `π_h = 1 − (1 − 2^(h−d))^Q`, and the expected number of digests is `Σ_{h<d} 2^(d−h)·π_h·(1 − π_h)`, which gives 2,623, 721, 403, 271 and 196 digests for the five levels). The opening proof is thus 326,252 bytes in expectation and at most 430,024, and the whole proof of the experiment is the front end's 15,024 bytes, its 26-byte envelope, the 1,568-byte commitment and the opening.

#### 7. The opening field: two designs

`H = F_2[x]/(x^128 + x^7 + x^2 + x + 1)` and `E = K[y]/(y^3 + y + 1)` have degrees 128 and 192 over `F_2`, and 128 does not divide 192, so `H` is not a subfield of `E`. The claim is in `H`. Two designs answer this.

**Design (a): `E = F192` and the 64 by 128 bridge.** This is the design of §1 to §6. Symbols are in `K`, 64 bits each, `2^24` of them at `t = 22`. Arithmetic in `E` exists in `jolt_field::binary` with its accumulator. Counted in carry-less multiplications of 64-bit words on aarch64, from the field code: a product in `K` is 3 (one product, a reduction of two); an element of `E` times an element of `K` is 9 (three products, three reductions), or 3 into an accumulator; a product in `E` is 12 (six products, three reductions), or 6 into an accumulator; a product in `H` is 6.

**Design (b): an opening field that contains `H`.** The smallest field that contains `H` and has at least `2^192` elements is the quadratic extension `E' = H[v]/(v^2 + v + x^121)`, with `2^256` elements. The polynomial is irreducible because the absolute trace of `x^121` in `H` is 1, checked by direct computation of `Σ_{k<128} (x^121)^(2^k)` outside the repository; it is the lowest power of `x` with trace 1. `E'` does not exist in the repository. A product in `E'` is three products in `H`, 18 carry-less multiplications, or 12 into an accumulator with a reduction of 4; an element of `E'` times an element of `H` is 12, or 8 into an accumulator; an element is 32 bytes. For symbols and challenges to multiply cheaply the code alphabet must be a subfield of `E'` in its representation, and the repository's `K` is not: `H` has a subfield of `2^64` elements, but it is not the set of words below `2^64` of `F128`, so a symbol of `K` would enter `E'` through a 64 by 64 matrix over `F_2`, or the code would run over that subfield with the arithmetic of `H` at twice the cost of `K` per butterfly. Design (b) therefore packs 128 bits to a symbol of `H`: `2^23` symbols at `t = 22`, level 0 with 32 lanes (`k_0 = 5`) and the same `c_0 = 18`, a domain inside `H`, and the same later schedule.

Design (b) does not remove the bridge. The bridge exists because bits are packed into symbols, not because of the missing embedding: the claims about a packed table are claims about its 128 bit-slices, and reducing them to one claim on symbols needs the bits of `eq_H(r, z)` in either design. In (b) it has shape 128 by 128, the weight is `Φ'_α(eq_H(r', z))` with `α ∈ E'` and `r' = (rho[7], r_6)`, and the verifier's recurrence is the same with `V ∈ E'^128`. What containment adds is a second form of the verifier's evaluation through the Frobenius automorphism of `E'` over `F_2`, which this spec does not use.

Counts at `t = 22`, for the prover algorithm of §9 in both designs:

| Quantity | (a) `K`, `E = F192` | (b) `H`, `E' = F256` |
|---|---:|---:|
| Symbols at level 0, leaf bytes | `2^24`, 512 | `2^23`, 512 |
| Level-0 butterflies × multiplications each | 301,989,888 × 3 | 150,994,944 × 6 |
| Later-level butterflies × multiplications each | 43,515,904 × 9 | 43,515,904 × 12 |
| Commit sample, multiplications | `3·2^24` = 50,331,648 | `8·2^23` = 67,108,864 |
| Sumcheck round 1, multiplications | `39·2^23` = 327,155,712 | `56·2^22` = 234,881,024 |
| Sumcheck later rounds, multiplications | `36·(2^23 − 1)` = 301,989,852 | `56·(2^22 − 1)` = 234,880,968 |
| Induced weights, 7,405,568 butterflies | × 9 = 66,650,112 | × 12 = 88,866,816 |
| **Carry-less multiplications, total** | **2,043,740,124** | **2,053,898,184** |
| Lookups of `Φ` (entry bytes) | 268,435,456 (24) | 134,217,728 (32) |
| BLAKE2s compressions, all trees | 8,159,227 | 9,142,267 |
| Later-level leaf bytes; query bytes at most | 384; 428,512 | 512; 447,456 |
| Weight vector; folded message after round 1 | 384 MiB; 192 MiB | 256 MiB; 128 MiB |
| Later codewords | 180 MiB | 240 MiB |
| Verifier bridge recurrence | 3,072 products in `E` | 2,944 products in `E'` |
| Proximity-gap term, weakest level | `2^-129.33` | `2^-193.33` |
| Query counts | 260, 65, 37, 26, 20 | the same to within 4 at level 0 |
| Field code that exists; measured implementation | all; yes, for `K` and `E` | none of `E'`; none |

The rows follow from §9: level 0 encodes `2^(k_0)` lanes with `c_0·2^(d_0 − 1)` butterflies each, so 64 lanes in `K` and 32 lanes in `H` cost the same 905,969,664 multiplications; later levels have the same butterfly counts and pay the wider field; the sumcheck of (b) runs over half as many symbols at a wider field. The level-0 tree is identical; the later trees of (b) hash 512-byte leaves, 8 compressions instead of 6 for each of 491,520 leaves. Query counts are set by the rates: with the slack going to zero the first level needs `128/log2(√2) = 256` queries, so a larger field buys at most 4.

Soundness. In (a) every algebraic term is below `2^-128` (§8). In (b) each gains 64 bits, and none of them is the weakest term in (a): the query terms are, and they do not depend on the field. The front end's terms are in `H` in both designs.

**Recommendation, for the owner to confirm: design (a).** The deciding count is the total of carry-less multiplications, 2,043,740,124 against 2,053,898,184, a difference of 0.5%: an opening field that contains `F128` does not make the prover cheaper in field arithmetic. What it does buy is 134,217,728 fewer table lookups and 192 MiB less at the peak, against 983,040 more hash compressions, 18,944 more proof bytes, a field type with its kernels and accumulator that does not exist, and a recursion in which no kernel of the measured implementation applies. No ledger term that is binding improves. The recommendation reverses if the lookups of `Φ` are measured to dominate the opening after the bridge is implemented and a kernel for `E'` is available at the product counts above.

#### 8. Security ledger

**Statement.** The scheme is analysed as an interactive protocol with round-by-round knowledge soundness: after each verifier challenge, the probability that a state with no valid witness becomes one with a valid witness is at most the error of that round. The target is an error of at most `2^-128` for every round of the scheme. This is a statement about the maximum over rounds and not about their sum. After the Fiat-Shamir transformation in the random-oracle model, a prover that makes `q` queries to the transcript's hash succeeds with probability at most `q` times the largest round error, plus the Merkle term. The ledger is a list of bounds under stated assumptions; nothing in it is a proof of security of the implementation.

**Assumptions.**

1. The proximity-gap theorem for Reed-Solomon codes up to the Johnson bound, in its list form for mutual correlated agreement (Ben-Sasson, Carmon, Haböck, Kopparty and Saraf, 2025, Theorem 4.6), with the constant below. It is proven and uses no conjecture on list sizes beyond the Johnson bound. It is applied to the code over `E` on the domain `S_d ⊂ K`; the theorem places no condition on the domain.
2. The round-by-round analysis of the WHIR recursion with interleaved codes, with the error terms listed below. It is taken from the analysis that accompanies the public implementation named in §10; the modifications of this spec (the commit sample, the bridge, the direct check of the last level) are argued here.
3. BLAKE2s-256 and the transcript's hash are random oracles.

**Quantities, per level.** `n = 2^d`, `ϱ = (2^c − 1)/2^d` (the rate parameter of the theorem for dimension `2^c`), slack `η`, radius `γ = 1 − √ϱ − η`, list bound `L = 1/(2·η·√ϱ)` (Johnson; the same for the interleaved code, since interleaving keeps the distance), `m = max(⌈√ϱ/η⌉, 3)`, and

```text
a = ( 2·(m + 1/2)^5 + 3·(m + 1/2)·γ·ϱ ) / (3·ϱ^(3/2)) · n + (m + 1/2)/√ϱ,        ε = a/|E|.
```

**Terms.** `J_i` is the number of claims batched at level `i ≥ 1`, the residual included: `J_1 = n_0 + 3` and `J_i = n_{i−1} + 2` after it, with `n_{i−1} ≤ Q_{i−1}`. Level 0 has one claim and no batching.

| Round | Error | What fails otherwise |
|---|---|---|
| Commit sample `z_0` | `C(L_0, 2)·c_0/\|E\|` | two members of the level-0 list agree at `z_0` in every lane |
| Lane combination of the commit sample | `L_0·k_0/\|E\|` | a list member that disagrees with `y_0` in some lane agrees after the combination with `eq_E(a^(0), ·)` |
| Bridge `α` | `127/\|E\|` | §3 |
| Batching `λ_i` | `(J_i − 1)·L_i/\|E\|` | a false claim cancels in the combination, for some list member |
| Fold round `j ≤ k_i` of level `i` | `2·L_i/\|E\| + 2^(k_i − j)·ε_i` | a sumcheck round of degree 2, or correlated agreement of the partial fold |
| Sample `z_{i+1}` | `C(L_{i+1}, 2)·c_i/\|E\|` | as the commit sample, for level `i+1` |
| Positions of level `i` | `(1 − γ_i)^(Q_i)` | every queried column agrees with a word that is `γ_i`-far |
| Closing round | `2/\|E\|` | a sumcheck round of degree 2 |
| Merkle | `q^2/2^257` for `q` hash queries | a collision of BLAKE2s-256 |

At the reference size, with `|E| = 2^192`, as `−log2` of the error (computed, in double precision):

| Level | `ϱ` | `m` | `η` | `γ` | `L` | `Q` | Positions | Fold, `j = 1` | Sample | Batching | `2L/\|E\|` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `(2^18−1)/2^19` | 187 | 0.003781 | 0.28911 | 187.0 | 260 | 128.00 | 129.33 | 173.74 | none | 183.45 |
| 1 | `(2^14−1)/2^18` | 47 | 0.005319 | 0.74469 | 376.0 | 65 | 128.03 | 137.74 | 171.72 | 175.41 | 182.45 |
| 2 | `(2^10−1)/2^17` | 35 | 0.002524 | 0.90913 | 2,242.2 | 37 | 128.02 | 136.33 | 166.93 | 174.82 | 179.87 |
| 3 | `(2^6−1)/2^16` | 16 | 0.001938 | 0.96706 | 8,322.0 | 26 | 128.02 | 138.33 | 163.63 | 173.73 | 177.98 |
| 4 | `(2^2−1)/2^15` | 5 | 0.001914 | 0.98852 | 27,306.7 | 20 | 128.89 | 142.17 | 160.94 | 172.51 | 176.26 |

The bridge is at 185.01 bits, the lane combination of the commit sample at 181.87 and a closing round at 191. The "Fold" column is the correlated-agreement part at its worst round, `2^(k_i − 1)·ε_i`; the sample of row `i` is the one that binds level `i` (the commit sample for row 0). The weakest term is the positions of level 0, `(1 − 0.28911)^260 = 2^-128.0006`, and the weakest algebraic term is the fold of level 0 at `2^-129.33`. Each `η` is the largest slack of the form `√ϱ/m` for which the query count is minimal while the fold, sample and batching terms of the level stay below `2^-128`. Every term of the scheme meets the target with no grinding.

**Why the list is over `E` and the table is still a table of bits.** The fold challenges are in `E`, so the relevant code is the one over `E`, and the word in the leaves of level 0, which is `K`-valued because a leaf symbol is 8 bytes, can be close to codewords whose messages are not `K`-valued. The commit sample selects at most one message `f` with values in `E`. The table that the scheme binds is the table of bits of the component of `f` in `K`, which is a function of `f` and so fixed at commit time, and §3 shows that an accepted opening proves the claims about that table.

**The direct check of the last level.** The claims of the last level's queries are checked against the final message, which is sent in the clear, and not batched; this removes the batching round that they would need and adds no term.

**What the field of the front end bounds.** Every challenge of the front end is in `H`, and each of its rounds has an error of the form `δ/2^128` with `δ ≥ 1`: the bound `8/2^128` for `rho` is `2^-125`, and a sumcheck round of degree `δ` is `δ/2^128`. A larger opening field does not strengthen any of them. The commit sample keeps them from being multiplied by the list size, and that is all the scheme does for them. An experiment that wants 128 bits from the front end needs challenges from a field larger than `H` in the front end, which is outside this spec.

**Grinding.** There is none. Grinding `g` bits before the positions of a level are drawn does not change that round's error; it multiplies the work of each attempt at that round by `2^g`. Counted as error it would let the query counts drop to 225, 56, 32, 23, 17 at `g = 17` with the position terms at `2^-111.0`, saving 57,760 of the 428,512 bytes at most (41,784 of 332,112 bytes measured on the existing implementation), for an expected `5·2^17` hash evaluations of grinding. It buys nothing for the fold, sample, batching, bridge or front-end terms. It is not adopted, so that the target is met as an error bound.

**What remains below the target.** Not the scheme's rounds. Below it are: the front end's terms, as above; the Merkle term, which is a statement about work and reaches `2^-128` only for `q ≤ 2^64.5` hash queries, a collision costing about `2^128` evaluations; and the multiplication by `q` of the Fiat-Shamir transformation, under which a round error of `2^-128` is security against `2^128/q`-fold advantage and not an error of `2^-128` after `q` attempts.

#### 9. The prover

**`commit`.** (1) Allocate the level-0 codeword, `2^(d_0)` positions of `2^(k_0)` symbols, `2^25` symbols and 256 MiB at `t = 22`. For each position `w < 2^(c_0)` copy the `2^(k_0)` words of the rows into position `w` of each of the two cosets of the message (the rate is 1/2), and run the additive transform position-major, each butterfly acting on a whole row of lanes with one twiddle in `K`: `2^(k_0)·c_0·2^(d_0−1) = 64·18·2^18 = 301,989,888` butterflies, each one multiplication in `K` and two additions. (2) Hash `2^(d_0)` leaves and the tree: `2^19` leaves of 512 bytes are 8 compressions each and the `2^19 − 1` nodes one each, 4,718,591 compressions over 301,989,824 bytes. (3) Absorb the root, draw `z_0`, build the table `eq_E(z_0, ·)` of `2^(c_0)` elements of `E` (6 MiB, `2^18` products in `E`), and compute the `2^(k_0)` values `y_0` in one pass over the rows: position `w` adds `eq_E(z_0, w)·p[u + 2^(k_0)·w]` to accumulator `u`, one product of an element of `E` by an element of `K` into an accumulator per symbol, `3·2^24` carry-less multiplications.

**State between `commit` and `open`.** `ProverState` holds the `Arc` of the rows, the level-0 codeword, the level-0 tree, `z_0`, `y_0` and `t`. At `t = 22`: `2^25·8 = 268,435,456` bytes of codeword, `(2^20 − 1)·32 = 33,554,400` bytes of tree and `18·24 + 64·24 + 8 = 1,976` bytes of scalars, 301,991,832 bytes, plus the shared reference to the 128 MiB of rows, which the scheme does not own. `open` consumes the state.

**`open`, level 0.** The weight is built once and kept. With `e'[k] = eq_H(r[1..μ), k)` taken as the product of two entries of split tables of `2^11` and `2^(μ−12)` elements of `H`, the pair of symbols `2k`, `2k+1` has `eq_H` values `e_1 = r[0]·e'[k]` and `e_0 = e'[k] + e_1`: two products in `H` per pair. `w[2k+1] = Φ_α(e_1)` and `w[2k] = Φ_α(e'[k]) + w[2k+1]`: two evaluations of `Φ_α`, 16 lookups each. The first round message is `u_0 = Σ_k p[2k]·w[2k]` and `u_2 = Σ_k (p[2k] + p[2k+1])·(w[2k] + w[2k+1])`, two products of `E` by `K` into accumulators per pair. After the challenge `a`, the message folds to `p[2k] + a·(p[2k] + p[2k+1])`, one product of `E` by `K`, and the weight to `w[2k] + a·(w[2k] + w[2k+1])`, one product in `E`, in place. Per pair: `12 + 6 + 9 + 12 = 39` carry-less multiplications and 32 lookups, over `2^23` pairs. Level 0 has no other weight: the commit sample enters at level 1, where its equality table has `2^18` elements.

**`open`, later rounds.** Round `j ≥ 2` has `2^(μ−j)` pairs of elements of `E`: two products into accumulators for the message (12) and two products for the folds (24), 36 per pair and `36·(2^23 − 1)` in all. At the start of level `i ≥ 1` the weights of the new claims are added: the equality table of the level's sample (and of the commit sample at level 1), and for the queries the vector `Σ_j λ_i^j·W_{x_j}`, which is the transpose of the encoder applied to the sparse vector with `λ_i^j` at position `x_j`. It is computed with one transposed transform on the domain of level `i−1`, `c_{i−1}·2^(d_{i−1}−1)` butterflies of `E` by `K`: 4,718,592, 1,835,008, 655,360 and 196,608 for the four later levels, 7,405,568 in all.

**`open`, later commitments.** After the `k_i` rounds of level `i < R−1` the folded message is `f_{i+1}`, already in memory. It is encoded with 16 lanes on the domain of dimension `d_{i+1}`: `16·c_{i+1}·2^(d_{i+1}−1)` butterflies of `E` by `K`, 29,360,128, 10,485,760, 3,145,728 and 524,288, in all 43,515,904. Its tree has `2^(d_{i+1})` leaves of 384 bytes, 6 compressions each: 3,440,636 compressions for the four later trees. The sample `y_{i+1}` is an inner product of `2^(c_i)` elements of `E`.

**Memory at `t = 22`.**

| Buffer | Bytes | Lifetime |
|---|---:|---|
| Rows (shared, not owned) | 128 MiB | caller |
| Level-0 codeword | 256 MiB | `commit` to the queries of level 0 |
| Level-0 tree | 32 MiB | the same |
| Weight `w`, `2^24` elements of `E`, folded in place | 384 MiB | `open`, round 1; 192 MiB after it, halving each round |
| Folded message, `2^23` elements of `E` | 192 MiB | from round 1, halving each round |
| Later codewords | 96, 48, 24, 12 MiB | each from its commitment to its queries |
| Later trees | 16, 8, 4, 2 MiB | the same |
| Tables of `Φ_α`, split equality tables | under 1 MiB | `open` |

The peak is in round 1 of `open`: `256 + 32 + 384 + 192 = 864` MiB owned by the scheme, 992 MiB with the rows (counted). By the time the level-1 codeword is allocated the weight and the message hold `2^18` elements each, 12 MiB together.

#### 10. Reuse

**Ported.** The recursion, its level schedule, the per-level parameter search, the additive transform over `F64` and over `F192` with twiddles in `F64`, the batched BLAKE2s leaf and node hashing, the lane-fold and round kernels, and the induction of query weights by the transposed transform exist in the public leanVM repository, in its `crates/pcs` (`whir.rs`, `whir_config.rs`, `whir_ntt_ext.rs`, `whir_induce.rs`, `ntt.rs`, `merkle.rs`). That repository is under the MIT licence, and its files in `crates/pcs` carry per-file credit and copyright lines with the identifier `Apache-2.0 OR MIT`. A port keeps the header of each source file verbatim at the top of every file derived from it, states in that file that it is modified, and adds the MIT licence text of the source with its copyright line to the repository's third-party notices. Jolt is itself under MIT and Apache-2.0, so no licence conflict arises. The port is a port of algorithms onto `jolt_field::binary`, not a dependency: the source has its own field types, transcript, allocator and thread pool.

**Not ported, and why.** The source commits a root alone and calls its commitment list binding; the commit sample is new. Its ring switch is 64 by 192, for a claim in `F192`, with a batching map built from six challenges and the Frobenius automorphism; the claim here is in `F128`, so the bridge of §3 and its prover kernel are new. The source's lanes are the high variables of the message and its level-0 leaf image is lane-descending; here the lanes are the low variables, which removes the packed copy and the transposition at level 0 and keeps the front end's variable order. The source's wire format and transcript are its own. The source's parameter calculator runs in floating point at configuration time; here its output is a frozen table.

**In the repository already.** `F64`, `F128` and `F192` with carry-less kernels and accumulators, `F192: ExtField<F64>` with `mul_base`, `F128::mul_x`, the Blake2b transcript with `squeeze_bytes`, the `blake2` crate (which provides BLAKE2s-256 for the verifier), `rayon`, and the two traits with their contract tests.

**New.** An accumulator method for an element of `F192` times an element of `F64` (three carry-less multiplications, no reduction); the additive transform; the Merkle tree and multiproof; the bridge on both sides; the protocol, its wire format and its parameter table.
