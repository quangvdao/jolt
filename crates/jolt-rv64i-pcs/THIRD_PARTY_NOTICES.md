# Third-party notices

The port manifest of `specs/rv64i-binary-commitment.md`, section 10, is
from the public leanVM repository at revision
`48a904208d682848dac0e18ef8b01ebfc40df9ad`. The MIT option is taken where a file
offers a choice. Source headers are retained by each derived file, which also
identifies its source path, revision and modifications.

## Port manifest

- `crates/pcs/src/whir.rs`
- `crates/pcs/src/whir_config.rs`
- `crates/pcs/src/whir_ntt_ext.rs`
- `crates/pcs/src/whir_induce.rs`
- `crates/pcs/src/ntt.rs`
- `crates/pcs/src/ntt/additive_ntt_f64.rs`
- `crates/pcs/src/merkle.rs`
- `crates/fiat_shamir/src/merkle.rs`
- `crates/primitives/src/hash.rs` (batched BLAKE2s in `src/arch/`)

## Source copyright and credit lines

`crates/pcs/src/whir.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
```

`crates/pcs/src/whir_config.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// CREDIT: https://github.com/bcc-research/bolt-rs, MIT.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
```

`crates/pcs/src/whir_ntt_ext.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
```

`crates/pcs/src/whir_induce.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
```

`crates/pcs/src/ntt.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
```

`crates/pcs/src/merkle.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
```

`crates/fiat_shamir/src/merkle.rs`:

```text
// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
```

## MIT licence

MIT License

Copyright (c) 2026 leanEthereum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
