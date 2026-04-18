extern crate alloc;

use alloc::{sync::Arc, vec::Vec};
use alloy_consensus::crypto::{
    backend::{install_default_provider, CryptoProvider},
    RecoveryError,
};
use alloy_primitives::Address;
#[cfg(feature = "software-keccak")]
use alloy_primitives::Keccak256 as SoftwareKeccak256;
use core::sync::atomic::{AtomicU64, Ordering};
use jolt_inlines_p256::{ecdsa_verify as ecdsa_verify_p256, P256Fr, P256Point};
use jolt_inlines_secp256k1::{
    ecdsa_verify, Secp256k1Fq, Secp256k1Fr, Secp256k1Point, Secp256k1PointExt,
};
use jolt_inlines_sha2::Sha256;
use reth_chainspec::ChainSpec;
use reth_evm_ethereum::EthEvmConfig;
use revm_precompile::{install_crypto, Crypto, DefaultCrypto, PrecompileError};
use serde::{Deserialize, Serialize};
use stateless::{stateless_validation_with_trie, Genesis, StatelessInput, UncompressedPublicKey};
use tries::zeth::SparseState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedStatelessInput {
    pub stateless_input: StatelessInput,
    pub public_keys: Vec<UncompressedPublicKey>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CryptoTraceStats {
    pub keccak_calls: u64,
    pub keccak_input_bytes: u64,
    pub signer_recover_calls: u64,
    pub signer_verify_calls: u64,
    pub precompile_sha256_calls: u64,
    pub precompile_sha256_input_bytes: u64,
    pub precompile_ecrecover_calls: u64,
    pub precompile_ecrecover_fallbacks: u64,
    pub precompile_p256verify_calls: u64,
    pub precompile_p256verify_fallbacks: u64,
    /// Histogram of Keccak input sizes, bucketed by multiples of Keccak's
    /// 136-byte absorption rate. Bucket `i` counts calls with input length in
    /// `[i * 136, (i + 1) * 136)` for `i < 7`; bucket 7 is the `>= 952 bytes`
    /// tail.
    pub keccak_size_hist_136: [u64; 8],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemopsTraceStats {
    pub memcpy_calls: u64,
    pub memcpy_bytes: u64,
    pub memmove_calls: u64,
    pub memmove_bytes: u64,
    pub memcpy_byte_only_calls: u64,
    pub memcpy_aligned_word_calls: u64,
    pub memcpy_misaligned_word_calls: u64,
    pub memmove_forward_byte_only_calls: u64,
    pub memmove_forward_aligned_word_calls: u64,
    pub memmove_forward_misaligned_word_calls: u64,
    pub memmove_backward_byte_only_calls: u64,
    pub memmove_backward_aligned_word_calls: u64,
    pub memmove_backward_misaligned_word_calls: u64,
    pub memcpy_size_hist: [u64; 8],
    pub memcpy_src_align_hist: [u64; 8],
    pub memcpy_dst_align_hist: [u64; 8],
    pub memcpy_align_diff_hist: [u64; 8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationOutput {
    pub success: bool,
    pub block_hash: [u8; 32],
    pub crypto_stats: CryptoTraceStats,
    pub memops_stats: MemopsTraceStats,
}

#[derive(Debug, Clone, Copy)]
struct JoltK256Provider;

#[derive(Debug, Clone, Copy)]
struct JoltRevmCrypto;

const SECP256K1_ORDER: [u64; 4] = [
    0xbfd25e8cd0364141,
    0xbaaedce6af48a03b,
    0xfffffffffffffffe,
    0xffffffffffffffff,
];
const SECP256K1_HALF_ORDER: [u64; 4] = [
    0xdfe92f46681b20a0,
    0x5d576e7357a4501d,
    0xffffffffffffffff,
    0x7fffffffffffffff,
];
const SECP256K1_SQRT_EXP: [u64; 4] = [
    0xffffffffbfffff0c,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0x3fffffffffffffff,
];

static KECCAK_CALLS: AtomicU64 = AtomicU64::new(0);
static KECCAK_INPUT_BYTES: AtomicU64 = AtomicU64::new(0);
static SIGNER_RECOVER_CALLS: AtomicU64 = AtomicU64::new(0);
static SIGNER_VERIFY_CALLS: AtomicU64 = AtomicU64::new(0);
static PRECOMPILE_SHA256_CALLS: AtomicU64 = AtomicU64::new(0);
static PRECOMPILE_SHA256_INPUT_BYTES: AtomicU64 = AtomicU64::new(0);
static PRECOMPILE_ECRECOVER_CALLS: AtomicU64 = AtomicU64::new(0);
static PRECOMPILE_ECRECOVER_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static PRECOMPILE_P256VERIFY_CALLS: AtomicU64 = AtomicU64::new(0);
static PRECOMPILE_P256VERIFY_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static KECCAK_SIZE_HIST_136: [AtomicU64; 8] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn reset_crypto_stats() {
    KECCAK_CALLS.store(0, Ordering::Relaxed);
    KECCAK_INPUT_BYTES.store(0, Ordering::Relaxed);
    SIGNER_RECOVER_CALLS.store(0, Ordering::Relaxed);
    SIGNER_VERIFY_CALLS.store(0, Ordering::Relaxed);
    PRECOMPILE_SHA256_CALLS.store(0, Ordering::Relaxed);
    PRECOMPILE_SHA256_INPUT_BYTES.store(0, Ordering::Relaxed);
    PRECOMPILE_ECRECOVER_CALLS.store(0, Ordering::Relaxed);
    PRECOMPILE_ECRECOVER_FALLBACKS.store(0, Ordering::Relaxed);
    PRECOMPILE_P256VERIFY_CALLS.store(0, Ordering::Relaxed);
    PRECOMPILE_P256VERIFY_FALLBACKS.store(0, Ordering::Relaxed);
    for bucket in KECCAK_SIZE_HIST_136.iter() {
        bucket.store(0, Ordering::Relaxed);
    }
}

fn snapshot_crypto_stats() -> CryptoTraceStats {
    let mut keccak_size_hist_136 = [0u64; 8];
    for (dst, src) in keccak_size_hist_136
        .iter_mut()
        .zip(KECCAK_SIZE_HIST_136.iter())
    {
        *dst = src.load(Ordering::Relaxed);
    }
    CryptoTraceStats {
        keccak_calls: KECCAK_CALLS.load(Ordering::Relaxed),
        keccak_input_bytes: KECCAK_INPUT_BYTES.load(Ordering::Relaxed),
        signer_recover_calls: SIGNER_RECOVER_CALLS.load(Ordering::Relaxed),
        signer_verify_calls: SIGNER_VERIFY_CALLS.load(Ordering::Relaxed),
        precompile_sha256_calls: PRECOMPILE_SHA256_CALLS.load(Ordering::Relaxed),
        precompile_sha256_input_bytes: PRECOMPILE_SHA256_INPUT_BYTES.load(Ordering::Relaxed),
        precompile_ecrecover_calls: PRECOMPILE_ECRECOVER_CALLS.load(Ordering::Relaxed),
        precompile_ecrecover_fallbacks: PRECOMPILE_ECRECOVER_FALLBACKS.load(Ordering::Relaxed),
        precompile_p256verify_calls: PRECOMPILE_P256VERIFY_CALLS.load(Ordering::Relaxed),
        precompile_p256verify_fallbacks: PRECOMPILE_P256VERIFY_FALLBACKS.load(Ordering::Relaxed),
        keccak_size_hist_136,
    }
}

#[cfg(target_arch = "riscv64")]
fn reset_memops_stats() {
    stateless_evm_memops::reset_trace_stats();
}

#[cfg(not(target_arch = "riscv64"))]
fn reset_memops_stats() {}

#[cfg(target_arch = "riscv64")]
fn snapshot_memops_stats() -> MemopsTraceStats {
    let stats = stateless_evm_memops::snapshot_trace_stats();
    MemopsTraceStats {
        memcpy_calls: stats.memcpy_calls,
        memcpy_bytes: stats.memcpy_bytes,
        memmove_calls: stats.memmove_calls,
        memmove_bytes: stats.memmove_bytes,
        memcpy_byte_only_calls: stats.memcpy_byte_only_calls,
        memcpy_aligned_word_calls: stats.memcpy_aligned_word_calls,
        memcpy_misaligned_word_calls: stats.memcpy_misaligned_word_calls,
        memmove_forward_byte_only_calls: stats.memmove_forward_byte_only_calls,
        memmove_forward_aligned_word_calls: stats.memmove_forward_aligned_word_calls,
        memmove_forward_misaligned_word_calls: stats.memmove_forward_misaligned_word_calls,
        memmove_backward_byte_only_calls: stats.memmove_backward_byte_only_calls,
        memmove_backward_aligned_word_calls: stats.memmove_backward_aligned_word_calls,
        memmove_backward_misaligned_word_calls: stats.memmove_backward_misaligned_word_calls,
        memcpy_size_hist: stats.memcpy_size_hist,
        memcpy_src_align_hist: stats.memcpy_src_align_hist,
        memcpy_dst_align_hist: stats.memcpy_dst_align_hist,
        memcpy_align_diff_hist: stats.memcpy_align_diff_hist,
    }
}

#[cfg(not(target_arch = "riscv64"))]
fn snapshot_memops_stats() -> MemopsTraceStats {
    MemopsTraceStats::default()
}

fn be_bytes_to_limbs(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, chunk) in bytes.rchunks_exact(8).enumerate() {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        limbs[i] = u64::from_be_bytes(word);
    }
    limbs
}

fn limbs_to_be_bytes(limbs: [u64; 4]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, limb) in limbs.iter().rev().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    bytes
}

fn cmp_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> core::cmp::Ordering {
    for i in (0..4).rev() {
        match lhs[i].cmp(&rhs[i]) {
            core::cmp::Ordering::Equal => continue,
            non_eq => return non_eq,
        }
    }
    core::cmp::Ordering::Equal
}

fn sub_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (tmp, overflow_1) = lhs[i].overflowing_sub(rhs[i]);
        let (tmp, overflow_2) = tmp.overflowing_sub(u64::from(borrow));
        out[i] = tmp;
        borrow = overflow_1 || overflow_2;
    }
    out
}

fn msg_to_secp256k1_scalar(bytes: &[u8; 32]) -> Secp256k1Fr {
    let mut limbs = be_bytes_to_limbs(bytes);
    if cmp_limbs(&limbs, &SECP256K1_ORDER) != core::cmp::Ordering::Less {
        limbs = sub_limbs(&limbs, &SECP256K1_ORDER);
    }
    Secp256k1Fr::from_u64_arr_unchecked(&limbs)
}

fn signature_to_secp256k1_scalar(bytes: &[u8; 32]) -> Option<Secp256k1Fr> {
    let scalar = Secp256k1Fr::from_u64_arr(&be_bytes_to_limbs(bytes)).ok()?;
    if scalar.is_zero() {
        return None;
    }
    Some(scalar)
}

fn public_key_to_point(public_key: &[u8; 65]) -> Option<Secp256k1Point> {
    if public_key[0] != 0x04 {
        return None;
    }

    let mut x_bytes = [0u8; 32];
    x_bytes.copy_from_slice(&public_key[1..33]);
    let x = Secp256k1Fq::from_u64_arr(&be_bytes_to_limbs(&x_bytes)).ok()?;

    let mut y_bytes = [0u8; 32];
    y_bytes.copy_from_slice(&public_key[33..65]);
    let y = Secp256k1Fq::from_u64_arr(&be_bytes_to_limbs(&y_bytes)).ok()?;

    Secp256k1Point::new(x, y).ok()
}

fn public_key_to_address(public_key: &[u8; 65]) -> Option<Address> {
    if public_key[0] != 0x04 {
        return None;
    }
    let digest = keccak256_digest(&public_key[1..65]);
    Some(Address::from_slice(&digest[12..]))
}

fn address_to_word(address: Address) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(address.as_slice());
    out
}

fn public_key_to_p256_point(public_key: &[u8; 64]) -> Option<P256Point> {
    let mut q_arr = [0u64; 8];

    let mut qx_bytes = [0u8; 32];
    qx_bytes.copy_from_slice(&public_key[..32]);
    q_arr[..4].copy_from_slice(&be_bytes_to_limbs(&qx_bytes));

    let mut qy_bytes = [0u8; 32];
    qy_bytes.copy_from_slice(&public_key[32..64]);
    q_arr[4..].copy_from_slice(&be_bytes_to_limbs(&qy_bytes));

    P256Point::from_u64_arr(&q_arr).ok()
}

fn keccak256_digest(input: &[u8]) -> [u8; 32] {
    KECCAK_CALLS.fetch_add(1, Ordering::Relaxed);
    KECCAK_INPUT_BYTES.fetch_add(input.len() as u64, Ordering::Relaxed);
    let bucket = (input.len() / 136).min(7);
    KECCAK_SIZE_HIST_136[bucket].fetch_add(1, Ordering::Relaxed);

    #[cfg(feature = "software-keccak")]
    {
        let mut hasher = SoftwareKeccak256::new();
        hasher.update(input);
        return hasher.finalize().0;
    }

    #[cfg(not(feature = "software-keccak"))]
    {
        jolt_inlines_keccak256::Keccak256::digest(input)
    }
}

fn secp256k1_fq_pow(base: &Secp256k1Fq, exponent: [u64; 4]) -> Secp256k1Fq {
    let mut acc = Secp256k1Fq::from_u64_arr_unchecked(&[1, 0, 0, 0]);
    for i in (0..256).rev() {
        acc = acc.square();
        if (exponent[i / 64] >> (i % 64)) & 1 == 1 {
            acc = acc.mul(base);
        }
    }
    acc
}

fn secp256k1_fr_point_mul(scalar: &Secp256k1Fr, point: &Secp256k1Point) -> Secp256k1Point {
    let mut res = Secp256k1Point::infinity();
    let k = scalar.e();
    for i in (0..256).rev() {
        if (k[i / 64] >> (i % 64)) & 1 == 1 {
            res = res.double_and_add(point);
        } else {
            res = res.double();
        }
    }
    res
}

fn recover_secp256k1_point(r: &Secp256k1Fr, recid: u8) -> Option<Secp256k1Point> {
    if recid > 1 {
        return None;
    }

    let x = Secp256k1Fq::from_u64_arr(&r.e()).ok()?;
    let y_sq = x.square().mul(&x).add(&Secp256k1Fq::seven());
    let mut y = secp256k1_fq_pow(&y_sq, SECP256K1_SQRT_EXP);
    if y.square() != y_sq {
        return None;
    }

    if (y.e()[0] & 1) != u64::from(recid) {
        y = y.neg();
    }

    Secp256k1Point::new(x, y).ok()
}

fn verify_and_compute_signer(
    public_key: &[u8; 65],
    sig: &[u8; 64],
    msg: &[u8; 32],
) -> Result<Address, RecoveryError> {
    SIGNER_VERIFY_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut r_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&sig[..32]);
    let r = signature_to_secp256k1_scalar(&r_bytes).ok_or_else(RecoveryError::new)?;

    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&sig[32..]);
    let s = signature_to_secp256k1_scalar(&s_bytes).ok_or_else(RecoveryError::new)?;

    let z = msg_to_secp256k1_scalar(msg);
    let q = public_key_to_point(public_key).ok_or_else(RecoveryError::new)?;
    ecdsa_verify(z, r, s, q).map_err(|_| RecoveryError::new())?;

    public_key_to_address(public_key).ok_or_else(RecoveryError::new)
}

fn recover_signer(sig: &[u8; 65], msg: &[u8; 32]) -> Result<Address, RecoveryError> {
    SIGNER_RECOVER_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut r_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&sig[..32]);
    let r = signature_to_secp256k1_scalar(&r_bytes).ok_or_else(RecoveryError::new)?;

    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&sig[32..64]);
    let mut s = signature_to_secp256k1_scalar(&s_bytes).ok_or_else(RecoveryError::new)?;

    let z = msg_to_secp256k1_scalar(msg);
    let mut recid = sig[64];
    if cmp_limbs(&s.e(), &SECP256K1_HALF_ORDER) == core::cmp::Ordering::Greater {
        s = s.neg();
        recid ^= 1;
    }

    let r_point = recover_secp256k1_point(&r, recid).ok_or_else(RecoveryError::new)?;
    let one = Secp256k1Fr::from_u64_arr_unchecked(&[1, 0, 0, 0]);
    let r_inv = one.div(&r);
    let u1 = s.mul(&r_inv);
    let u2 = z.neg().mul(&r_inv);
    let q = secp256k1_fr_point_mul(&u1, &r_point)
        .add(&secp256k1_fr_point_mul(&u2, &Secp256k1Point::generator()));

    if q.is_infinity() || ecdsa_verify(z, r, s, q.clone()).is_err() {
        return Err(RecoveryError::new());
    }

    let mut encoded = [0u8; 65];
    encoded[0] = 0x04;
    encoded[1..33].copy_from_slice(&limbs_to_be_bytes(q.x().e()));
    encoded[33..65].copy_from_slice(&limbs_to_be_bytes(q.y().e()));
    public_key_to_address(&encoded).ok_or_else(RecoveryError::new)
}

fn ecrecover_precompile_address(
    sig: &[u8; 64],
    recid: u8,
    msg: &[u8; 32],
) -> Result<[u8; 32], PrecompileError> {
    let mut encoded_sig = [0u8; 65];
    encoded_sig[..64].copy_from_slice(sig);
    encoded_sig[64] = recid;
    let address =
        recover_signer(&encoded_sig, msg).map_err(|_| PrecompileError::Secp256k1RecoverFailed)?;
    Ok(address_to_word(address))
}

fn verify_p256_signature(msg: &[u8; 32], sig: &[u8; 64], public_key: &[u8; 64]) -> Option<bool> {
    // `jolt-inlines-p256` currently rejects z = 0, while the revm precompile
    // accepts any 32-byte prehash. Preserve exact EVM semantics by falling
    // back to revm for that edge case.
    if msg.iter().all(|byte| *byte == 0) {
        return None;
    }

    let z = P256Fr::from_u64_arr(&be_bytes_to_limbs(msg)).ok()?;

    let mut r_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&sig[..32]);
    let r = P256Fr::from_u64_arr(&be_bytes_to_limbs(&r_bytes)).ok()?;

    let mut s_bytes = [0u8; 32];
    s_bytes.copy_from_slice(&sig[32..]);
    let s = P256Fr::from_u64_arr(&be_bytes_to_limbs(&s_bytes)).ok()?;

    let q = public_key_to_p256_point(public_key)?;
    Some(ecdsa_verify_p256(z, r, s, q).is_ok())
}

impl CryptoProvider for JoltK256Provider {
    fn recover_signer_unchecked(
        &self,
        sig: &[u8; 65],
        msg: &[u8; 32],
    ) -> Result<Address, RecoveryError> {
        recover_signer(sig, msg)
    }

    fn verify_and_compute_signer_unchecked(
        &self,
        pubkey: &[u8; 65],
        sig: &[u8; 64],
        msg: &[u8; 32],
    ) -> Result<Address, RecoveryError> {
        verify_and_compute_signer(pubkey, sig, msg)
    }
}

impl Crypto for JoltRevmCrypto {
    fn sha256(&self, input: &[u8]) -> [u8; 32] {
        PRECOMPILE_SHA256_CALLS.fetch_add(1, Ordering::Relaxed);
        PRECOMPILE_SHA256_INPUT_BYTES.fetch_add(input.len() as u64, Ordering::Relaxed);
        Sha256::digest(input)
    }

    fn secp256k1_ecrecover(
        &self,
        sig: &[u8; 64],
        recid: u8,
        msg: &[u8; 32],
    ) -> Result<[u8; 32], PrecompileError> {
        PRECOMPILE_ECRECOVER_CALLS.fetch_add(1, Ordering::Relaxed);
        ecrecover_precompile_address(sig, recid, msg).or_else(|_| {
            PRECOMPILE_ECRECOVER_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            let fallback = DefaultCrypto;
            fallback.secp256k1_ecrecover(sig, recid, msg)
        })
    }

    fn secp256r1_verify_signature(&self, msg: &[u8; 32], sig: &[u8; 64], pk: &[u8; 64]) -> bool {
        PRECOMPILE_P256VERIFY_CALLS.fetch_add(1, Ordering::Relaxed);
        verify_p256_signature(msg, sig, pk).unwrap_or_else(|| {
            PRECOMPILE_P256VERIFY_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            let fallback = DefaultCrypto;
            fallback.secp256r1_verify_signature(msg, sig, pk)
        })
    }
}

fn install_jolt_crypto() {
    let _ = install_default_provider(Arc::new(JoltK256Provider));
    let _ = install_crypto(JoltRevmCrypto);
}

/// `alloy-primitives` routes `keccak256(...)` through this hook when its
/// `native-keccak` feature is enabled.
///
/// # Safety
///
/// `bytes` must point to `len` readable bytes, and `output` must point to at
/// least 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    let input = unsafe { core::slice::from_raw_parts(bytes, len) };
    let digest = keccak256_digest(input);
    unsafe {
        core::ptr::copy_nonoverlapping(digest.as_ptr(), output, digest.len());
    }
}

#[jolt::provable(
    heap_size = 268435456,
    stack_size = 262144,
    max_input_size = 16777216,
    max_output_size = 256,
    max_trace_length = 134217728
)]
pub fn stateless_validate(input: &[u8]) -> ValidationOutput {
    #[cfg(target_arch = "riscv64")]
    stateless_evm_memops::link_overrides();
    install_jolt_crypto();
    reset_crypto_stats();
    reset_memops_stats();

    let prepared = match postcard::from_bytes::<PreparedStatelessInput>(input) {
        Ok(prepared) => prepared,
        Err(_) => {
            return ValidationOutput {
                success: false,
                block_hash: [0u8; 32],
                crypto_stats: CryptoTraceStats::default(),
                memops_stats: MemopsTraceStats::default(),
            };
        }
    };

    let block_hash = prepared.stateless_input.block.hash_slow().0;
    let genesis = Genesis {
        config: prepared.stateless_input.chain_config.clone(),
        ..Default::default()
    };
    let chain_spec: Arc<ChainSpec> = Arc::new(genesis.into());
    let evm_config = EthEvmConfig::new(chain_spec.clone());

    let success = stateless_validation_with_trie::<SparseState, _, _>(
        prepared.stateless_input.block,
        prepared.public_keys,
        prepared.stateless_input.witness,
        chain_spec,
        evm_config,
    )
    .is_ok();
    let crypto_stats = snapshot_crypto_stats();
    let memops_stats = snapshot_memops_stats();

    ValidationOutput {
        success,
        block_hash,
        crypto_stats,
        memops_stats,
    }
}
