use ark_serialize::{CanonicalDeserialize, Compress, SerializationError, Validate};

/// A lightweight, allocation-free reader over a byte slice.
///
/// This is used by the proof-bundle streaming verifier to parse inputs without
/// deserializing large owned structures (e.g. `JoltProof`).
#[derive(Clone, Copy)]
pub struct SliceReader<'a> {
    buf: &'a [u8],
}

#[derive(Debug)]
pub enum DecodeError {
    Eof { needed: usize, remaining: usize },
    Invalid(&'static str),
    Ark(SerializationError),
}

impl<'a> SliceReader<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn as_slice(&self) -> &'a [u8] {
        self.buf
    }

    #[inline]
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.buf.len() < n {
            return Err(DecodeError::Eof {
                needed: n,
                remaining: self.buf.len(),
            });
        }
        let (head, tail) = self.buf.split_at(n);
        self.buf = tail;
        Ok(head)
    }

    #[inline]
    pub fn read_u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    #[inline]
    pub fn read_u32_le(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    pub fn read_u64_le(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Reads a `usize` encoded as little-endian `u64`.
    #[inline]
    pub fn read_usize_u64_le(&mut self) -> Result<usize, DecodeError> {
        let v = self.read_u64_le()?;
        usize::try_from(v).map_err(|_| DecodeError::Invalid("usize overflow"))
    }

    /// Reads a length-delimited byte slice where the length is encoded as `u32` LE.
    #[inline]
    pub fn read_bytes_u32_len(&mut self) -> Result<&'a [u8], DecodeError> {
        let n = self.read_u32_le()? as usize;
        self.take(n)
    }

    /// Reads a BN254 G1 affine point encoded as:
    /// - u8 is_infinity (0/1)
    /// - x (Fq, canonical compressed)
    /// - y (Fq, canonical compressed)
    ///
    /// This avoids curve-point decompression (the host should store points uncompressed).
    #[inline]
    pub fn read_bn254_g1_affine_uncompressed_unchecked(
        &mut self,
    ) -> Result<ark_bn254::G1Affine, DecodeError> {
        let is_inf = self.read_u8()?;
        if is_inf == 1 {
            return Ok(ark_bn254::G1Affine::identity());
        }
        if is_inf != 0 {
            return Err(DecodeError::Invalid("invalid infinity flag"));
        }
        let x: ark_bn254::Fq = self.read_canonical(Compress::Yes, Validate::No)?;
        let y: ark_bn254::Fq = self.read_canonical(Compress::Yes, Validate::No)?;
        Ok(ark_bn254::G1Affine::new_unchecked(x, y))
    }

    /// Reads a Grumpkin affine point encoded as:
    /// - u8 is_infinity (0/1)
    /// - x (Fq, canonical compressed)
    /// - y (Fq, canonical compressed)
    ///
    /// This avoids curve-point decompression (the host should store points uncompressed).
    #[inline]
    pub fn read_grumpkin_affine_uncompressed_unchecked(
        &mut self,
    ) -> Result<ark_grumpkin::Affine, DecodeError> {
        let is_inf = self.read_u8()?;
        if is_inf == 1 {
            return Ok(ark_grumpkin::Affine::identity());
        }
        if is_inf != 0 {
            return Err(DecodeError::Invalid("invalid infinity flag"));
        }
        let x: ark_grumpkin::Fq = self.read_canonical(Compress::Yes, Validate::No)?;
        let y: ark_grumpkin::Fq = self.read_canonical(Compress::Yes, Validate::No)?;
        Ok(ark_grumpkin::Affine::new_unchecked(x, y))
    }

    /// Reads an arkworks canonical object from the remaining bytes, consuming exactly
    /// the bytes read by the deserializer.
    ///
    /// This is intended for *small* sub-objects inside the proof bundle (e.g. configs),
    /// not for the full proof.
    #[inline]
    pub fn read_canonical<T: CanonicalDeserialize>(
        &mut self,
        compress: Compress,
        validate: Validate,
    ) -> Result<T, DecodeError> {
        // `&mut &[u8]` implements `Read` and advances the slice as bytes are consumed.
        T::deserialize_with_mode(&mut self.buf, compress, validate).map_err(DecodeError::Ark)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_serialize::CanonicalSerialize;

    #[test]
    fn read_primitives_and_bytes() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(b"abc");

        let mut r = SliceReader::new(&buf);
        assert_eq!(r.read_u32_le().unwrap(), 1);
        assert_eq!(r.read_u64_le().unwrap(), 2);
        let bytes = r.read_bytes_u32_len().unwrap();
        assert_eq!(bytes, b"abc");
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn read_canonical_advances_slice() {
        let mut buf = Vec::new();
        1234u32.serialize_compressed(&mut buf).unwrap();
        9999u32.serialize_compressed(&mut buf).unwrap();

        let mut r = SliceReader::new(&buf);
        let a: u32 = r
            .read_canonical(Compress::Yes, Validate::No)
            .expect("decode a");
        let b: u32 = r
            .read_canonical(Compress::Yes, Validate::No)
            .expect("decode b");
        assert_eq!(a, 1234);
        assert_eq!(b, 9999);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn read_affine_uncompressed_unchecked_roundtrip_g1() {
        use ark_bn254::G1Affine;
        use ark_ec::AffineRepr;

        let p = G1Affine::generator();
        let mut buf = Vec::new();
        buf.push(0u8);
        p.x.serialize_compressed(&mut buf).unwrap();
        p.y.serialize_compressed(&mut buf).unwrap();

        let mut r = SliceReader::new(&buf);
        let q = r.read_bn254_g1_affine_uncompressed_unchecked().unwrap();
        assert_eq!(p, q);
        assert_eq!(r.remaining(), 0);
    }
}
