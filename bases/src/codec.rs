// Copyright 2026 FastLabs Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::DecodeError;
use crate::DecodeSliceError;
use crate::Decoder;
use crate::EncodeError;
use crate::EncodedDisplay;
use crate::Encoder;
use crate::codec::sealed::Profile;

/// A named binary-to-text encoding profile.
///
/// Implemented by zero-sized profiles such as [`Base64`](crate::Base64) and
/// [`Base32HexUnpadded`](crate::Base32HexUnpadded). This trait is sealed and
/// cannot be implemented outside this crate.
///
/// # Examples
///
/// ```
/// use bases::Base64;
/// use bases::Codec;
///
/// fn encode_with<C: Codec>(input: &[u8], output: &mut [u8]) -> usize {
///     C::encode_into(input, output).unwrap().len()
/// }
///
/// let mut output = [0_u8; 8];
/// let written = encode_with::<Base64>(b"foo", &mut output);
/// assert_eq!(&output[..written], b"Zm9v");
/// ```
pub trait Codec: sealed::Sealed + Sized + 'static {
    /// Returns the profile's stable human-readable name.
    #[must_use]
    #[inline]
    fn name() -> &'static str {
        Self::Profile::NAME
    }

    /// Computes the exact encoded length.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::LengthOverflow`] if the result cannot be
    /// represented by `usize`. A successful calculation does not guarantee
    /// that a buffer of that size can be allocated.
    #[inline]
    fn encoded_len(input_len: usize) -> Result<usize, EncodeError> {
        Self::Profile::encoded_len_impl(input_len).ok_or(EncodeError::LengthOverflow)
    }

    /// Computes an upper bound for the decoded length.
    ///
    /// This calculation accepts every `usize` without overflow. It does not
    /// inspect or validate encoded input, including its length or padding.
    #[must_use]
    #[inline]
    fn max_decoded_len(input_len: usize) -> usize {
        Self::Profile::max_decoded_len_impl(input_len)
    }

    /// Encodes `input` into `output`, returning the written prefix.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::LengthOverflow`] if the encoded length overflows
    /// `usize`, or [`EncodeError::OutputTooSmall`] if `output` is shorter than
    /// the exact encoded length.
    #[inline]
    fn encode_into<'output>(
        input: &[u8],
        output: &'output mut [u8],
    ) -> Result<&'output mut str, EncodeError> {
        encode_into::<Self>(input, output)
    }

    /// Decodes `input` into `output`, returning the written prefix.
    ///
    /// Decoding is strict: non-alphabet bytes, misplaced padding, and
    /// non-zero unused bits are rejected. After any error, the contents of
    /// `output` are unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeSliceError::Decode`] if `input` is malformed, or
    /// [`DecodeSliceError::OutputTooSmall`] if `output` is shorter than the
    /// decoded length. If `output` is too small, `input` may not have been
    /// fully validated.
    #[inline]
    fn decode_into(
        input: impl AsRef<[u8]>,
        output: &mut [u8],
    ) -> Result<&mut [u8], DecodeSliceError> {
        decode_into::<Self>(input.as_ref(), output)
    }

    /// Decodes `buf` in place, returning the written prefix.
    ///
    /// After any error, the contents of `buf` are unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if `buf` is malformed.
    #[inline]
    fn decode_in_place(buf: &mut [u8]) -> Result<&mut [u8], DecodeError> {
        decode_in_place::<Self>(buf)
    }

    /// Decodes `input` into a fixed-size byte array.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::InvalidLength`] if the decoded length does not
    /// equal `N`, or another [`DecodeError`] variant if `input` is malformed.
    #[inline]
    fn decode_array<const N: usize>(input: impl AsRef<[u8]>) -> Result<[u8; N], DecodeError> {
        decode_array::<Self, N>(input.as_ref())
    }

    /// Encodes `input` into a newly allocated string.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[must_use]
    #[inline]
    fn encode(input: &[u8]) -> String {
        encode::<Self>(input)
    }

    /// Encodes `input` and appends it to `buf`, returning the appended slice.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[inline]
    fn encode_append<'buf>(input: &[u8], buf: &'buf mut String) -> &'buf str {
        encode_append::<Self>(input, buf)
    }

    /// Decodes `input` into a newly allocated byte vector.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if `input` is malformed.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[inline]
    fn decode(input: impl AsRef<[u8]>) -> Result<Vec<u8>, DecodeError> {
        decode::<Self>(input.as_ref())
    }

    /// Decodes `input` and appends the decoded bytes to `buf`, returning the
    /// number of bytes written.
    ///
    /// On error, `buf` is truncated back to its original length.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if `input` is malformed.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[inline]
    fn decode_append(input: impl AsRef<[u8]>, buf: &mut Vec<u8>) -> Result<usize, DecodeError> {
        decode_append::<Self>(input.as_ref(), buf)
    }

    /// Returns a lazy display adapter for encoded `input`.
    ///
    /// Formatting does not allocate. See [`EncodedDisplay`] for formatting
    /// flag behavior.
    #[must_use]
    #[inline]
    fn display(input: &[u8]) -> EncodedDisplay<'_, Self> {
        EncodedDisplay::new(input)
    }

    /// Creates an incremental encoder for this profile.
    #[must_use]
    #[inline]
    fn encoder() -> Encoder<Self> {
        Encoder::new()
    }

    /// Creates an incremental decoder for this profile.
    #[must_use]
    #[inline]
    fn decoder() -> Decoder<Self> {
        Decoder::new()
    }

    /// Wraps `writer` in an encoding writer for this profile.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[must_use]
    #[inline]
    fn encoder_writer<W: std::io::Write>(writer: W) -> crate::io::EncoderWriter<Self, W> {
        crate::io::EncoderWriter::new(writer)
    }

    /// Wraps `reader` in a decoding reader for this profile.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[must_use]
    #[inline]
    fn decoder_reader<R: std::io::Read>(reader: R) -> crate::io::DecoderReader<Self, R> {
        crate::io::DecoderReader::new(reader)
    }

    /// Serializes `bytes` as an encoded string with `serde`.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `serializer`.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[inline]
    fn serialize<T: ?Sized + AsRef<[u8]>, S: serde_core::Serializer>(
        bytes: &T,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        crate::serde::serialize::<Self, T, S>(bytes, serializer)
    }

    /// Deserializes an encoded string into a fixed-size byte array with `serde`.
    ///
    /// # Errors
    ///
    /// Returns a deserialization error if the input is not a valid encoded
    /// string for this profile or does not decode to `N` bytes.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[inline]
    fn deserialize_array<'de, D: serde_core::Deserializer<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<[u8; N], D::Error> {
        crate::serde::deserialize_array::<Self, D, N>(deserializer)
    }

    /// Deserializes an encoded string into a byte vector with `serde`.
    ///
    /// # Errors
    ///
    /// Returns a deserialization error if the input is not a valid encoded
    /// string for this profile.
    #[cfg(all(feature = "serde", feature = "alloc"))]
    #[cfg_attr(docsrs, doc(cfg(all(feature = "serde", feature = "alloc"))))]
    #[inline]
    fn deserialize<'de, D: serde_core::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        crate::serde::deserialize::<Self, D>(deserializer)
    }
}

impl<T> Codec for T where T: sealed::Profile + 'static {}

/// Converts an ASCII byte slice to `&mut str` without re-validating UTF-8.
///
/// # Safety
///
/// Every byte in `bytes` must be valid ASCII (`<= 0x7f`), hence valid UTF-8.
#[allow(unsafe_code)]
#[inline]
pub(crate) unsafe fn ascii_to_str_mut(bytes: &mut [u8]) -> &mut str {
    debug_assert!(bytes.is_ascii());
    // SAFETY: Caller guarantees all bytes are ASCII and therefore valid UTF-8.
    unsafe { core::str::from_utf8_unchecked_mut(bytes) }
}

/// Converts an ASCII byte vector to `String` without re-validating UTF-8.
///
/// # Safety
///
/// Every byte in `bytes` must be valid ASCII (`<= 0x7f`), hence valid UTF-8.
#[cfg(feature = "alloc")]
#[allow(unsafe_code)]
#[inline]
unsafe fn ascii_vec_to_string(bytes: Vec<u8>) -> String {
    debug_assert!(bytes.is_ascii());
    // SAFETY: Caller guarantees all bytes are ASCII and therefore valid UTF-8.
    unsafe { String::from_utf8_unchecked(bytes) }
}

#[cfg(feature = "alloc")]
struct TruncateGuard<'a> {
    vec: &'a mut Vec<u8>,
    len: usize,
}

#[cfg(feature = "alloc")]
impl Drop for TruncateGuard<'_> {
    fn drop(&mut self) {
        self.vec.truncate(self.len);
    }
}

#[cfg(feature = "alloc")]
#[allow(unsafe_code)]
#[inline]
fn append_encoded_ascii<C: Codec>(input: &[u8], buf: &mut String, start: usize, required: usize) {
    let new_len = start
        .checked_add(required)
        .expect("the encoded length fits in usize");
    // SAFETY: `TruncateGuard` restores the valid UTF-8 length `start` if allocation or encoding
    // unwinds, and `C::Profile::encode_impl` writes `required` ASCII bytes into `start..new_len`.
    let vec = unsafe { buf.as_mut_vec() };
    let guard = TruncateGuard { vec, len: start };
    guard.vec.resize(new_len, 0);
    let written = C::Profile::encode_impl(input, &mut guard.vec[start..new_len]);
    debug_assert_eq!(written, required);
    debug_assert!(guard.vec[start..new_len].is_ascii());
    core::mem::forget(guard);
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn encode_into<'output, C: Codec>(
    input: &[u8],
    output: &'output mut [u8],
) -> Result<&'output mut str, EncodeError> {
    let required = C::encoded_len(input.len())?;
    if output.len() < required {
        return Err(EncodeError::OutputTooSmall {
            required,
            available: output.len(),
        });
    }

    let written = C::Profile::encode_impl(input, output);
    debug_assert_eq!(written, required);

    // SAFETY: Binary-to-text backends always write ASCII bytes into `output[..written]`.
    Ok(unsafe { ascii_to_str_mut(&mut output[..written]) })
}

#[inline]
pub(crate) fn decode_into<'output, C: Codec>(
    input: &[u8],
    output: &'output mut [u8],
) -> Result<&'output mut [u8], DecodeSliceError> {
    let required = C::Profile::decoded_len_impl(input)?;
    if output.len() < required {
        return Err(DecodeSliceError::OutputTooSmall {
            required,
            available: output.len(),
        });
    }

    let written = C::Profile::decode_impl(input, output)?;
    debug_assert_eq!(written, required);
    Ok(&mut output[..written])
}

#[inline]
pub(crate) fn decode_in_place<C: Codec>(buf: &mut [u8]) -> Result<&mut [u8], DecodeError> {
    let written = C::Profile::decode_in_place_impl(buf)?;
    Ok(&mut buf[..written])
}

#[inline]
pub(crate) fn decode_array<C: Codec, const N: usize>(input: &[u8]) -> Result<[u8; N], DecodeError> {
    let required = C::Profile::decoded_len_impl(input)?;
    if required != N {
        return Err(DecodeError::InvalidLength {
            length: input.len(),
        });
    }

    let mut output = [0_u8; N];
    let written = C::Profile::decode_impl(input, &mut output)?;
    debug_assert_eq!(written, N);
    Ok(output)
}

#[cfg(feature = "alloc")]
#[allow(unsafe_code)]
#[inline]
pub(crate) fn encode<C: Codec>(input: &[u8]) -> String {
    let required =
        C::encoded_len(input.len()).expect("the encoded length of an existing slice fits in usize");
    let mut output = vec![0_u8; required];
    let written = C::Profile::encode_impl(input, &mut output);
    debug_assert_eq!(written, required);
    // SAFETY: Binary-to-text backends always write ASCII bytes into `output`.
    unsafe { ascii_vec_to_string(output) }
}

#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn encode_append<'buf, C: Codec>(input: &[u8], buf: &'buf mut String) -> &'buf str {
    let start = buf.len();
    let required =
        C::encoded_len(input.len()).expect("the encoded length of an existing slice fits in usize");
    if required != 0 {
        append_encoded_ascii::<C>(input, buf, start, required);
    }
    debug_assert_eq!(buf.len() - start, required);
    &buf[start..]
}

#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn decode<C: Codec>(input: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let required = C::Profile::decoded_len_impl(input)?;
    let mut output = vec![0_u8; required];
    let written = C::Profile::decode_impl(input, &mut output)?;
    debug_assert_eq!(written, required);
    Ok(output)
}

#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn decode_append<C: Codec>(
    input: &[u8],
    buf: &mut Vec<u8>,
) -> Result<usize, DecodeError> {
    let required = C::Profile::decoded_len_impl(input)?;
    let original_len = buf.len();
    let new_len = original_len
        .checked_add(required)
        .expect("decoded vector length overflow");
    buf.resize(new_len, 0);

    match C::Profile::decode_impl(input, &mut buf[original_len..]) {
        Ok(written) => {
            debug_assert_eq!(written, required);
            Ok(written)
        }
        Err(error) => {
            buf.truncate(original_len);
            Err(error)
        }
    }
}

pub(crate) mod sealed {
    use crate::DecodeError;

    /// A direct `Codec: Profile` bound would let downstream generic code skip
    /// padding and output-capacity checks:
    ///
    /// ```compile_fail,E0599
    /// use bases::Codec;
    /// fn unchecked<C: Codec>(input: &[u8], output: &mut [u8]) {
    ///     let _ = C::decode_impl(input, output);
    /// }
    /// ```
    pub trait Sealed {
        type Profile: Profile;
    }

    impl<T: Profile> Sealed for T {
        type Profile = T;
    }

    /// Quanta must satisfy `0 < INPUT_QUANTUM < ENCODED_QUANTUM` and fit the
    /// stream buffers; sizing and buffering rely on these bounds.
    pub trait Profile {
        const NAME: &'static str;
        const INPUT_QUANTUM: usize;
        const ENCODED_QUANTUM: usize;
        const PADDED: bool;

        fn encoded_len_impl(input_len: usize) -> Option<usize>;
        fn max_decoded_len_impl(input_len: usize) -> usize;
        fn decoded_len_impl(input: &[u8]) -> Result<usize, DecodeError>;
        fn encode_impl(input: &[u8], output: &mut [u8]) -> usize;
        fn decode_impl(input: &[u8], output: &mut [u8]) -> Result<usize, DecodeError>;
        fn decode_in_place_impl(buf: &mut [u8]) -> Result<usize, DecodeError>;
        fn decoded_padding_reduction(padding: usize) -> Option<usize>;
    }
}
