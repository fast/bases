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
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::ptr;
use core::slice;
use core::str;
#[cfg(feature = "std")]
use std::io;

#[cfg(feature = "serde")]
use serde_core::Deserializer;
#[cfg(feature = "serde")]
use serde_core::Serializer;

use crate::DecodeError;
use crate::DecodeSliceError;
use crate::Decoder;
use crate::EncodeError;
use crate::EncodedDisplay;
use crate::Encoder;
use crate::codec::sealed::Profile;
#[cfg(feature = "std")]
use crate::io::DecoderReader;
#[cfg(feature = "std")]
use crate::io::EncoderWriter;
#[cfg(feature = "serde")]
use crate::serde;

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

    /// Validates encoded `input` without allocating or writing decoded bytes.
    ///
    /// This enforces the profile's full alphabet, padding, case, and canonical
    /// trailing-bit rules. Calling `validate` immediately before decoding the
    /// same input duplicates validation work.
    ///
    /// # Errors
    ///
    /// Returns the same [`DecodeError`], including error precedence and byte
    /// offset, as decoding with sufficient output capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use bases::Base64;
    /// use bases::DecodeError;
    ///
    /// assert_eq!(Base64::validate("Zg=="), Ok(()));
    /// assert_eq!(
    ///     Base64::validate("Zh=="),
    ///     Err(DecodeError::NonCanonical { index: 1 }),
    /// );
    /// ```
    #[inline]
    fn validate(input: impl AsRef<[u8]>) -> Result<(), DecodeError> {
        validate::<Self>(input.as_ref())
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

    /// Encodes into possibly uninitialized storage and returns the initialized ASCII prefix.
    ///
    /// Previously initialized slots stay initialized on success and error. Their values may change.
    ///
    /// # Errors
    ///
    /// Returns the same length and capacity errors as [`encode_into`](Self::encode_into).
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    ///
    /// use bases::Base64;
    ///
    /// let mut output = [MaybeUninit::uninit(); 8];
    /// let text = Base64::encode_into_uninit(b"foo", &mut output).unwrap();
    /// assert_eq!(text, "Zm9v");
    /// ```
    #[inline]
    fn encode_into_uninit<'output>(
        input: &[u8],
        output: &'output mut [MaybeUninit<u8>],
    ) -> Result<&'output mut str, EncodeError> {
        encode_into_uninit::<Self>(input, output)
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

    /// Decodes `input` into possibly uninitialized storage and returns the initialized prefix.
    ///
    /// An error may leave partial writes and gives no initialized length. Previously initialized
    /// slots remain initialized, but their values may change.
    ///
    /// # Errors
    ///
    /// Returns the same decoding and capacity errors as [`decode_into`](Self::decode_into).
    /// An undersized buffer can be reported before the entire input has been validated.
    ///
    /// # Examples
    ///
    /// Writing into a vector's spare capacity leaves its length unchanged. Use the returned
    /// slice's length to include the decoded bytes in the vector.
    ///
    /// ```
    /// use bases::Base64;
    ///
    /// let mut bytes = Vec::with_capacity(8);
    /// bytes.extend_from_slice(b"id:");
    /// let start = bytes.len();
    /// let written = Base64::decode_into_uninit("Zg==", bytes.spare_capacity_mut())?.len();
    /// // SAFETY: The successful decode initialized the next `written` bytes of spare capacity.
    /// unsafe { bytes.set_len(start + written) };
    /// assert_eq!(bytes.as_slice(), b"id:f");
    /// # Ok::<(), bases::DecodeSliceError>(())
    /// ```
    #[inline]
    fn decode_into_uninit(
        input: impl AsRef<[u8]>,
        output: &mut [MaybeUninit<u8>],
    ) -> Result<&mut [u8], DecodeSliceError> {
        decode_into_uninit::<Self>(input.as_ref(), output)
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
    fn encoder_writer<W: io::Write>(writer: W) -> EncoderWriter<Self, W> {
        EncoderWriter::new(writer)
    }

    /// Wraps `reader` in a decoding reader for this profile.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[must_use]
    #[inline]
    fn decoder_reader<R: io::Read>(reader: R) -> DecoderReader<Self, R> {
        DecoderReader::new(reader)
    }

    /// Serializes `bytes` as an encoded string with `serde`.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `serializer`.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[inline]
    fn serialize<T: ?Sized + AsRef<[u8]>, S: Serializer>(
        bytes: &T,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serde::serialize::<Self, T, S>(bytes, serializer)
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
    fn deserialize_array<'de, D: Deserializer<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<[u8; N], D::Error> {
        serde::deserialize_array::<Self, D, N>(deserializer)
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
    fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        serde::deserialize::<Self, D>(deserializer)
    }
}

impl<T> Codec for T where T: sealed::Profile + 'static {}

/// Views an initialized byte slice as write-only `MaybeUninit<u8>` slots.
#[allow(unsafe_code)]
#[inline]
pub fn as_uninit_slice(bytes: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    let len = bytes.len();
    // SAFETY: `MaybeUninit<u8>` has the same layout and alignment as `u8`, `bytes` is uniquely
    // borrowed for the returned lifetime, and backends only write initialized `u8` values so
    // `bytes` remains initialized even on error.
    unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<MaybeUninit<u8>>(), len) }
}

/// # Safety
///
/// Every slot in `bytes` must be initialized.
#[allow(unsafe_code)]
#[inline]
pub unsafe fn assume_init_slice_mut(bytes: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    // SAFETY: The caller guarantees initialization; the cast preserves layout and borrow.
    unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u8>(), bytes.len()) }
}

/// Converts an ASCII byte slice to `&mut str` without re-validating UTF-8.
///
/// # Safety
///
/// Every byte in `bytes` must be valid ASCII (`<= 0x7f`), hence valid UTF-8.
#[allow(unsafe_code)]
#[inline]
pub unsafe fn ascii_to_str_mut(bytes: &mut [u8]) -> &mut str {
    debug_assert!(bytes.is_ascii());
    // SAFETY: Caller guarantees all bytes are ASCII and therefore valid UTF-8.
    unsafe { str::from_utf8_unchecked_mut(bytes) }
}

/// Converts an initialized ASCII `MaybeUninit<u8>` slice to `&str` without re-validating UTF-8.
///
/// # Safety
///
/// Every byte in `bytes` must be initialized and valid ASCII (`<= 0x7f`), hence valid UTF-8.
#[allow(unsafe_code)]
#[inline]
pub unsafe fn uninit_ascii_to_str(bytes: &[MaybeUninit<u8>]) -> &str {
    let len = bytes.len();
    // SAFETY: Caller guarantees all `len` elements are initialized ASCII bytes, and
    // `MaybeUninit<u8>` has the same layout and alignment as `u8`.
    let init = unsafe { slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), len) };
    debug_assert!(init.is_ascii());
    // SAFETY: Caller guarantees all bytes are ASCII and therefore valid UTF-8.
    unsafe { str::from_utf8_unchecked(init) }
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
#[allow(unsafe_code)]
#[inline]
fn append_encoded_ascii<C: Codec>(input: &[u8], buf: &mut String, start: usize, required: usize) {
    let new_len = start
        .checked_add(required)
        .expect("the encoded length fits in usize");
    // SAFETY: `buf.as_mut_vec()` borrows the backing UTF-8 buffer; we reserve `required` bytes
    // and only advance the length to `new_len` after `encode_impl` initializes `start..new_len`
    // with ASCII bytes, leaving the existing `..start` prefix untouched on unwind.
    let vec = unsafe { buf.as_mut_vec() };
    vec.reserve(required);
    let written = C::Profile::encode_impl(input, &mut vec.spare_capacity_mut()[..required]);
    debug_assert_eq!(written, required);
    // SAFETY: `encode_impl` wrote `required` initialized ASCII bytes into `start..new_len`,
    // preserving valid UTF-8 for `buf`.
    unsafe { vec.set_len(new_len) };
    debug_assert!(vec[start..new_len].is_ascii());
}

#[inline]
pub fn encode_into<'output, C: Codec>(
    input: &[u8],
    output: &'output mut [u8],
) -> Result<&'output mut str, EncodeError> {
    encode_into_uninit::<C>(input, as_uninit_slice(output))
}

#[allow(unsafe_code)]
#[inline]
pub fn encode_into_uninit<'output, C: Codec>(
    input: &[u8],
    output: &'output mut [MaybeUninit<u8>],
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

    let output = &mut output[..written];
    // SAFETY: The backend initialized this prefix as ASCII.
    Ok(unsafe {
        let bytes = assume_init_slice_mut(output);
        ascii_to_str_mut(bytes)
    })
}

#[inline]
pub fn validate<C: Codec>(input: &[u8]) -> Result<(), DecodeError> {
    let plan = C::Profile::decode_plan_impl(input)?;
    C::Profile::validate_with_plan_impl(input, plan)
}

#[inline]
pub fn decode_into<'output, C: Codec>(
    input: &[u8],
    output: &'output mut [u8],
) -> Result<&'output mut [u8], DecodeSliceError> {
    decode_into_uninit::<C>(input, as_uninit_slice(output))
}

#[allow(unsafe_code)]
#[inline]
pub fn decode_into_uninit<'output, C: Codec>(
    input: &[u8],
    output: &'output mut [MaybeUninit<u8>],
) -> Result<&'output mut [u8], DecodeSliceError> {
    let plan = C::Profile::decode_plan_impl(input)?;
    if output.len() < plan.required {
        return Err(DecodeSliceError::OutputTooSmall {
            required: plan.required,
            available: output.len(),
        });
    }

    let out = &mut output[..plan.required];
    let written = C::Profile::decode_with_plan_impl(input, plan, out)?;
    debug_assert_eq!(written, plan.required);
    // SAFETY: Successful decoding initializes exactly the `plan.required` bytes in `out`.
    Ok(unsafe { assume_init_slice_mut(out) })
}

#[inline]
pub fn decode_in_place<C: Codec>(buf: &mut [u8]) -> Result<&mut [u8], DecodeError> {
    let written = C::Profile::decode_in_place_impl(buf)?;
    Ok(&mut buf[..written])
}

#[allow(unsafe_code)]
#[inline]
pub fn decode_array<C: Codec, const N: usize>(input: &[u8]) -> Result<[u8; N], DecodeError> {
    let plan = C::Profile::decode_plan_impl(input)?;
    if plan.required != N {
        return Err(DecodeError::InvalidLength {
            length: input.len(),
        });
    }

    let mut output = [MaybeUninit::<u8>::uninit(); N];
    let written = C::Profile::decode_with_plan_impl(input, plan, &mut output)?;
    debug_assert_eq!(written, N);
    // SAFETY: `decode_with_plan_impl` succeeded and initialized all `N` elements of `output`, and
    // `[MaybeUninit<u8>; N]` has the same layout and alignment as `[u8; N]`.
    Ok(unsafe { ptr::read(output.as_ptr().cast::<[u8; N]>()) })
}

#[cfg(feature = "alloc")]
#[allow(unsafe_code)]
#[inline]
pub fn encode<C: Codec>(input: &[u8]) -> String {
    let required =
        C::encoded_len(input.len()).expect("the encoded length of an existing slice fits in usize");
    let mut output = Vec::with_capacity(required);
    let written = C::Profile::encode_impl(input, &mut output.spare_capacity_mut()[..required]);
    debug_assert_eq!(written, required);
    // SAFETY: `encode_impl` wrote `written == required` initialized ASCII bytes into
    // `output.spare_capacity_mut()[..required]`, and `output.capacity() >= required`.
    unsafe {
        output.set_len(required);
        ascii_vec_to_string(output)
    }
}

#[cfg(feature = "alloc")]
#[inline]
pub fn encode_append<'buf, C: Codec>(input: &[u8], buf: &'buf mut String) -> &'buf str {
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
#[allow(unsafe_code)]
#[inline]
pub fn decode<C: Codec>(input: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let plan = C::Profile::decode_plan_impl(input)?;
    let mut output = Vec::with_capacity(plan.required);
    let written = C::Profile::decode_with_plan_impl(
        input,
        plan,
        &mut output.spare_capacity_mut()[..plan.required],
    )?;
    debug_assert_eq!(written, plan.required);
    // SAFETY: `decode_with_plan_impl` succeeded and initialized all `plan.required` bytes in
    // `output.spare_capacity_mut()[..plan.required]`.
    unsafe { output.set_len(plan.required) };
    Ok(output)
}

#[cfg(feature = "alloc")]
#[allow(unsafe_code)]
#[inline]
pub fn decode_append<C: Codec>(input: &[u8], buf: &mut Vec<u8>) -> Result<usize, DecodeError> {
    let plan = C::Profile::decode_plan_impl(input)?;
    let original_len = buf.len();
    let new_len = original_len
        .checked_add(plan.required)
        .expect("decoded vector length overflow");
    buf.reserve(plan.required);

    let written = C::Profile::decode_with_plan_impl(
        input,
        plan,
        &mut buf.spare_capacity_mut()[..plan.required],
    )?;
    debug_assert_eq!(written, plan.required);
    // SAFETY: `decode_with_plan_impl` succeeded and initialized all `plan.required` bytes in
    // `buf.spare_capacity_mut()[..plan.required]`.
    unsafe { buf.set_len(new_len) };
    Ok(written)
}

pub(crate) mod sealed {
    use core::mem::MaybeUninit;

    use crate::DecodeError;
    use crate::backend::DecodePlan;
    use crate::policy::sealed::DecodePadding;

    /// A direct `Codec: Profile` bound would let downstream generic code skip
    /// padding and output-capacity checks:
    ///
    /// ```compile_fail,E0599
    /// use bases::Codec;
    /// fn unchecked<C: Codec>(input: &[u8], output: &mut [u8]) {
    ///     let _ = C::decode_with_plan_impl(input, output);
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
    ///
    /// Output methods must initialize the full reported prefix on success, using ASCII bytes
    /// for encoding. Every write must preserve initialization of previously initialized slots,
    /// including when decoding returns an error.
    pub trait Profile {
        const NAME: &'static str;
        const INPUT_QUANTUM: usize;
        const ENCODED_QUANTUM: usize;
        const DECODE_PADDING: DecodePadding;

        fn encoded_len_impl(input_len: usize) -> Option<usize>;
        fn max_decoded_len_impl(input_len: usize) -> usize;
        fn decode_plan_impl(input: &[u8]) -> Result<DecodePlan, DecodeError>;
        fn validate_with_plan_impl(input: &[u8], plan: DecodePlan) -> Result<(), DecodeError>;
        fn encode_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize;
        fn encode_quantum_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize;
        fn decode_with_plan_impl(
            input: &[u8],
            plan: DecodePlan,
            output: &mut [MaybeUninit<u8>],
        ) -> Result<usize, DecodeError>;
        fn decode_quantum_impl(
            input: &[u8],
            output: &mut [MaybeUninit<u8>],
        ) -> Result<usize, DecodeError>;
        fn decode_in_place_impl(buf: &mut [u8]) -> Result<usize, DecodeError>;
        fn decoded_padding_reduction(padding: usize) -> Option<usize>;
    }
}

#[cfg(test)]
mod tests {
    use core::hint;
    use core::mem::MaybeUninit;

    use crate::Base16Lower;
    use crate::Base32;
    use crate::Base64;
    use crate::DecodeError;
    use crate::DecodeSliceError;

    #[test]
    fn initialized_storage_remains_readable_after_decode_error() {
        let mut storage = [7_u8; 12];
        assert_eq!(
            Base64::decode_into("Zm9v!!!!", &mut storage[2..8]),
            Err(DecodeSliceError::Decode(DecodeError::InvalidByte {
                index: 4,
                byte: b'!',
            })),
        );
        assert_eq!(&storage[..2], &[7; 2]);
        assert_eq!(&storage[8..], &[7; 4]);
        for byte in storage {
            hint::black_box(byte);
        }
    }

    #[test]
    fn uninitialized_results_are_borrowed_prefixes() {
        let mut encoded = [MaybeUninit::<u8>::uninit(); 8];
        let encoded_ptr = encoded.as_ptr().cast::<u8>();
        let text = Base64::encode_into_uninit(b"foo", &mut encoded).unwrap();
        assert_eq!(text, "Zm9v");
        assert_eq!(text.as_ptr(), encoded_ptr);

        let mut decoded = [MaybeUninit::<u8>::uninit(); 8];
        let decoded_ptr = decoded.as_ptr().cast::<u8>();
        let bytes = Base64::decode_into_uninit("/w==", &mut decoded).unwrap();
        assert_eq!(bytes, [0xff]);
        assert_eq!(bytes.as_ptr(), decoded_ptr);

        let mut hex = [MaybeUninit::<u8>::uninit(); 2];
        assert_eq!(
            Base16Lower::encode_into_uninit(b"\xff", &mut hex).unwrap(),
            "ff"
        );
        assert_eq!(
            Base16Lower::decode_into_uninit("ff", &mut hex).unwrap(),
            [0xff]
        );
    }

    #[test]
    fn base32_bulk_output_initializes_only_the_successful_prefix() {
        let mut raw = [0_u8; 161];
        raw[160] = b'f';
        let mut wire = [b'A'; 264];
        wire[256..].copy_from_slice(b"MY======");

        let mut encoded = [MaybeUninit::<u8>::uninit(); 272];
        assert_eq!(
            Base32::encode_into_uninit(&raw, &mut encoded)
                .unwrap()
                .as_bytes(),
            wire,
        );
        let mut decoded = [MaybeUninit::<u8>::uninit(); Base32::max_decoded_len(264)];
        assert_eq!(Base32::decode_into_uninit(wire, &mut decoded).unwrap(), raw);

        wire[255] = b'!';
        let mut output = [MaybeUninit::<u8>::uninit(); Base32::max_decoded_len(264)];
        assert!(Base32::decode_into_uninit(wire, &mut output).is_err());
    }
}
