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

//! Base32 codec profiles.

#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;
use core::hash::Hash;
use core::hash::Hasher;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
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
use crate::backend;
use crate::backend::DecodePlan;
use crate::codec;
use crate::codec::sealed::Profile;
#[cfg(feature = "std")]
use crate::io::DecoderReader;
#[cfg(feature = "std")]
use crate::io::EncoderWriter;
use crate::policy;
use crate::policy::sealed::DecodeCasePolicy;
use crate::policy::sealed::DecodePadding;
use crate::policy::sealed::LetterCase;
use crate::policy::sealed::PaddingPolicy;
use crate::policy::sealed::profile_name;
#[cfg(feature = "serde")]
use crate::serde;

/// The RFC 4648 standard Base32 alphabet (`A-Z`, `2-7`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Standard;

/// The RFC 4648 Extended Hex Base32 alphabet (`0-9`, `A-V`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ExtendedHex;

impl sealed::Base32Alphabet for Standard {
    const EXTENDED_HEX: bool = false;
}

impl sealed::Base32Alphabet for ExtendedHex {
    const EXTENDED_HEX: bool = true;
}

pub(crate) mod sealed {
    pub trait Base32Alphabet: 'static {
        const EXTENDED_HEX: bool;
    }
}

/// A configurable RFC 4648 Base32 profile.
///
/// The type parameters select the alphabet (`Alphabet`, defaulting to
/// [`Standard`]), padding policy (`Padding`, defaulting to
/// [`policy::Padded`]), encoded letter case (`Case`, defaulting to
/// [`policy::Upper`]), and decoding case policy (`DecodeCase`, defaulting to
/// [`policy::Strict`]). Use [`policy::PaddedOptional`] or [`policy::UnpaddedOptional`] to accept
/// both input forms while emitting padded or unpadded output, respectively.
pub struct Base32<
    Alphabet = Standard,
    Padding = policy::Padded,
    Case = policy::Upper,
    DecodeCase = policy::Strict,
>(PhantomData<(Alphabet, Padding, Case, DecodeCase)>);

impl<Alphabet, Padding, Case, DecodeCase> Clone for Base32<Alphabet, Padding, Case, DecodeCase> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Alphabet, Padding, Case, DecodeCase> Copy for Base32<Alphabet, Padding, Case, DecodeCase> {}

impl<Alphabet, Padding, Case, DecodeCase> fmt::Debug
    for Base32<Alphabet, Padding, Case, DecodeCase>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Base32").finish()
    }
}

impl<Alphabet, Padding, Case, DecodeCase> Default for Base32<Alphabet, Padding, Case, DecodeCase> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Alphabet, Padding, Case, DecodeCase> PartialEq
    for Base32<Alphabet, Padding, Case, DecodeCase>
{
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<Alphabet, Padding, Case, DecodeCase> Eq for Base32<Alphabet, Padding, Case, DecodeCase> {}

impl<Alphabet, Padding, Case, DecodeCase> Hash for Base32<Alphabet, Padding, Case, DecodeCase> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<Alphabet, Padding, Case, DecodeCase> Base32<Alphabet, Padding, Case, DecodeCase> {
    /// Creates a profile value.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Alphabet, Padding, Case, DecodeCase> Base32<Alphabet, Padding, Case, DecodeCase>
where
    Alphabet: sealed::Base32Alphabet,
    Padding: PaddingPolicy,
    Case: LetterCase,
    DecodeCase: DecodeCasePolicy,
{
    /// Returns the profile's stable human-readable name.
    #[must_use]
    #[inline]
    pub const fn name() -> &'static str {
        match (Alphabet::EXTENDED_HEX, Case::UPPERCASE, DecodeCase::STRICT) {
            (false, true, true) => profile_name!("base32", Padding),
            (false, false, true) => profile_name!("base32-lower", Padding),
            (false, true, false) => profile_name!("base32-mixed", Padding),
            (false, false, false) => profile_name!("base32-lower-mixed", Padding),
            (true, true, true) => profile_name!("base32hex", Padding),
            (true, false, true) => profile_name!("base32hex-lower", Padding),
            (true, true, false) => profile_name!("base32hex-mixed", Padding),
            (true, false, false) => profile_name!("base32hex-lower-mixed", Padding),
        }
    }

    /// Computes the exact encoded length.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::LengthOverflow`] if the result cannot be
    /// represented by `usize`. A successful calculation does not guarantee
    /// that a buffer of that size can be allocated.
    #[inline]
    pub const fn encoded_len(input_len: usize) -> Result<usize, EncodeError> {
        let len = if Padding::ENCODE_PADDED {
            backend::base32::encoded_len::<true>(input_len)
        } else {
            backend::base32::encoded_len::<false>(input_len)
        };
        match len {
            Some(len) => Ok(len),
            None => Err(EncodeError::LengthOverflow),
        }
    }

    /// Computes an upper bound for the decoded length.
    ///
    /// This calculation accepts every `usize` without overflow. It does not
    /// inspect or validate encoded input, including its length or padding.
    #[must_use]
    #[inline]
    pub const fn max_decoded_len(input_len: usize) -> usize {
        backend::base32::max_decoded_len(input_len)
    }

    /// Computes the exact decoded length if `input` can be decoded successfully.
    ///
    /// This does not fully validate the input. See
    /// [`Codec::decoded_len`](crate::Codec::decoded_len).
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::InvalidLength`] or [`DecodeError::InvalidPadding`]
    /// for structural errors detected while computing the output size.
    #[inline]
    pub fn decoded_len(input: impl AsRef<[u8]>) -> Result<usize, DecodeError> {
        codec::decoded_len::<Self>(input.as_ref())
    }

    /// Validates encoded input without allocating or writing decoded bytes.
    ///
    /// See [`Codec::check`](crate::Codec::check) for details.
    ///
    /// # Errors
    ///
    /// Returns the same [`DecodeError`] as decoding with sufficient output capacity.
    #[inline]
    pub fn check(input: impl AsRef<[u8]>) -> Result<(), DecodeError> {
        codec::check::<Self>(input.as_ref())
    }

    /// Encodes `input` into `output`, returning the written prefix.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::LengthOverflow`] if the encoded length overflows
    /// `usize`, or [`EncodeError::OutputTooSmall`] if `output` is shorter than
    /// the exact encoded length.
    #[inline]
    pub fn encode_into<'output>(
        input: &[u8],
        output: &'output mut [u8],
    ) -> Result<&'output mut str, EncodeError> {
        codec::encode_into::<Self>(input, output)
    }

    /// Encodes into possibly uninitialized storage and returns the initialized ASCII prefix.
    ///
    /// See [`Codec::encode_into_uninit`](crate::Codec::encode_into_uninit) for details.
    #[inline]
    pub fn encode_into_uninit<'output>(
        input: &[u8],
        output: &'output mut [MaybeUninit<u8>],
    ) -> Result<&'output mut str, EncodeError> {
        codec::encode_into_uninit::<Self>(input, output)
    }

    /// Decodes `input` into `output`, returning the written prefix.
    ///
    /// After any error, the contents of `output` are unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeSliceError::Decode`] if `input` is malformed, or
    /// [`DecodeSliceError::OutputTooSmall`] if `output` is shorter than the
    /// decoded length.
    #[inline]
    pub fn decode_into(
        input: impl AsRef<[u8]>,
        output: &mut [u8],
    ) -> Result<&mut [u8], DecodeSliceError> {
        codec::decode_into::<Self>(input.as_ref(), output)
    }

    /// Decodes into possibly uninitialized storage and returns the initialized prefix.
    ///
    /// See [`Codec::decode_into_uninit`](crate::Codec::decode_into_uninit) for details.
    #[inline]
    pub fn decode_into_uninit(
        input: impl AsRef<[u8]>,
        output: &mut [MaybeUninit<u8>],
    ) -> Result<&mut [u8], DecodeSliceError> {
        codec::decode_into_uninit::<Self>(input.as_ref(), output)
    }

    /// Decodes `buf` in place, returning the written prefix.
    ///
    /// After any error, the contents of `buf` are unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if `buf` is malformed.
    #[inline]
    pub fn decode_in_place(buf: &mut [u8]) -> Result<&mut [u8], DecodeError> {
        codec::decode_in_place::<Self>(buf)
    }

    /// Decodes `input` into a fixed-size byte array.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::InvalidLength`] if the decoded length does not
    /// equal `N`, or another [`DecodeError`] variant if `input` is malformed.
    #[inline]
    pub fn decode_array<const N: usize>(input: impl AsRef<[u8]>) -> Result<[u8; N], DecodeError> {
        codec::decode_array::<Self, N>(input.as_ref())
    }

    /// Encodes `input` into a newly allocated string.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[must_use]
    #[inline]
    pub fn encode(input: &[u8]) -> String {
        codec::encode::<Self>(input)
    }

    /// Encodes `input` and appends it to `buf`, returning the appended slice.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[inline]
    pub fn encode_append<'buf>(input: &[u8], buf: &'buf mut String) -> &'buf str {
        codec::encode_append::<Self>(input, buf)
    }

    /// Decodes `input` into a newly allocated byte vector.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if `input` is malformed.
    #[cfg(feature = "alloc")]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[inline]
    pub fn decode(input: impl AsRef<[u8]>) -> Result<Vec<u8>, DecodeError> {
        codec::decode::<Self>(input.as_ref())
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
    pub fn decode_append(input: impl AsRef<[u8]>, buf: &mut Vec<u8>) -> Result<usize, DecodeError> {
        codec::decode_append::<Self>(input.as_ref(), buf)
    }

    /// Returns a lazy display adapter for encoded `input`.
    #[must_use]
    #[inline]
    pub const fn display(input: &[u8]) -> EncodedDisplay<'_, Self> {
        EncodedDisplay::new(input)
    }

    /// Creates an incremental encoder for this profile.
    #[must_use]
    #[inline]
    pub const fn encoder() -> Encoder<Self> {
        Encoder::new()
    }

    /// Creates an incremental decoder for this profile.
    #[must_use]
    #[inline]
    pub const fn decoder() -> Decoder<Self> {
        Decoder::new()
    }

    /// Wraps `writer` in an encoding writer for this profile.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[must_use]
    #[inline]
    pub fn encoder_writer<W: io::Write>(writer: W) -> EncoderWriter<Self, W> {
        EncoderWriter::new(writer)
    }

    /// Wraps `reader` in a decoding reader for this profile.
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    #[must_use]
    #[inline]
    pub fn decoder_reader<R: io::Read>(reader: R) -> DecoderReader<Self, R> {
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
    pub fn serialize<T: ?Sized + AsRef<[u8]>, S: Serializer>(
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
    pub fn deserialize_array<'de, D: Deserializer<'de>, const N: usize>(
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
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        serde::deserialize::<Self, D>(deserializer)
    }
}

impl<Alphabet, Padding, Case, DecodeCase> Profile for Base32<Alphabet, Padding, Case, DecodeCase>
where
    Alphabet: sealed::Base32Alphabet,
    Padding: PaddingPolicy,
    Case: LetterCase,
    DecodeCase: DecodeCasePolicy,
{
    const NAME: &'static str = Self::name();
    const INPUT_QUANTUM: usize = 5;
    const ENCODED_QUANTUM: usize = 8;
    const DECODE_PADDING: DecodePadding = Padding::DECODE_PADDING;

    #[inline]
    fn encoded_len_impl(input_len: usize) -> Option<usize> {
        if Padding::ENCODE_PADDED {
            backend::base32::encoded_len::<true>(input_len)
        } else {
            backend::base32::encoded_len::<false>(input_len)
        }
    }

    #[inline]
    fn max_decoded_len_impl(input_len: usize) -> usize {
        backend::base32::max_decoded_len(input_len)
    }

    #[inline]
    fn decode_plan_impl(input: &[u8]) -> Result<DecodePlan, DecodeError> {
        if Padding::DECODE_PADDING.uses_padded_decoder(input) {
            backend::base32::decode_plan::<true>(input)
        } else {
            backend::base32::decode_plan::<false>(input)
        }
    }

    #[inline]
    fn check_with_plan_impl(input: &[u8], plan: DecodePlan) -> Result<(), DecodeError> {
        match (Alphabet::EXTENDED_HEX, Case::UPPERCASE, DecodeCase::STRICT) {
            (false, false, false) => {
                backend::base32::check_with_plan::<false, false, false>(input, plan)
            }
            (false, false, true) => {
                backend::base32::check_with_plan::<false, false, true>(input, plan)
            }
            (false, true, false) => {
                backend::base32::check_with_plan::<false, true, false>(input, plan)
            }
            (false, true, true) => {
                backend::base32::check_with_plan::<false, true, true>(input, plan)
            }
            (true, false, false) => {
                backend::base32::check_with_plan::<true, false, false>(input, plan)
            }
            (true, false, true) => {
                backend::base32::check_with_plan::<true, false, true>(input, plan)
            }
            (true, true, false) => {
                backend::base32::check_with_plan::<true, true, false>(input, plan)
            }
            (true, true, true) => backend::base32::check_with_plan::<true, true, true>(input, plan),
        }
    }

    #[inline]
    fn encode_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
        match (
            Alphabet::EXTENDED_HEX,
            Padding::ENCODE_PADDED,
            Case::UPPERCASE,
        ) {
            (false, false, false) => backend::base32::encode::<false, false, false>(input, output),
            (false, false, true) => backend::base32::encode::<false, false, true>(input, output),
            (false, true, false) => backend::base32::encode::<false, true, false>(input, output),
            (false, true, true) => backend::base32::encode::<false, true, true>(input, output),
            (true, false, false) => backend::base32::encode::<true, false, false>(input, output),
            (true, false, true) => backend::base32::encode::<true, false, true>(input, output),
            (true, true, false) => backend::base32::encode::<true, true, false>(input, output),
            (true, true, true) => backend::base32::encode::<true, true, true>(input, output),
        }
    }

    #[inline]
    fn encode_quantum_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
        match (Alphabet::EXTENDED_HEX, Case::UPPERCASE) {
            (false, false) => backend::base32::encode_quantum::<false, false>(input, output),
            (false, true) => backend::base32::encode_quantum::<false, true>(input, output),
            (true, false) => backend::base32::encode_quantum::<true, false>(input, output),
            (true, true) => backend::base32::encode_quantum::<true, true>(input, output),
        }
    }

    #[inline]
    fn decode_with_plan_impl(
        input: &[u8],
        plan: DecodePlan,
        output: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        match (Alphabet::EXTENDED_HEX, Case::UPPERCASE, DecodeCase::STRICT) {
            (false, false, false) => {
                backend::base32::decode_with_plan::<false, false, false>(input, plan, output)
            }
            (false, false, true) => {
                backend::base32::decode_with_plan::<false, false, true>(input, plan, output)
            }
            (false, true, false) => {
                backend::base32::decode_with_plan::<false, true, false>(input, plan, output)
            }
            (false, true, true) => {
                backend::base32::decode_with_plan::<false, true, true>(input, plan, output)
            }
            (true, false, false) => {
                backend::base32::decode_with_plan::<true, false, false>(input, plan, output)
            }
            (true, false, true) => {
                backend::base32::decode_with_plan::<true, false, true>(input, plan, output)
            }
            (true, true, false) => {
                backend::base32::decode_with_plan::<true, true, false>(input, plan, output)
            }
            (true, true, true) => {
                backend::base32::decode_with_plan::<true, true, true>(input, plan, output)
            }
        }
    }

    #[inline]
    fn decode_quantum_impl(
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        match (Alphabet::EXTENDED_HEX, Case::UPPERCASE, DecodeCase::STRICT) {
            (false, false, false) => {
                backend::base32::decode_quantum::<false, false, false>(input, output)
            }
            (false, false, true) => {
                backend::base32::decode_quantum::<false, false, true>(input, output)
            }
            (false, true, false) => {
                backend::base32::decode_quantum::<false, true, false>(input, output)
            }
            (false, true, true) => {
                backend::base32::decode_quantum::<false, true, true>(input, output)
            }
            (true, false, false) => {
                backend::base32::decode_quantum::<true, false, false>(input, output)
            }
            (true, false, true) => {
                backend::base32::decode_quantum::<true, false, true>(input, output)
            }
            (true, true, false) => {
                backend::base32::decode_quantum::<true, true, false>(input, output)
            }
            (true, true, true) => {
                backend::base32::decode_quantum::<true, true, true>(input, output)
            }
        }
    }

    #[inline]
    fn decode_in_place_impl(buf: &mut [u8]) -> Result<usize, DecodeError> {
        match (
            Alphabet::EXTENDED_HEX,
            Padding::DECODE_PADDING.uses_padded_decoder(buf),
            Case::UPPERCASE,
            DecodeCase::STRICT,
        ) {
            (false, false, false, false) => {
                backend::base32::decode_in_place::<false, false, false, false>(buf)
            }
            (false, false, false, true) => {
                backend::base32::decode_in_place::<false, false, false, true>(buf)
            }
            (false, false, true, false) => {
                backend::base32::decode_in_place::<false, false, true, false>(buf)
            }
            (false, false, true, true) => {
                backend::base32::decode_in_place::<false, false, true, true>(buf)
            }
            (false, true, false, false) => {
                backend::base32::decode_in_place::<false, true, false, false>(buf)
            }
            (false, true, false, true) => {
                backend::base32::decode_in_place::<false, true, false, true>(buf)
            }
            (false, true, true, false) => {
                backend::base32::decode_in_place::<false, true, true, false>(buf)
            }
            (false, true, true, true) => {
                backend::base32::decode_in_place::<false, true, true, true>(buf)
            }
            (true, false, false, false) => {
                backend::base32::decode_in_place::<true, false, false, false>(buf)
            }
            (true, false, false, true) => {
                backend::base32::decode_in_place::<true, false, false, true>(buf)
            }
            (true, false, true, false) => {
                backend::base32::decode_in_place::<true, false, true, false>(buf)
            }
            (true, false, true, true) => {
                backend::base32::decode_in_place::<true, false, true, true>(buf)
            }
            (true, true, false, false) => {
                backend::base32::decode_in_place::<true, true, false, false>(buf)
            }
            (true, true, false, true) => {
                backend::base32::decode_in_place::<true, true, false, true>(buf)
            }
            (true, true, true, false) => {
                backend::base32::decode_in_place::<true, true, true, false>(buf)
            }
            (true, true, true, true) => {
                backend::base32::decode_in_place::<true, true, true, true>(buf)
            }
        }
    }

    #[inline]
    fn decoded_padding_reduction(padding: usize) -> Option<usize> {
        match padding {
            0 => Some(0),
            1 => Some(1),
            3 => Some(2),
            4 => Some(3),
            6 => Some(4),
            _ => None,
        }
    }
}
