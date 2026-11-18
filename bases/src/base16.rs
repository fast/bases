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

//! Base16 (hexadecimal) codec profiles.

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
use crate::policy::sealed::LetterCase;
#[cfg(feature = "serde")]
use crate::serde;

/// A configurable RFC 4648 Base16 (hexadecimal) profile.
///
/// The type parameters select the encoded letter case (`Case`, defaulting to
/// [`policy::Lower`]) and decoding case policy (`DecodeCase`, defaulting to
/// [`policy::Mixed`]).
pub struct Base16<Case = policy::Lower, DecodeCase = policy::Mixed>(
    PhantomData<(Case, DecodeCase)>,
);

impl<Case, DecodeCase> Clone for Base16<Case, DecodeCase> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Case, DecodeCase> Copy for Base16<Case, DecodeCase> {}

impl<Case, DecodeCase> fmt::Debug for Base16<Case, DecodeCase> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Base16").finish()
    }
}

impl<Case, DecodeCase> Default for Base16<Case, DecodeCase> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Case, DecodeCase> PartialEq for Base16<Case, DecodeCase> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<Case, DecodeCase> Eq for Base16<Case, DecodeCase> {}

impl<Case, DecodeCase> Hash for Base16<Case, DecodeCase> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<Case, DecodeCase> Base16<Case, DecodeCase> {
    /// Creates a profile value.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Case, DecodeCase> Base16<Case, DecodeCase>
where
    Case: LetterCase,
    DecodeCase: DecodeCasePolicy,
{
    /// Returns the profile's stable human-readable name.
    #[must_use]
    #[inline]
    pub const fn name() -> &'static str {
        match (Case::UPPERCASE, DecodeCase::STRICT) {
            (false, false) => "base16-lower",
            (true, false) => "base16-upper",
            (false, true) => "base16-lower-strict",
            (true, true) => "base16-upper-strict",
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
        match backend::base16::encoded_len(input_len) {
            Some(len) => Ok(len),
            None => Err(EncodeError::LengthOverflow),
        }
    }

    /// Computes an upper bound for the decoded length.
    ///
    /// This calculation accepts every `usize` without overflow. It does not
    /// inspect or validate encoded input.
    #[must_use]
    #[inline]
    pub const fn max_decoded_len(input_len: usize) -> usize {
        backend::base16::max_decoded_len(input_len)
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

impl<Case, DecodeCase> Profile for Base16<Case, DecodeCase>
where
    Case: LetterCase,
    DecodeCase: DecodeCasePolicy,
{
    const NAME: &'static str = Self::name();
    const INPUT_QUANTUM: usize = 1;
    const ENCODED_QUANTUM: usize = 2;
    const ACCEPTS_PADDING: bool = false;

    #[inline]
    fn encoded_len_impl(input_len: usize) -> Option<usize> {
        backend::base16::encoded_len(input_len)
    }

    #[inline]
    fn max_decoded_len_impl(input_len: usize) -> usize {
        backend::base16::max_decoded_len(input_len)
    }

    #[inline]
    fn decode_plan_impl(input: &[u8]) -> Result<DecodePlan, DecodeError> {
        backend::base16::decode_plan(input)
    }

    #[inline(always)]
    fn encode_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
        if Case::UPPERCASE {
            backend::base16::encode::<true>(input, output)
        } else {
            backend::base16::encode::<false>(input, output)
        }
    }

    #[inline]
    fn encode_quantum_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
        if Case::UPPERCASE {
            backend::base16::encode_quantum::<true>(input, output)
        } else {
            backend::base16::encode_quantum::<false>(input, output)
        }
    }

    #[inline]
    fn decode_with_plan_impl(
        input: &[u8],
        plan: DecodePlan,
        output: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        match (Case::UPPERCASE, DecodeCase::STRICT) {
            (true, true) => backend::base16::decode_with_plan::<true, true>(input, plan, output),
            (true, false) => backend::base16::decode_with_plan::<true, false>(input, plan, output),
            (false, true) => backend::base16::decode_with_plan::<false, true>(input, plan, output),
            (false, false) => {
                backend::base16::decode_with_plan::<false, false>(input, plan, output)
            }
        }
    }

    #[inline]
    fn decode_quantum_impl(
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        match (Case::UPPERCASE, DecodeCase::STRICT) {
            (true, true) => backend::base16::decode_quantum::<true, true>(input, output),
            (true, false) => backend::base16::decode_quantum::<true, false>(input, output),
            (false, true) => backend::base16::decode_quantum::<false, true>(input, output),
            (false, false) => backend::base16::decode_quantum::<false, false>(input, output),
        }
    }

    #[inline]
    fn decode_in_place_impl(buf: &mut [u8]) -> Result<usize, DecodeError> {
        match (Case::UPPERCASE, DecodeCase::STRICT) {
            (true, true) => backend::base16::decode_in_place::<true, true>(buf),
            (true, false) => backend::base16::decode_in_place::<true, false>(buf),
            (false, true) => backend::base16::decode_in_place::<false, true>(buf),
            (false, false) => backend::base16::decode_in_place::<false, false>(buf),
        }
    }

    #[inline]
    fn decoded_padding_reduction(padding: usize) -> Option<usize> {
        (padding == 0).then_some(0)
    }
}
