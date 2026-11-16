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

//! Base64 codec profiles.

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
use crate::policy::sealed::PaddingPolicy;
#[cfg(feature = "serde")]
use crate::serde;

/// The RFC 4648 standard Base64 alphabet (`A-Z`, `a-z`, `0-9`, `+`, `/`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Standard;

/// The RFC 4648 Base64URL alphabet (`A-Z`, `a-z`, `0-9`, `-`, `_`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct UrlSafe;

impl sealed::Base64Alphabet for Standard {
    const URL_SAFE: bool = false;
}

impl sealed::Base64Alphabet for UrlSafe {
    const URL_SAFE: bool = true;
}

pub(crate) mod sealed {
    pub trait Base64Alphabet: 'static {
        const URL_SAFE: bool;
    }
}

/// A configurable RFC 4648 Base64 profile.
///
/// The type parameters select the alphabet (`Alphabet`, defaulting to
/// [`Standard`]) and padding policy (`Padding`, defaulting to
/// [`policy::Padded`]).
pub struct Base64<Alphabet = Standard, Padding = policy::Padded>(PhantomData<(Alphabet, Padding)>);

impl<Alphabet, Padding> Clone for Base64<Alphabet, Padding> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Alphabet, Padding> Copy for Base64<Alphabet, Padding> {}

impl<Alphabet, Padding> fmt::Debug for Base64<Alphabet, Padding> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Base64").finish()
    }
}

impl<Alphabet, Padding> Default for Base64<Alphabet, Padding> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Alphabet, Padding> PartialEq for Base64<Alphabet, Padding> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<Alphabet, Padding> Eq for Base64<Alphabet, Padding> {}

impl<Alphabet, Padding> Hash for Base64<Alphabet, Padding> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<Alphabet, Padding> Base64<Alphabet, Padding> {
    /// Creates a profile value.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Alphabet, Padding> Base64<Alphabet, Padding>
where
    Alphabet: sealed::Base64Alphabet,
    Padding: PaddingPolicy,
{
    /// Returns the profile's stable human-readable name.
    #[must_use]
    #[inline]
    pub const fn name() -> &'static str {
        match (Alphabet::URL_SAFE, Padding::PADDED) {
            (false, true) => "base64",
            (false, false) => "base64-unpadded",
            (true, true) => "base64url",
            (true, false) => "base64url-unpadded",
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
        let len = if Padding::PADDED {
            backend::base64::encoded_len::<true>(input_len)
        } else {
            backend::base64::encoded_len::<false>(input_len)
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
        backend::base64::max_decoded_len(input_len)
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

impl<Alphabet, Padding> Profile for Base64<Alphabet, Padding>
where
    Alphabet: sealed::Base64Alphabet,
    Padding: PaddingPolicy,
{
    const NAME: &'static str = Self::name();
    const INPUT_QUANTUM: usize = 3;
    const ENCODED_QUANTUM: usize = 4;
    const PADDED: bool = Padding::PADDED;

    #[inline]
    fn encoded_len_impl(input_len: usize) -> Option<usize> {
        if Padding::PADDED {
            backend::base64::encoded_len::<true>(input_len)
        } else {
            backend::base64::encoded_len::<false>(input_len)
        }
    }

    #[inline]
    fn max_decoded_len_impl(input_len: usize) -> usize {
        backend::base64::max_decoded_len(input_len)
    }

    #[inline]
    fn decode_plan_impl(input: &[u8]) -> Result<DecodePlan, DecodeError> {
        if Padding::PADDED {
            backend::base64::decode_plan::<true>(input)
        } else {
            backend::base64::decode_plan::<false>(input)
        }
    }

    #[cfg_attr(target_arch = "x86_64", inline(always))]
    #[cfg_attr(not(target_arch = "x86_64"), inline)]
    fn encode_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
        match (Alphabet::URL_SAFE, Padding::PADDED) {
            (false, false) => backend::base64::encode::<false, false>(input, output),
            (false, true) => backend::base64::encode::<false, true>(input, output),
            (true, false) => backend::base64::encode::<true, false>(input, output),
            (true, true) => backend::base64::encode::<true, true>(input, output),
        }
    }

    #[inline]
    fn encode_quantum_impl(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
        if Alphabet::URL_SAFE {
            backend::base64::encode_quantum::<true>(input, output)
        } else {
            backend::base64::encode_quantum::<false>(input, output)
        }
    }

    #[inline]
    fn decode_with_plan_impl(
        input: &[u8],
        plan: DecodePlan,
        output: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        if Alphabet::URL_SAFE {
            backend::base64::decode_with_plan::<true>(input, plan, output)
        } else {
            backend::base64::decode_with_plan::<false>(input, plan, output)
        }
    }

    #[inline]
    fn decode_quantum_impl(
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecodeError> {
        if Alphabet::URL_SAFE {
            backend::base64::decode_quantum::<true>(input, output)
        } else {
            backend::base64::decode_quantum::<false>(input, output)
        }
    }

    #[inline]
    fn decode_in_place_impl(buf: &mut [u8]) -> Result<usize, DecodeError> {
        match (Alphabet::URL_SAFE, Padding::PADDED) {
            (false, false) => backend::base64::decode_in_place::<false, false>(buf),
            (false, true) => backend::base64::decode_in_place::<false, true>(buf),
            (true, false) => backend::base64::decode_in_place::<true, false>(buf),
            (true, true) => backend::base64::decode_in_place::<true, true>(buf),
        }
    }

    #[inline]
    fn decoded_padding_reduction(padding: usize) -> Option<usize> {
        match padding {
            0 => Some(0),
            1 => Some(1),
            2 => Some(2),
            _ => None,
        }
    }
}
