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

//! Fast, strict [RFC 4648](https://www.rfc-editor.org/rfc/rfc4648) Base16, Base32, Base32Hex, Base64, and Base64URL
//! codecs for Rust.
//!
//! # Quick start
//!
//! All core operations work in `#![no_std]` without an allocator using caller-supplied buffers or
//! fixed-size arrays:
//!
//! ```
//! use bases::Base64;
//!
//! let mut text = [0_u8; 8];
//! let encoded = Base64::encode_into(b"hello", &mut text).unwrap();
//! assert_eq!(encoded, "aGVsbG8=");
//!
//! let mut bytes = [0_u8; 5];
//! assert_eq!(
//!     Base64::decode_into(&*encoded, &mut bytes).unwrap(),
//!     b"hello"
//! );
//!
//! let exact: [u8; 5] = Base64::decode_array("aGVsbG8=").unwrap();
//! assert_eq!(&exact, b"hello");
//! ```
//!
//! # Profiles
//!
//! Ten crate-root profile aliases cover the standard RFC 4648 configurations:
//!
//! | Profile               | Encoding            | Output & padding    | Decode policy             |
//! |-----------------------|---------------------|---------------------|---------------------------|
//! | [`Base16Lower`]       | Base16 (hex)        | lowercase, unpadded | case-insensitive          |
//! | [`Base16Upper`]       | RFC 4648 Base16     | uppercase, unpadded | case-insensitive          |
//! | [`Base32`]            | RFC 4648 Base32     | uppercase, padded   | uppercase, exact padding  |
//! | [`Base32Unpadded`]    | RFC 4648 Base32     | uppercase, unpadded | uppercase, no padding     |
//! | [`Base32Hex`]         | RFC 4648 Base32Hex  | uppercase, padded   | uppercase, exact padding  |
//! | [`Base32HexUnpadded`] | RFC 4648 Base32Hex  | uppercase, unpadded | uppercase, no padding     |
//! | [`Base64`]            | RFC 4648 Base64     | padded              | standard alphabet, padded |
//! | [`Base64Unpadded`]    | RFC 4648 Base64     | unpadded            | standard alphabet, no pad |
//! | [`Base64Url`]         | RFC 4648 Base64URL  | padded              | URL-safe alphabet, padded |
//! | [`Base64UrlUnpadded`] | RFC 4648 Base64URL  | unpadded            | URL-safe alphabet, no pad |
//!
//! Use [`Base16Lower`] for the same lowercase output as `hex::encode`. For JWS Base64URL fields,
//! use [`Base64UrlUnpadded`]. [RFC 7515 §2](https://www.rfc-editor.org/rfc/rfc7515.html#section-2)
//!
//! Configurable zero-sized family types ([`base16::Base16`], [`base32::Base32`], and
//! [`base64::Base64`]) accept [`policy`] markers for additional combinations, such as strict
//! single-case Base16 (`base16::Base16<policy::Lower, policy::Strict>`). Every profile exposes
//! inherent methods directly without requiring trait imports, and implements the sealed [`Codec`]
//! trait for generic callers.
//!
//! For the NSEC3 Next Hashed Owner Name field, use
//! `base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Lower, policy::Mixed>`
//! for lowercase output and case-insensitive decoding. [RFC 5155 §3.3](https://www.rfc-editor.org/rfc/rfc5155.html#section-3.3)
//!
//! To decode both fully padded and unpadded Base32 or Base64 input, choose
//! [`policy::PaddedOptional`] for padded output or [`policy::UnpaddedOptional`] for unpadded
//! output.
//!
//! # API layers
//!
//! - **Sizing, validation, and caller buffers (`no_std`)**: `encoded_len`, `max_decoded_len`,
//!   `validate`, `encode_into`, `decode_into`, `decode_in_place`, and `decode_array` operate
//!   without allocation, with `const fn` support on concrete profile `new`, `name`, `encoded_len`,
//!   `max_decoded_len`, `display`, `encoder`, and `decoder`. [`Codec::encode_into_uninit`] and
//!   [`Codec::decode_into_uninit`] accept caller-provided `MaybeUninit<u8>` storage.
//! - **Owned and append helpers (`alloc`)**: `encode`, `encode_append`, `decode`, and transactional
//!   `decode_append` return or extend `String` and `Vec<u8>`.
//! - **Incremental streaming (`no_std` / `std`)**: [`Encoder`] and [`Decoder`] process arbitrarily
//!   chunked input while buffering at most one partial quantum, returning [`FinishError`] on
//!   finalization failures so callers can recover the stream or convert directly into
//!   [`EncodeError`] or [`DecodeSliceError`]; under `std`, `io::EncoderWriter` and
//!   `io::DecoderReader` adapt `std::io::Write` and `std::io::Read`.
//! - **Formatting (`no_std`)**: [`EncodedDisplay`] (via `display`) and [`HexDisplay`] (`LowerHex`
//!   `{:x}` and `UpperHex` `{:X}` only) write encoded text directly through `core::fmt` without
//!   allocating.
//! - **Serde (`serde`)**: `serialize`, `deserialize_array`, and (`alloc`) `deserialize` integrate
//!   with `#[serde(with = "...")]`.
//!
//! # Feature flags
//!
//! - `default = []`: `#![no_std]` with zero dependencies, exposing caller-buffer, in-place,
//!   fixed-array, streaming, and formatting APIs.
//! - `alloc`: enables `encode`, `encode_append`, `decode`, and `decode_append` for `String` and
//!   `Vec<u8>`.
//! - `std`: enables `alloc`, `io::EncoderWriter` and `io::DecoderReader`, and runtime x86/x86_64
//!   SIMD detection.
//! - `serde`: enables `serde_core` serialization and fixed-array deserialization in `no_std`, plus
//!   `Vec<u8>` deserialization when combined with `alloc`.
//!
//! # Strictness and security
//!
//! Decoders enforce strict RFC 4648 conformance: whitespace, bytes outside the selected alphabet,
//! misplaced or forbidden padding, trailing input after a padded final quantum, and non-zero unused
//! bits in the final symbol are rejected via [`DecodeError`] and [`DecodeSliceError`].
//!
//! Base encoding is not encryption. This crate does not guarantee constant-time execution and uses
//! data-dependent branches, table lookups, and private scalar or SIMD backends.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod base16;
pub mod base32;
pub mod base64;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod io;
pub mod policy;

pub(crate) mod backend;
mod codec;
mod display;
mod error;
#[cfg(feature = "serde")]
mod serde;
mod stream;

pub use self::codec::Codec;
pub use self::display::EncodedDisplay;
pub use self::display::HexDisplay;
pub use self::error::DecodeError;
pub use self::error::DecodeSliceError;
pub use self::error::EncodeError;
pub use self::error::FinishError;
pub use self::stream::Decoder;
pub use self::stream::Encoder;

/// A Base16 profile with lowercase encoded output and case-insensitive decoding.
pub type Base16Lower = base16::Base16<policy::Lower, policy::Mixed>;

/// An RFC 4648 Base16 profile with uppercase encoded output and case-insensitive decoding.
pub type Base16Upper = base16::Base16<policy::Upper, policy::Mixed>;

/// The padded RFC 4648 Base32 profile.
///
/// Encoding uses uppercase letters; decoding rejects lowercase input.
pub type Base32 = base32::Base32<base32::Standard, policy::Padded, policy::Upper, policy::Strict>;

/// The unpadded form of [`Base32`].
pub type Base32Unpadded =
    base32::Base32<base32::Standard, policy::Unpadded, policy::Upper, policy::Strict>;

/// The padded RFC 4648 Base32 Extended Hex profile.
///
/// Encoding uses uppercase letters; decoding rejects lowercase input.
pub type Base32Hex =
    base32::Base32<base32::ExtendedHex, policy::Padded, policy::Upper, policy::Strict>;

/// The unpadded form of [`Base32Hex`]. Decoding requires uppercase input.
pub type Base32HexUnpadded =
    base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Upper, policy::Strict>;

/// The padded RFC 4648 Base64 profile.
pub type Base64 = base64::Base64<base64::Standard, policy::Padded>;

/// The unpadded RFC 4648 Base64 profile.
pub type Base64Unpadded = base64::Base64<base64::Standard, policy::Unpadded>;

/// The padded RFC 4648 Base64URL profile.
pub type Base64Url = base64::Base64<base64::UrlSafe, policy::Padded>;

/// The unpadded RFC 4648 Base64URL profile.
pub type Base64UrlUnpadded = base64::Base64<base64::UrlSafe, policy::Unpadded>;

#[cfg(test)]
mod tests {
    use core::fmt;
    use core::fmt::Write;

    use crate::Base16Lower;
    use crate::Base16Upper;
    use crate::Base32;
    use crate::Base32Hex;
    use crate::Base32HexUnpadded;
    use crate::Base32Unpadded;
    use crate::Base64;
    use crate::Base64Unpadded;
    use crate::Base64Url;
    use crate::Base64UrlUnpadded;
    use crate::Codec;
    use crate::DecodeError;
    use crate::DecodeSliceError;
    use crate::HexDisplay;
    use crate::base16::Base16;
    use crate::policy;

    const _: usize = match Base64::encoded_len(5) {
        Ok(len) => len,
        Err(_) => panic!("encoded length overflow"),
    };
    const _: usize = Base64::max_decoded_len(8);
    const _: &str = Base16Lower::name();
    const _: Base64 = Base64::new();

    #[test]
    fn decode_array_and_in_place_no_std() {
        let arr: [u8; 4] = Base16Lower::decode_array("DeAdBeEf").unwrap();
        assert_eq!(arr, [0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(
            Base16Lower::decode_array::<3>("deadbeef"),
            Err(DecodeError::InvalidLength { length: 8 })
        );

        let mut buf = *b"aGVsbG8=";
        let decoded = Base64::decode_in_place(&mut buf).unwrap();
        assert_eq!(decoded, b"hello");

        let mut hex_buf = *b"deadbeef";
        assert_eq!(
            Base16::<policy::Lower, policy::Strict>::decode_in_place(&mut hex_buf).unwrap(),
            b"\xde\xad\xbe\xef"
        );
        let mut bad_strict = *b"deAd";
        assert_eq!(
            Base16::<policy::Lower, policy::Strict>::decode_in_place(&mut bad_strict),
            Err(DecodeError::InvalidByte {
                index: 2,
                byte: b'A'
            })
        );
    }

    #[test]
    fn decode_into_reports_slice_errors() {
        let mut small = [0_u8; 2];
        assert_eq!(
            Base64::decode_into("Zm9v", &mut small),
            Err(DecodeSliceError::OutputTooSmall {
                required: 3,
                available: 2,
            })
        );
        let mut output = [0_u8; 3];
        assert_eq!(
            Base64::decode_into("Zm!v", &mut output),
            Err(DecodeSliceError::Decode(DecodeError::InvalidByte {
                index: 2,
                byte: b'!'
            }))
        );
    }

    #[test]
    fn hex_display_formats_lower_and_upper_only() {
        struct Buf([u8; 16], usize);
        impl Write for Buf {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.0[self.1..self.1 + s.len()].copy_from_slice(s.as_bytes());
                self.1 += s.len();
                Ok(())
            }
        }

        let mut out = Buf([0; 16], 0);
        write!(&mut out, "{:x}", HexDisplay(b"\xde\xad")).unwrap();
        assert_eq!(&out.0[..out.1], b"dead");

        out.1 = 0;
        write!(&mut out, "{:X}", HexDisplay(b"\xde\xad")).unwrap();
        assert_eq!(&out.0[..out.1], b"DEAD");
    }

    #[test]
    fn all_canonical_aliases_implement_codec() {
        fn check_name<C: Codec>(expected: &str) {
            assert_eq!(C::name(), expected);
        }

        check_name::<Base16Lower>("base16-lower");
        check_name::<Base16Upper>("base16-upper");
        check_name::<Base32>("base32");
        check_name::<Base32Unpadded>("base32-unpadded");
        check_name::<Base32Hex>("base32hex");
        check_name::<Base32HexUnpadded>("base32hex-unpadded");
        check_name::<Base64>("base64");
        check_name::<Base64Unpadded>("base64-unpadded");
        check_name::<Base64Url>("base64url");
        check_name::<Base64UrlUnpadded>("base64url-unpadded");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn append_methods_and_transactional_decode_append() {
        use alloc::string::String;
        use alloc::vec;

        let mut text = String::from("prefix:");
        let appended = Base64::encode_append(b"hello", &mut text);
        assert_eq!(appended, "aGVsbG8=");
        assert_eq!(text, "prefix:aGVsbG8=");

        let big = vec![0x5a_u8; 1000];
        let mut text2 = String::from("head:");
        let appended2 = Base32::encode_append(&big, &mut text2);
        assert_eq!(appended2, Base32::encode(&big));

        let mut bytes = vec![1, 2, 3];
        let count = Base64::decode_append("aGVsbG8=", &mut bytes).unwrap();
        assert_eq!(count, 5);
        assert_eq!(bytes, b"\x01\x02\x03hello");

        assert_eq!(
            Base64::decode_append("Zm!v", &mut bytes),
            Err(DecodeError::InvalidByte {
                index: 2,
                byte: b'!'
            })
        );
        // Must remain completely unchanged after error.
        assert_eq!(bytes, b"\x01\x02\x03hello");
    }
}
