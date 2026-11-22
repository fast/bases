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

#[path = "contracts/buffers.rs"]
mod buffers;
#[path = "contracts/io.rs"]
mod io;
#[path = "contracts/lengths.rs"]
mod lengths;
#[path = "contracts/malformed.rs"]
mod malformed;
#[path = "contracts/serde.rs"]
mod serialization;
#[path = "contracts/streaming.rs"]
mod streaming;

use bases::Codec;
use bases::DecodeSliceError;
use bases::base16;
use bases::base32;
use bases::base64;
use bases::policy;
use data_encoding::Encoding;
use data_encoding::Specification;

// Keep the independent oracle's alphabet, padding and case rules explicit. In particular,
// a round trip through two implementations sharing the same bug is not a wire-format oracle.
fn reference(alphabet: &str, padded: bool, mixed: bool) -> Encoding {
    let mut spec = Specification::new();
    spec.symbols.push_str(alphabet);
    spec.padding = padded.then_some('=');
    if mixed {
        for symbol in alphabet.chars().filter(char::is_ascii_alphabetic) {
            spec.translate.from.push(if symbol.is_ascii_uppercase() {
                symbol.to_ascii_lowercase()
            } else {
                symbol.to_ascii_uppercase()
            });
            spec.translate.to.push(symbol);
        }
    }
    spec.encoding().unwrap()
}

fn payload(len: usize) -> Vec<u8> {
    let mut state = 0x243f_6a88_u32 ^ len as u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

#[test]
fn optional_padding_accepts_only_canonical_tail_lengths() {
    fn check_decode<C: Codec>(wire: &str, expected: Option<&[u8]>) {
        let mut out = vec![0; C::max_decoded_len(wire.len())];
        let decoded = C::decode_into(wire, &mut out);
        assert_eq!(
            C::validate(wire).map_err(DecodeSliceError::from),
            decoded.as_ref().map(|_| ()).map_err(|&error| error),
            "{wire}"
        );
        assert_eq!(decoded.ok().as_deref(), expected, "{wire}");
        let mut in_place = wire.as_bytes().to_vec();
        assert_eq!(
            C::decode_in_place(&mut in_place).ok().as_deref(),
            expected,
            "{wire}"
        );
        for chunk in 1..=wire.len().max(1) {
            assert_eq!(
                streaming::decode::<C>(wire.as_bytes(), chunk)
                    .ok()
                    .as_deref(),
                expected,
                "{wire}, chunk {chunk}"
            );
        }
    }

    fn check<Padded: Codec, Unpadded: Codec>(reference: &Encoding) {
        for len in 0..=10 {
            let raw = payload(len);
            let padded = reference.encode(&raw);
            let bare = padded.trim_end_matches('=');
            assert_eq!(Padded::encode(&raw), padded);
            assert_eq!(Unpadded::encode(&raw), bare);
            assert_eq!(streaming::encode::<Padded>(&raw, 1), padded);
            assert_eq!(streaming::encode::<Unpadded>(&raw, 1), bare);

            let required = padded.len() - bare.len();
            for count in 0..=required + 1 {
                let wire = format!("{bare}{}", "=".repeat(count));
                let expected = (count == 0 || count == required).then_some(raw.as_slice());
                check_decode::<Padded>(&wire, expected);
                check_decode::<Unpadded>(&wire, expected);
            }
        }
    }

    check::<
        base64::Base64<base64::Standard, policy::PaddedOptional>,
        base64::Base64<base64::Standard, policy::UnpaddedOptional>,
    >(&data_encoding::BASE64);
    check::<
        base64::Base64<base64::UrlSafe, policy::PaddedOptional>,
        base64::Base64<base64::UrlSafe, policy::UnpaddedOptional>,
    >(&data_encoding::BASE64URL);
    check::<
        base32::Base32<base32::Standard, policy::PaddedOptional>,
        base32::Base32<base32::Standard, policy::UnpaddedOptional>,
    >(&data_encoding::BASE32);
    check::<
        base32::Base32<base32::ExtendedHex, policy::PaddedOptional>,
        base32::Base32<base32::ExtendedHex, policy::UnpaddedOptional>,
    >(&data_encoding::BASE32HEX);
}

macro_rules! profile {
    ($name:ident, $codec:ty, $alphabet:expr, $padded:literal, $mixed:literal) => {
        mod $name {
            use super::*;
            type C = $codec;

            #[test]
            fn buffers_and_formatting() {
                buffers::check::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn uninitialized_output_buffers() {
                buffers::uninitialized::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn fixed_arrays() {
                buffers::arrays::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn exhaustive_short_inputs() {
                buffers::short_inputs::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn alphabet_padding_and_trailing_bits() {
                malformed::differential::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn errors_and_rollback() {
                malformed::errors::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn stream_boundaries_and_retries() {
                streaming::check::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn serde_representations() {
                serialization::check::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn length_boundaries() {
                lengths::check::<C>(&reference($alphabet, $padded, $mixed));
            }
            #[test]
            fn io_boundaries() {
                io::check::<C>(&reference($alphabet, $padded, $mixed));
            }
        }
    };
}

const HEX_LOWER: &str = "0123456789abcdef";
const HEX_UPPER: &str = "0123456789ABCDEF";
const BASE32_UPPER: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const BASE32_LOWER: &str = "abcdefghijklmnopqrstuvwxyz234567";
const BASE32HEX_UPPER: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
const BASE32HEX_LOWER: &str = "0123456789abcdefghijklmnopqrstuv";
const BASE64: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64URL: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

profile!(hex_lower, bases::Base16Lower, HEX_LOWER, false, true);
profile!(hex_upper, bases::Base16Upper, HEX_UPPER, false, true);
profile!(hex_lower_strict, base16::Base16<policy::Lower, policy::Strict>, HEX_LOWER, false, false);
profile!(hex_upper_strict, base16::Base16<policy::Upper, policy::Strict>, HEX_UPPER, false, false);
profile!(b32, bases::Base32, BASE32_UPPER, true, false);
profile!(
    b32_unpadded,
    bases::Base32Unpadded,
    BASE32_UPPER,
    false,
    false
);
profile!(b32hex, bases::Base32Hex, BASE32HEX_UPPER, true, false);
profile!(
    b32hex_unpadded,
    bases::Base32HexUnpadded,
    BASE32HEX_UPPER,
    false,
    false
);
profile!(b64, bases::Base64, BASE64, true, false);
profile!(b64_unpadded, bases::Base64Unpadded, BASE64, false, false);
profile!(b64url, bases::Base64Url, BASE64URL, true, false);
profile!(
    b64url_unpadded,
    bases::Base64UrlUnpadded,
    BASE64URL,
    false,
    false
);

profile!(b32_lower, base32::Base32<base32::Standard, policy::Padded, policy::Lower>, BASE32_LOWER, true, false);
profile!(b32_lower_unpadded, base32::Base32<base32::Standard, policy::Unpadded, policy::Lower>, BASE32_LOWER, false, false);
profile!(b32_mixed, base32::Base32<base32::Standard, policy::Padded, policy::Upper, policy::Mixed>, BASE32_UPPER, true, true);
profile!(b32_mixed_unpadded, base32::Base32<base32::Standard, policy::Unpadded, policy::Upper, policy::Mixed>, BASE32_UPPER, false, true);
profile!(b32_lower_mixed, base32::Base32<base32::Standard, policy::Padded, policy::Lower, policy::Mixed>, BASE32_LOWER, true, true);
profile!(b32_lower_mixed_unpadded, base32::Base32<base32::Standard, policy::Unpadded, policy::Lower, policy::Mixed>, BASE32_LOWER, false, true);
profile!(b32hex_lower, base32::Base32<base32::ExtendedHex, policy::Padded, policy::Lower>, BASE32HEX_LOWER, true, false);
profile!(b32hex_lower_unpadded, base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Lower>, BASE32HEX_LOWER, false, false);
profile!(b32hex_mixed, base32::Base32<base32::ExtendedHex, policy::Padded, policy::Upper, policy::Mixed>, BASE32HEX_UPPER, true, true);
profile!(b32hex_mixed_unpadded, base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Upper, policy::Mixed>, BASE32HEX_UPPER, false, true);
profile!(b32hex_lower_mixed, base32::Base32<base32::ExtendedHex, policy::Padded, policy::Lower, policy::Mixed>, BASE32HEX_LOWER, true, true);
profile!(b32hex_lower_mixed_unpadded, base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Lower, policy::Mixed>, BASE32HEX_LOWER, false, true);
