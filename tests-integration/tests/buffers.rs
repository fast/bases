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

use std::array;

use bases::Base16Lower;
use bases::Base16Upper;
use bases::Base32;
use bases::Base32Hex;
use bases::Base32HexUnpadded;
use bases::Base32Unpadded;
use bases::Base64;
use bases::Base64Unpadded;
use bases::Base64Url;
use bases::Base64UrlUnpadded;
use bases::Codec;
use bases::DecodeError;
use bases::DecodeSliceError;
use bases::base16;
use bases::base32;
use bases::policy;

type Base16LowerStrict = base16::Base16<policy::Lower, policy::Strict>;
type Base16UpperStrict = base16::Base16<policy::Upper, policy::Strict>;
type Base32LowerStrict =
    base32::Base32<base32::Standard, policy::Padded, policy::Lower, policy::Strict>;
type Base32HexLowerMixedUnpadded =
    base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Lower, policy::Mixed>;
type Base32UpperMixedUnpadded =
    base32::Base32<base32::Standard, policy::Unpadded, policy::Upper, policy::Mixed>;

#[test]
fn decode_array_handles_exact_mismatched_and_malformed_inputs() {
    fn check_exact<C: Codec, const N: usize>() {
        let input: [u8; N] = array::from_fn(|idx| (idx * 31 + 5) as u8);
        let encoded = C::encode(&input);
        let decoded: [u8; N] = C::decode_array(&encoded).unwrap();
        assert_eq!(decoded, input, "profile {}", C::name());
    }

    fn check_family<C: Codec>() {
        check_exact::<C, 0>();
        check_exact::<C, 1>();
        check_exact::<C, 2>();
        check_exact::<C, 3>();
        check_exact::<C, 4>();
        check_exact::<C, 5>();
        check_exact::<C, 8>();
        check_exact::<C, 16>();
        check_exact::<C, 32>();
        check_exact::<C, 64>();
    }

    check_family::<Base16Lower>();
    check_family::<Base16Upper>();
    check_family::<Base32>();
    check_family::<Base32Unpadded>();
    check_family::<Base32Hex>();
    check_family::<Base64>();
    check_family::<Base64Unpadded>();
    check_family::<Base64UrlUnpadded>();

    // Inherent decode_array without Codec import for concrete profiles.
    let digest: [u8; 4] = Base16Lower::decode_array("DeAdBeEf").unwrap();
    assert_eq!(digest, [0xde, 0xad, 0xbe, 0xef]);

    // Decoded length != N must report InvalidLength with the encoded input length.
    let too_long = Base64::encode(b"hello");
    assert_eq!(
        Base64::decode_array::<4>(&too_long),
        Err(DecodeError::InvalidLength {
            length: too_long.len()
        })
    );
    let too_short = Base64::encode(b"hi");
    assert_eq!(
        Base64::decode_array::<4>(&too_short),
        Err(DecodeError::InvalidLength {
            length: too_short.len()
        })
    );
    assert_eq!(
        Base16Lower::decode_array::<1>(""),
        Err(DecodeError::InvalidLength { length: 0 })
    );
    assert_eq!(
        Base16Lower::decode_array::<0>("00"),
        Err(DecodeError::InvalidLength { length: 2 })
    );

    // Malformed bytes, padding, and non-canonical tails must still report their cause.
    assert_eq!(
        Base64::decode_array::<3>("Zm!v"),
        Err(DecodeError::InvalidByte {
            index: 2,
            byte: b'!'
        })
    );
    assert_eq!(
        Base64Unpadded::decode_array::<3>("Zg=="),
        Err(DecodeError::InvalidPadding { index: 2 })
    );
    assert_eq!(
        Base64::decode_array::<6>("Zg==AAAA"),
        Err(DecodeError::InvalidPadding { index: 2 })
    );
    assert_eq!(
        Base64::decode_array::<1>("Zh=="),
        Err(DecodeError::NonCanonical { index: 1 })
    );
    assert_eq!(
        Base64::decode_array::<1>("Zg"),
        Err(DecodeError::InvalidLength { length: 2 })
    );
}

#[test]
fn decode_in_place_matches_roundtrips_and_rejects_simd_boundary_errors() {
    fn check_roundtrip<C: Codec>(input: &[u8]) {
        let encoded = C::encode(input);
        let mut buf = encoded.into_bytes();
        let decoded = C::decode_in_place(&mut buf).unwrap();
        assert_eq!(
            decoded,
            input,
            "decode_in_place {} at len {}",
            C::name(),
            input.len()
        );
    }

    for len in 0..=260 {
        let input: Vec<u8> = (0..len).map(|i| (i * 41 + 9) as u8).collect();
        check_roundtrip::<Base16Lower>(&input);
        check_roundtrip::<Base16Upper>(&input);
        check_roundtrip::<Base16LowerStrict>(&input);
        check_roundtrip::<Base16UpperStrict>(&input);
        check_roundtrip::<Base32>(&input);
        check_roundtrip::<Base32Unpadded>(&input);
        check_roundtrip::<Base32Hex>(&input);
        check_roundtrip::<Base32HexUnpadded>(&input);
        check_roundtrip::<Base32LowerStrict>(&input);
        check_roundtrip::<Base32HexLowerMixedUnpadded>(&input);
        check_roundtrip::<Base32UpperMixedUnpadded>(&input);
        check_roundtrip::<Base64>(&input);
        check_roundtrip::<Base64Unpadded>(&input);
        check_roundtrip::<Base64Url>(&input);
        check_roundtrip::<Base64UrlUnpadded>(&input);
    }

    // Exercise mixed-case in-place decoding across SIMD boundaries.
    for len in [1, 16, 32, 64, 128, 256] {
        let input: Vec<u8> = (0..len).map(|i| (i * 19 + 3) as u8).collect();
        let mut hex_buf: Vec<u8> = Base16Lower::encode(&input)
            .bytes()
            .enumerate()
            .map(|(i, b)| {
                if i % 2 == 0 {
                    b.to_ascii_uppercase()
                } else {
                    b
                }
            })
            .collect();
        assert_eq!(Base16Upper::decode_in_place(&mut hex_buf).unwrap(), input);

        let mut b32_buf: Vec<u8> = Base32UpperMixedUnpadded::encode(&input)
            .bytes()
            .enumerate()
            .map(|(i, b)| {
                if i % 2 == 1 {
                    b.to_ascii_lowercase()
                } else {
                    b
                }
            })
            .collect();
        assert_eq!(
            Base32UpperMixedUnpadded::decode_in_place(&mut b32_buf).unwrap(),
            input
        );
    }

    // Probe invalid bytes at scalar and SIMD chunk boundaries for Base16, Base32, and Base64.
    let raw_260: Vec<u8> = (0..260).map(|i| (i * 7 + 1) as u8).collect();
    let base_offsets = [0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255];

    let hex_valid = Base16Lower::encode(&raw_260).into_bytes();
    for &idx in &base_offsets {
        if idx < hex_valid.len() {
            let mut bad = hex_valid.clone();
            bad[idx] = b'!';
            assert_eq!(
                Base16Lower::decode_in_place(&mut bad),
                Err(DecodeError::InvalidByte {
                    index: idx,
                    byte: b'!'
                })
            );

            // Strict vs Mixed case checks.
            if hex_valid[idx].is_ascii_lowercase() {
                let mut flipped = hex_valid.clone();
                let bad_byte = flipped[idx].to_ascii_uppercase();
                flipped[idx] = bad_byte;
                assert_eq!(
                    Base16LowerStrict::decode_in_place(&mut flipped),
                    Err(DecodeError::InvalidByte {
                        index: idx,
                        byte: bad_byte,
                    })
                );
            }
        }
    }

    let b32_valid = Base32::encode(&raw_260).into_bytes();
    for &idx in &base_offsets {
        if idx < b32_valid.len() {
            let mut bad = b32_valid.clone();
            bad[idx] = b'0';
            assert_eq!(
                Base32::decode_in_place(&mut bad),
                Err(DecodeError::InvalidByte {
                    index: idx,
                    byte: b'0'
                })
            );
            if b32_valid[idx].is_ascii_uppercase() {
                let mut lowered = b32_valid.clone();
                let bad_byte = lowered[idx].to_ascii_lowercase();
                lowered[idx] = bad_byte;
                assert_eq!(
                    Base32::decode_in_place(&mut lowered),
                    Err(DecodeError::InvalidByte {
                        index: idx,
                        byte: bad_byte,
                    })
                );
            }
        }
    }

    let b64_valid = Base64::encode(&raw_260).into_bytes();
    for &idx in &base_offsets {
        if idx < b64_valid.len() {
            let mut bad = b64_valid.clone();
            bad[idx] = b'-';
            assert_eq!(
                Base64::decode_in_place(&mut bad),
                Err(DecodeError::InvalidByte {
                    index: idx,
                    byte: b'-'
                })
            );
        }
    }

    // Interior padding, forbidden padding, non-canonical tail, and invalid lengths.
    let mut interior_pad = b"Zm9vZg==Zm9v".to_vec();
    assert_eq!(
        Base64::decode_in_place(&mut interior_pad),
        Err(DecodeError::InvalidPadding { index: 6 })
    );
    let mut unpadded_pad = b"Zg==".to_vec();
    assert_eq!(
        Base64Unpadded::decode_in_place(&mut unpadded_pad),
        Err(DecodeError::InvalidPadding { index: 2 })
    );
    let mut non_canonical = b"MZXW7".to_vec();
    assert_eq!(
        Base32Unpadded::decode_in_place(&mut non_canonical),
        Err(DecodeError::NonCanonical { index: 4 })
    );
    let mut odd_hex = b"abc".to_vec();
    assert_eq!(
        Base16Lower::decode_in_place(&mut odd_hex),
        Err(DecodeError::InvalidLength { length: 3 })
    );
}

#[test]
fn encode_append_and_transactional_decode_append() {
    fn check_append<C: Codec>(input: &[u8]) {
        let mut text = String::from("header:");
        let appended = C::encode_append(input, &mut text);
        let expected = C::encode(input);
        assert_eq!(appended, expected);
        assert_eq!(text, format!("header:{expected}"));

        let mut out = vec![0xaa, 0xbb, 0xcc];
        let written = C::decode_append(&expected, &mut out).unwrap();
        assert_eq!(written, input.len());
        assert_eq!(&out[..3], &[0xaa, 0xbb, 0xcc]);
        assert_eq!(&out[3..], input);
    }

    // Exercise lengths up to > 2x the 480-byte internal encode_append chunk size.
    for len in [0, 1, 2, 3, 5, 64, 479, 480, 481, 960, 1024] {
        let data: Vec<u8> = (0..len).map(|i| (i * 23 + 17) as u8).collect();
        check_append::<Base16Lower>(&data);
        check_append::<Base32>(&data);
        check_append::<Base64UrlUnpadded>(&data);
    }

    // Verify transactional rollback on decode_append leaves prefix completely untouched.
    let prefix = vec![0x01, 0x02, 0xfe, 0xff];
    let mut bytes = prefix.clone();

    let large_valid = Base64::encode(&[0x42_u8; 96]);
    let mut late_bad_byte = large_valid.clone().into_bytes();
    late_bad_byte[127] = b'!';
    assert_eq!(
        Base64::decode_append(&late_bad_byte, &mut bytes),
        Err(DecodeError::InvalidByte {
            index: 127,
            byte: b'!'
        })
    );
    assert_eq!(bytes, prefix);

    assert_eq!(
        Base64::decode_append("Zg", &mut bytes),
        Err(DecodeError::InvalidLength { length: 2 })
    );
    assert_eq!(bytes, prefix);

    assert_eq!(
        Base64Unpadded::decode_append("Zg==", &mut bytes),
        Err(DecodeError::InvalidPadding { index: 2 })
    );
    assert_eq!(bytes, prefix);

    assert_eq!(
        Base64::decode_append("Zh==", &mut bytes),
        Err(DecodeError::NonCanonical { index: 1 })
    );
    assert_eq!(bytes, prefix);

    // Subsequent valid decode_append succeeds after rolled-back failures.
    let written = Base64::decode_append("aGVsbG8=", &mut bytes).unwrap();
    assert_eq!(written, 5);
    assert_eq!(bytes, b"\x01\x02\xfe\xffhello");
}

#[test]
fn decode_into_and_reject_preserve_exact_output_bounds_and_error_offsets() {
    fn check<C: Codec>(raw: &[u8]) {
        let encoded = C::encode(raw);
        let mut exact = vec![0_u8; raw.len()];
        assert_eq!(C::decode_into(&encoded, &mut exact).unwrap(), raw);
        assert_eq!(exact, raw);

        let mut sentinel = vec![0xa5_u8; raw.len() + 32];
        assert_eq!(C::decode_into(&encoded, &mut sentinel).unwrap(), raw);
        assert_eq!(&sentinel[..raw.len()], raw);
        assert!(sentinel[raw.len()..].iter().all(|&b| b == 0xa5));
    }

    for raw_len in [10, 12, 15, 16, 20, 24, 25, 27, 28, 30, 40, 48, 60, 64] {
        let raw: Vec<u8> = (0..raw_len).map(|i| (i * 29 + 3) as u8).collect();
        check::<Base64>(&raw);
        check::<Base64UrlUnpadded>(&raw);
        check::<Base32>(&raw);
        check::<Base32Unpadded>(&raw);
    }

    // Specific worst-case offset inside 64B Base64 SIMD block (byte 63).
    let raw_64: Vec<u8> = (0..64).map(|i| (i * 11 + 7) as u8).collect();
    let mut bad = Base64::encode(&raw_64).into_bytes();
    bad[63] = b'!';
    let expected = DecodeError::InvalidByte {
        index: 63,
        byte: b'!',
    };
    assert_eq!(Base64::decode(&bad), Err(expected));
    let mut dst = vec![0x5a_u8; 64];
    assert_eq!(
        Base64::decode_into(&bad, &mut dst),
        Err(DecodeSliceError::Decode(expected))
    );
    let mut app = vec![0x01, 0x02];
    assert_eq!(Base64::decode_append(&bad, &mut app), Err(expected));
    assert_eq!(app, vec![0x01, 0x02]);
    assert_eq!(Base64::decode_in_place(&mut bad), Err(expected));
}
