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

use bases::Codec;
use bases::DecodeError;
use bases::DecodeSliceError;
use bases::EncodeError;
use data_encoding::Encoding;
use googletest::prelude::*;

use crate::payload;

pub fn check<C: Codec>(reference: &Encoding) {
    for len in (0..=1025).chain([4096]) {
        let raw = payload(len);
        let expected = reference.encode(&raw);
        assert_eq!(C::encoded_len(len).unwrap(), expected.len());
        assert_that!(C::encode(&raw), eq(&expected));
        assert_that!(C::decode(&expected), ok(eq(&raw)));
        assert_that!(format!("{:>20.3}", C::display(&raw)), eq(&expected));

        // Both ends are guarded, including deliberately unaligned SIMD input/output.
        for offset in [0, 1, 7, 15, 31] {
            let mut source = vec![0x5a; offset];
            source.extend_from_slice(&raw);
            let mut encoded = vec![0xa5; offset + expected.len() + 32];
            assert_eq!(
                C::encode_into(
                    &source[offset..],
                    &mut encoded[offset..offset + expected.len()]
                )
                .unwrap(),
                expected.as_str(),
            );
            assert_that!(&encoded[..offset], each(eq(&0xa5)));
            assert_that!(&encoded[offset + expected.len()..], each(eq(&0xa5)));
            encoded.fill(0xa5);
            assert_eq!(
                C::encode_into(&source[offset..], &mut encoded[offset..]).unwrap(),
                expected.as_str()
            );
            assert_that!(&encoded[..offset], each(eq(&0xa5)));
            assert_that!(&encoded[offset + expected.len()..], each(eq(&0xa5)));

            for capacity in [len, C::max_decoded_len(expected.len()), len + 32] {
                let mut decoded = vec![0xa5; offset + capacity + 32];
                assert_eq!(
                    C::decode_into(
                        &encoded[offset..offset + expected.len()],
                        &mut decoded[offset..offset + capacity]
                    )
                    .unwrap(),
                    raw,
                );
                assert_that!(&decoded[..offset], each(eq(&0xa5)));
                assert_that!(&decoded[offset + len..], each(eq(&0xa5)));
            }
            assert_eq!(
                C::decode_in_place(&mut encoded[offset..offset + expected.len()]).unwrap(),
                raw,
            );
            assert_that!(&encoded[..offset], each(eq(&0xa5)));
            assert_that!(&encoded[offset + expected.len()..], each(eq(&0xa5)));
        }

        if !raw.is_empty() {
            for capacity in [0, len - 1] {
                assert_that!(
                    C::decode_into(&expected, &mut vec![0; capacity]),
                    err(eq(&DecodeSliceError::OutputTooSmall {
                        required: len,
                        available: capacity
                    }))
                );
            }
            assert_that!(
                C::encode_into(&raw, &mut vec![0; expected.len() - 1]),
                err(eq(&EncodeError::OutputTooSmall {
                    required: expected.len(),
                    available: expected.len() - 1
                }))
            );
        }

        // Exercise growth as well as spare capacity, with a non-ASCII string prefix.
        for reserve in [0, expected.len() + 64] {
            let mut text = String::from("前缀:");
            text.reserve(reserve);
            assert_eq!(C::encode_append(&raw, &mut text), expected);
            assert_eq!(text, format!("前缀:{expected}"));
            let mut decoded = vec![0, 0xff, 0x80];
            decoded.reserve(reserve);
            assert_eq!(C::decode_append(&expected, &mut decoded).unwrap(), len);
            assert_eq!(&decoded[..3], [0, 0xff, 0x80]);
            assert_eq!(&decoded[3..], raw);
        }
    }
}

pub fn short_inputs<C: Codec>(reference: &Encoding) {
    // Exhaust all one- and two-byte messages, including the partial-quantum packing rules.
    for byte in 0..=255 {
        let wire = reference.encode(&[byte]);
        assert_eq!(C::encode(&[byte]), wire);
        assert_eq!(C::decode_array::<1>(&wire).unwrap(), [byte]);
    }
    for word in 0..=u16::MAX {
        let raw = word.to_be_bytes();
        let wire = reference.encode(&raw);
        assert_eq!(C::encode(&raw), wire);
        assert_eq!(C::decode_array::<2>(&wire).unwrap(), raw);
    }
}

pub fn arrays<C: Codec>(reference: &Encoding) {
    fn check<C: Codec, const N: usize>(reference: &Encoding) {
        let raw = payload(N);
        let wire = reference.encode(&raw);
        assert_eq!(C::decode_array::<N>(&wire).unwrap().as_slice(), raw);
        for len in [N.saturating_sub(1), N + 1] {
            if len == N {
                continue;
            }
            let wrong = reference.encode(&payload(len));
            assert_eq!(
                C::decode_array::<N>(&wrong),
                Err(DecodeError::InvalidLength {
                    length: wrong.len()
                })
            );
        }
        if N != 0 {
            let mut bad = wire.into_bytes();
            bad[0] = b'!';
            assert_eq!(
                C::decode_array::<N>(&bad),
                Err(DecodeError::InvalidByte {
                    index: 0,
                    byte: b'!'
                })
            );
        }
    }
    check::<C, 0>(reference);
    check::<C, 1>(reference);
    check::<C, 2>(reference);
    check::<C, 3>(reference);
    check::<C, 4>(reference);
    check::<C, 5>(reference);
    check::<C, 15>(reference);
    check::<C, 16>(reference);
    check::<C, 31>(reference);
    check::<C, 32>(reference);
    check::<C, 63>(reference);
    check::<C, 64>(reference);
    check::<C, 65>(reference);
    check::<C, 1024>(reference);
}
