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

use core::mem::MaybeUninit;

use bases::Base16Lower;
use bases::Base32;
use bases::Base64;
use bases::Codec;
use bases::DecodeError;
use bases::DecodeSliceError;
use bases::base32;
use bases::base64;
use bases::policy;
use data_encoding::Encoding;
use googletest::prelude::*;

use crate::payload;
use crate::streaming;

#[test]
fn validation_preserves_structural_and_symbol_error_precedence() {
    assert_eq!(
        Base64::check("!"),
        Err(DecodeError::InvalidLength { length: 1 }),
    );
    assert_eq!(
        Base64::check("!==="),
        Err(DecodeError::InvalidPadding { index: 1 }),
    );
    assert_eq!(
        Base32::check("!Y====="),
        Err(DecodeError::InvalidLength { length: 7 }),
    );
    assert_eq!(
        Base32::check("!Y====AA"),
        Err(DecodeError::InvalidByte {
            index: 0,
            byte: b'!'
        }),
    );
    assert_eq!(
        Base64::check("Zh==!!!!"),
        Err(DecodeError::InvalidPadding { index: 2 }),
    );
    assert_eq!(
        Base16Lower::check("=="),
        Err(DecodeError::InvalidByte {
            index: 0,
            byte: b'='
        }),
    );
}

#[test]
fn optional_padding_preserves_alphabet_and_trailing_bit_errors() {
    type Base64 = base64::Base64<base64::Standard, policy::UnpaddedOptional>;
    type Base32 = base32::Base32<base32::Standard, policy::PaddedOptional>;

    for wire in [b"Zm9vZh".as_slice(), b"Zm9vZh=="] {
        assert_error::<Base64>(wire, DecodeError::NonCanonical { index: 5 });
    }
    assert_error::<Base64>(
        b"Z ==",
        DecodeError::InvalidByte {
            index: 1,
            byte: b' ',
        },
    );
    assert_error::<Base64>(b"Zg==AAAA", DecodeError::InvalidPadding { index: 2 });
    for wire in [b"MZ".as_slice(), b"MZ======"] {
        assert_error::<Base32>(wire, DecodeError::NonCanonical { index: 1 });
    }
    assert_error::<Base32>(
        b"my======",
        DecodeError::InvalidByte {
            index: 0,
            byte: b'm',
        },
    );
    assert_error::<Base32>(
        b"MY======AAAAAAAA",
        DecodeError::InvalidPadding { index: 2 },
    );
}

pub fn differential<C: Codec>(reference: &Encoding) {
    let mut output = vec![0; 256];
    let mut in_place = Vec::new();
    let mut check = |wire: &[u8]| {
        // data-encoding accepts concatenated padded blocks; Bases accepts one message only.
        let interior_padding = wire
            .iter()
            .position(|&byte| byte == b'=')
            .is_some_and(|index| wire[index..].iter().any(|&byte| byte != b'='));
        let expected = if interior_padding {
            Err(())
        } else {
            reference.decode(wire).map_err(|_| ())
        };
        let decoded = C::decode_into(wire, &mut output);
        assert_eq!(
            C::check(wire).map_err(DecodeSliceError::from),
            decoded.as_ref().map(|_| ()).map_err(|&error| error),
            "validation wire: {wire:?}"
        );
        assert_eq!(
            decoded.map_err(|_| ()).as_deref(),
            expected.as_deref(),
            "wire: {wire:?}"
        );
        in_place.clear();
        in_place.extend_from_slice(wire);
        let actual = C::decode_in_place(&mut in_place).map_err(|_| ());
        assert_eq!(
            actual.as_deref(),
            expected.as_deref(),
            "in-place wire: {wire:?}"
        );
    };

    // Every possible byte in every lane, including case, '=', non-ASCII bytes and unused bits.
    for len in [1, 2, 3, 4, 5, 6, 31, 32, 33, 64, 65] {
        let mut wire = reference.encode(&payload(len)).into_bytes();
        for index in 0..wire.len() {
            let original = wire[index];
            for byte in 0..=255 {
                wire[index] = byte;
                check(&wire);
            }
            wire[index] = original;
        }
    }

    // Enumerate all length residues and padding shapes, including all-padding messages.
    let zero = reference.specification().symbols.as_bytes()[0];
    for len in 0..=40 {
        for padding in 0..=len {
            let mut wire = vec![zero; len];
            wire[len - padding..].fill(b'=');
            check(&wire);
        }
    }
}

pub fn errors<C: Codec>(reference: &Encoding) {
    for len in [
        8, 15, 16, 17, 19, 20, 21, 23, 24, 25, 64, 159, 160, 164, 1024,
    ] {
        let mut bad = reference.encode(&payload(len)).into_bytes();
        let last = bad.iter().rposition(|&byte| byte != b'=').unwrap();
        for first in [0, last / 2] {
            let original = bad[first];
            bad[first] = b'!';
            bad[last] = 0xff;
            let expected = DecodeError::InvalidByte {
                index: first,
                byte: b'!',
            };
            assert_eq!(C::check(&bad), Err(expected));
            let mut output = vec![0; C::max_decoded_len(bad.len())];
            assert_eq!(
                C::decode_into(&bad, &mut output),
                Err(DecodeSliceError::Decode(expected)),
            );
            bad[first] = original;
        }
    }

    let wire = reference.encode(&payload(65));
    for index in 0..wire.trim_end_matches('=').len() {
        let mut bad = wire.as_bytes().to_vec();
        bad[index] = b'!';
        assert_error::<C>(&bad, DecodeError::InvalidByte { index, byte: b'!' });
    }

    let spec = reference.specification();
    let bits = reference.bit_width();
    for len in 1..=5 {
        let wire = reference.encode(&payload(len));
        let data_len = wire.trim_end_matches('=').len();
        let unused = data_len * bits - len * 8;
        if unused == 0 {
            continue;
        }
        let mut bad = wire.into_bytes();
        let index = data_len - 1;
        let value = spec
            .symbols
            .as_bytes()
            .iter()
            .position(|&b| b == bad[index])
            .unwrap();
        for low_bits in 1..(1 << unused) {
            bad[index] = spec.symbols.as_bytes()[value | low_bits];
            assert_error::<C>(&bad, DecodeError::NonCanonical { index });
        }
    }

    if spec.padding.is_some() {
        let padded = reference.encode(b"f");
        let index = padded.find('=').unwrap();
        let joined = format!("{padded}{}", reference.encode(b"foobar"));
        assert_error::<C>(joined.as_bytes(), DecodeError::InvalidPadding { index });
    }
}

fn assert_error<C: Codec>(wire: &[u8], expected: DecodeError) {
    assert_eq!(C::check(wire), Err(expected));
    let mut output = vec![0; C::max_decoded_len(wire.len())];
    assert_that!(
        C::decode_into(wire, &mut output),
        err(eq(&DecodeSliceError::Decode(expected)))
    );
    let mut uninitialized = vec![MaybeUninit::<u8>::uninit(); output.len()];
    assert_that!(
        C::decode_into_uninit(wire, &mut uninitialized),
        err(eq(&DecodeSliceError::Decode(expected)))
    );
    assert_that!(C::decode(wire), err(eq(&expected)));
    assert_that!(C::decode_in_place(&mut wire.to_vec()), err(eq(&expected)));

    for reserve in [0, output.len() + 64] {
        let mut appended = vec![0, 0x80, 0xff];
        appended.reserve(reserve);
        assert_that!(C::decode_append(wire, &mut appended), err(eq(expected)));
        assert_eq!(appended, [0, 0x80, 0xff]);
        assert_eq!(C::decode_append("", &mut appended), Ok(0));
    }
    for chunk in [1, 2, 3, 7, 16, 31, wire.len()] {
        assert_eq!(
            streaming::decode::<C>(wire, chunk),
            Err(DecodeSliceError::Decode(expected))
        );
    }
}
