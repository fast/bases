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

use bases::Base16Lower;
use bases::Base16Upper;
use bases::Base32;
use bases::Base32HexUnpadded;
use bases::Base32Unpadded;
use bases::Base64;
use bases::Base64Unpadded;
use bases::Base64Url;
use bases::DecodeError;
use bases::DecodeSliceError;
use bases::EncodeError;

#[test]
fn rejects_invalid_lengths() {
    assert_decode_error::<Base16Upper>(b"0", DecodeError::InvalidLength { length: 1 });
    assert_decode_error::<Base16Lower>(b"0", DecodeError::InvalidLength { length: 1 });
    assert_decode_error::<Base32>(b"MY", DecodeError::InvalidLength { length: 2 });
    assert_decode_error::<Base32Unpadded>(b"M", DecodeError::InvalidLength { length: 1 });
    assert_decode_error::<Base64>(b"Zg", DecodeError::InvalidLength { length: 2 });
    assert_decode_error::<Base64Unpadded>(b"Z", DecodeError::InvalidLength { length: 1 });
}

#[test]
fn rejects_bytes_outside_the_alphabet() {
    let field = vec![b'Z', 0xff, b'=', b'='];
    assert_eq!(
        Base64::decode(field),
        Err(DecodeError::InvalidByte {
            index: 1,
            byte: 0xff
        })
    );
    assert_decode_error::<Base16Upper>(
        b"0g",
        DecodeError::InvalidByte {
            index: 1,
            byte: b'g',
        },
    );
    assert_decode_error::<Base16Lower>(
        b"0G",
        DecodeError::InvalidByte {
            index: 1,
            byte: b'G',
        },
    );
    assert_decode_error::<Base32Unpadded>(
        b"M!",
        DecodeError::InvalidByte {
            index: 1,
            byte: b'!',
        },
    );
    assert_decode_error::<Base64Unpadded>(
        b"Z!",
        DecodeError::InvalidByte {
            index: 1,
            byte: b'!',
        },
    );
    assert_decode_error::<Base64Url>(
        b"+/8=",
        DecodeError::InvalidByte {
            index: 0,
            byte: b'+',
        },
    );
}

#[test]
fn strict_profiles_reject_whitespace_nul_and_visual_aliases() {
    for &byte in b"\r\n \t\0" {
        let input = [b'Z', byte, b'9', b'v'];
        assert_decode_error::<Base64>(&input, DecodeError::InvalidByte { index: 1, byte });
    }

    assert_decode_error::<Base32Unpadded>(
        b"M0",
        DecodeError::InvalidByte {
            index: 1,
            byte: b'0',
        },
    );
    assert_decode_error::<Base32Unpadded>(
        b"my",
        DecodeError::InvalidByte {
            index: 0,
            byte: b'm',
        },
    );
    assert_decode_error::<Base32HexUnpadded>(
        b"co",
        DecodeError::InvalidByte {
            index: 0,
            byte: b'c',
        },
    );
}

#[test]
fn rejects_misplaced_or_forbidden_padding() {
    assert_decode_error::<Base64>(b"Zg==AAAA", DecodeError::InvalidPadding { index: 2 });
    assert_decode_error::<Base64Unpadded>(b"Zg==", DecodeError::InvalidPadding { index: 2 });
    assert_decode_error::<Base64>(b"Z===", DecodeError::InvalidPadding { index: 1 });
    assert_decode_error::<Base32>(
        b"MY======AAAAAAAA",
        DecodeError::InvalidPadding { index: 2 },
    );
    assert_decode_error::<Base32Unpadded>(b"MY======", DecodeError::InvalidPadding { index: 2 });
    assert_decode_error::<Base32>(b"M=======", DecodeError::InvalidPadding { index: 1 });
}

#[test]
fn rejects_non_canonical_trailing_bits() {
    assert_decode_error::<Base64>(b"Zh==", DecodeError::NonCanonical { index: 1 });
    assert_decode_error::<Base64Unpadded>(b"Zm9", DecodeError::NonCanonical { index: 2 });
    assert_decode_error::<Base32>(b"MZ======", DecodeError::NonCanonical { index: 1 });
    assert_decode_error::<Base32Unpadded>(b"MZXW7", DecodeError::NonCanonical { index: 4 });
}

#[test]
fn reports_small_output_buffers_without_panicking() {
    let mut encoded = [0_u8; 1];
    assert_eq!(
        Base64::encode_into(b"f", &mut encoded),
        Err(EncodeError::OutputTooSmall {
            required: 4,
            available: 1,
        })
    );

    let mut decoded = [0_u8; 2];
    assert_eq!(
        Base64::decode_into(b"Zm9v", &mut decoded),
        Err(DecodeSliceError::OutputTooSmall {
            required: 3,
            available: 2,
        })
    );

    // Length and trailing-padding validation precede OutputTooSmall; byte validation follows it.
    assert_eq!(
        Base64::decode_into(b"Zg", &mut []),
        Err(DecodeSliceError::Decode(DecodeError::InvalidLength {
            length: 2
        }))
    );
    assert_eq!(
        Base64::decode_into(b"A===", &mut []),
        Err(DecodeSliceError::Decode(DecodeError::InvalidPadding {
            index: 1
        }))
    );
    assert_eq!(
        Base32::decode_into(b"AA======", &mut [0_u8; 0]),
        Err(DecodeSliceError::OutputTooSmall {
            required: 1,
            available: 0,
        })
    );
    assert_eq!(
        Base64::decode_into(b"Zm!v", &mut decoded),
        Err(DecodeSliceError::OutputTooSmall {
            required: 3,
            available: 2,
        })
    );
}

fn assert_decode_error<C: bases::Codec>(input: &[u8], expected: DecodeError) {
    let mut output = [0_u8; 64];
    assert_eq!(
        C::decode_into(input, &mut output),
        Err(DecodeSliceError::Decode(expected)),
        "decoding {:?} with {}",
        input,
        C::name()
    );
    assert_eq!(
        C::decode(input),
        Err(expected),
        "owned decoding {:?} with {}",
        input,
        C::name()
    );
}
