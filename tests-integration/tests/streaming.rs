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

use bases::Base32;
use bases::Base64;
use bases::Base64Unpadded;
use bases::DecodeError;
use bases::DecodeSliceError;
use bases::Decoder;
use bases::EncodeError;
use bases::Encoder;

#[test]
fn small_output_errors_leave_stream_state_unchanged() {
    let mut encoder = Base64::encoder();
    let mut output = [0_u8; 4];
    assert_eq!(encoder.update(b"fo", &mut output).unwrap(), "");
    assert_eq!(
        encoder.update(b"o", &mut output[..3]),
        Err(EncodeError::OutputTooSmall {
            required: 4,
            available: 3,
        })
    );
    assert_eq!(encoder.update(b"o", &mut output).unwrap(), "Zm9v");
    assert_eq!(encoder.finish(&mut output).unwrap(), "");

    let mut decoder = Base64::decoder();
    let mut output = [0_u8; 3];
    assert_eq!(decoder.update(b"Zm", &mut output).unwrap(), b"");
    assert_eq!(
        decoder.update(b"9v", &mut output[..2]),
        Err(DecodeSliceError::OutputTooSmall {
            required: 3,
            available: 2,
        })
    );
    assert_eq!(decoder.update(b"9v", &mut output).unwrap(), b"foo");
    assert_eq!(decoder.finish(&mut output).unwrap(), b"");
}

#[test]
fn decoder_reports_offsets_from_the_start_of_the_stream() {
    let mut decoder = Base64Unpadded::decoder();
    let mut output = [0_u8; 3];
    assert_eq!(decoder.update(String::from("Zm"), &mut []).unwrap(), b"");
    assert_eq!(decoder.update("9v", &mut output).unwrap(), b"foo");
    assert_eq!(
        decoder.update(vec![0xff, b'm', b'9', b'v'], &mut output),
        Err(DecodeSliceError::Decode(DecodeError::InvalidByte {
            index: 4,
            byte: 0xff
        }))
    );
    assert_eq!(decoder.update(b"Y", &mut output).unwrap(), b"");
    assert_eq!(
        decoder.update(b"m!y", &mut output),
        Err(DecodeSliceError::Decode(DecodeError::InvalidByte {
            index: 6,
            byte: b'!',
        }))
    );

    assert_eq!(decoder.update(b"mFy", &mut output).unwrap(), b"bar");
    assert_eq!(decoder.finish(&mut output).unwrap(), b"");
}

#[test]
fn padded_messages_cannot_be_concatenated_at_any_split() {
    fn check<C: bases::Codec>(message: &str, expected: &[u8], next: &str) {
        let index = message.find('=').unwrap();
        let input = format!("{message}{next}");
        for split in 0..=message.len() {
            let mut decoder = C::decoder();
            let mut decoded = vec![0; expected.len()];
            let written = decoder.update(&input[..split], &mut decoded).unwrap().len();
            let mut scratch = [0; 32];
            // Once padding has been accepted, even an empty destination must report trailing input.
            let capacity = if split == message.len() {
                0
            } else {
                scratch.len()
            };
            assert_eq!(
                decoder.update(&input[split..], &mut scratch[..capacity]),
                Err(DecodeSliceError::Decode(DecodeError::InvalidPadding {
                    index
                })),
                "{} at split {split}: {input}",
                C::name(),
            );
            let tail = decoder
                .update(&message[split..], &mut decoded[written..])
                .unwrap();
            assert_eq!(written + tail.len(), expected.len());
            assert_eq!(decoded, expected);
            assert_eq!(decoder.finish_output_len(), Ok(0));
            assert_eq!(decoder.finish(&mut []).unwrap(), b"");
        }
    }

    for next in ["A", "AAAA", "Zg==", "AAAAA"] {
        check::<Base64>("Zm9vZg==", b"foof", next);
    }
    for next in ["A", "AAAAAAAA", "MY======", "AAAAAAAAA"] {
        check::<Base32>("MZXW6YTBMY======", b"foobaf", next);
    }
}

#[test]
fn finish_applies_the_profile_padding_policy() {
    let mut padded = Base64::decoder();
    let mut output = [0_u8; 1];
    assert_eq!(padded.update(b"Zg", &mut output).unwrap(), b"");
    assert_eq!(
        padded.finish_output_len(),
        Err(DecodeError::InvalidLength { length: 2 })
    );
    let (error, mut padded) = padded.finish(&mut output).unwrap_err().into_parts();
    assert_eq!(
        error,
        DecodeSliceError::Decode(DecodeError::InvalidLength { length: 2 })
    );
    assert_eq!(padded.update("==", &mut output).unwrap(), b"f");
    assert_eq!(padded.finish(&mut output).unwrap(), b"");

    let mut unpadded = Base64Unpadded::decoder();
    assert_eq!(unpadded.update(b"Zg", &mut output).unwrap(), b"");
    assert_eq!(unpadded.finish_output_len(), Ok(1));
    assert_eq!(unpadded.finish(&mut output).unwrap(), b"f");
}

#[test]
fn failed_decoder_finish_preserves_state_and_global_offsets() {
    let cases = [
        ("Z", DecodeError::InvalidLength { length: 5 }),
        ("Zh", DecodeError::NonCanonical { index: 5 }),
        (
            "Z!",
            DecodeError::InvalidByte {
                index: 5,
                byte: b'!',
            },
        ),
    ];

    for (tail, expected) in cases {
        let mut decoder = Base64Unpadded::decoder();
        let mut output = [0_u8; 3];
        assert_eq!(decoder.update("Zm9v", &mut output).unwrap(), b"foo");
        assert_eq!(decoder.update(tail, &mut output).unwrap(), b"");

        let (error, decoder) = decoder.finish(&mut output).unwrap_err().into_parts();
        assert_eq!(error, DecodeSliceError::Decode(expected));
        assert_eq!(
            decoder.finish(&mut output).unwrap_err().into_error(),
            DecodeSliceError::Decode(expected)
        );
    }
}

#[test]
fn finish_errors_can_be_propagated_without_recovery() {
    fn encode<C: bases::Codec>(
        encoder: Encoder<C>,
        output: &mut [u8],
    ) -> Result<&str, EncodeError> {
        Ok(encoder.finish(output)?)
    }

    fn decode<C: bases::Codec>(
        decoder: Decoder<C>,
        output: &mut [u8],
    ) -> Result<&[u8], DecodeSliceError> {
        Ok(decoder.finish(output)?)
    }

    let mut encoder = Base64::encoder();
    encoder.update(b"f", &mut []).unwrap();
    assert_eq!(
        encode(encoder, &mut []),
        Err(EncodeError::OutputTooSmall {
            required: 4,
            available: 0
        }),
    );

    let mut decoder = Base64Unpadded::decoder();
    decoder.update("Zg", &mut []).unwrap();
    assert_eq!(
        decode(decoder, &mut []),
        Err(DecodeSliceError::OutputTooSmall {
            required: 1,
            available: 0
        }),
    );
}
