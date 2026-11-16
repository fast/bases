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
use bases::DecodeSliceError;
use bases::EncodeError;
use data_encoding::Encoding;

use crate::payload;

pub fn check<C: Codec>(reference: &Encoding) {
    for len in (0..=40).chain([63, 64, 65, 255, 256, 257, 1025]) {
        let raw = payload(len);
        let wire = reference.encode(&raw);
        for chunk in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 257] {
            assert_eq!(
                encode::<C>(&raw, chunk),
                wire,
                "raw length {len}, chunk {chunk}"
            );
            assert_eq!(
                decode::<C>(wire.as_bytes(), chunk).unwrap(),
                raw,
                "raw length {len}, chunk {chunk}"
            );
        }

        // All two-way splits expose combinations of pending fragments that uniform chunks miss.
        if len <= 40 {
            for split in 0..=wire.len() {
                let mut decoder = C::decoder();
                let mut output = vec![0; C::max_decoded_len(wire.len())];
                let first = decoder.update(&wire[..split], &mut output).unwrap().len();
                let second = decoder
                    .update(&wire[split..], &mut output[first..])
                    .unwrap()
                    .len();
                let last = decoder.finish(&mut output[first + second..]).unwrap().len();
                assert_eq!(&output[..first + second + last], raw, "split {split}");
            }
        }
    }
    retry_invalid_update::<C>(reference);
}

fn encode<C: Codec>(raw: &[u8], chunk: usize) -> String {
    let mut encoder = C::encoder();
    let mut encoded = String::new();
    for part in raw.chunks(chunk) {
        let required = encoder.update_output_len(part.len()).unwrap();
        let mut output = vec![0; required];
        assert_eq!(encoder.update(&[], &mut []).unwrap(), "");
        if required != 0 {
            assert_eq!(
                encoder.update(part, &mut output[..required - 1]),
                Err(EncodeError::OutputTooSmall {
                    required,
                    available: required - 1
                })
            );
        }
        let mut cloned = encoder.clone();
        let mut expected = vec![0; required];
        assert_eq!(
            encoder.update(part, &mut output).unwrap(),
            cloned.update(part, &mut expected).unwrap()
        );
        encoded.push_str(std::str::from_utf8(&output).unwrap());
    }
    let required = encoder.finish_output_len();
    let mut output = vec![0; required];
    if required != 0 {
        let failure = encoder.finish(&mut output[..required - 1]).unwrap_err();
        assert_eq!(
            failure.error(),
            &EncodeError::OutputTooSmall {
                required,
                available: required - 1
            }
        );
        encoder = failure.into_inner();
    }
    encoded.push_str(encoder.finish(&mut output).unwrap());
    encoded
}

pub fn decode<C: Codec>(wire: &[u8], chunk: usize) -> Result<Vec<u8>, DecodeSliceError> {
    let mut decoder = C::decoder();
    let mut decoded = Vec::new();
    for part in wire.chunks(chunk) {
        assert_eq!(decoder.update([], &mut []).unwrap(), b"");
        let mut output = vec![0; decoder.max_update_output_len(part.len())];
        decoded.extend_from_slice(decoder.update(part, &mut output)?);
    }
    let required = decoder.finish_output_len()?;
    let mut output = vec![0; required];
    if required != 0 {
        let failure = decoder.finish(&mut output[..required - 1]).unwrap_err();
        assert_eq!(
            failure.error(),
            &DecodeSliceError::OutputTooSmall {
                required,
                available: required - 1
            }
        );
        decoder = failure.into_inner();
    }
    decoded.extend_from_slice(decoder.finish(&mut output)?);
    Ok(decoded)
}

fn retry_invalid_update<C: Codec>(reference: &Encoding) {
    let raw = payload(65);
    let wire = reference.encode(&raw);
    for split in 0..=16 {
        let mut decoder = C::decoder();
        let mut output = vec![0; raw.len()];
        let first = decoder.update(&wire[..split], &mut output).unwrap().len();
        let required = decoder
            .clone()
            .update(&wire[split..], &mut vec![0; raw.len()])
            .unwrap()
            .len();
        let mut bad = wire.as_bytes()[split..].to_vec();
        // The error follows successful bulk writes, which must not commit the stream state.
        bad[32] = b'!';
        assert!(matches!(
            decoder.update(&bad, &mut output[first..]),
            Err(DecodeSliceError::Decode(_))
        ));
        assert_eq!(
            decoder.update(&wire[split..], &mut output[first..first + required - 1]),
            Err(DecodeSliceError::OutputTooSmall {
                required,
                available: required - 1
            })
        );
        let second = decoder
            .update(&wire[split..], &mut output[first..])
            .unwrap()
            .len();
        let last = decoder.finish(&mut output[first + second..]).unwrap().len();
        assert_eq!(&output[..first + second + last], raw);
    }
}
