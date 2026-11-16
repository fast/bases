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
use bases::EncodeError;
use data_encoding::Encoding;

pub fn check<C: Codec>(reference: &Encoding) {
    let (input_quantum, encoded_quantum) = match reference.bit_width() {
        4 => (1, 2),
        5 => (5, 8),
        6 => (3, 4),
        _ => unreachable!(),
    };
    assert_lengths::<C>(
        input_quantum,
        encoded_quantum,
        reference.specification().padding.is_some(),
    );
}

fn assert_lengths<C: Codec>(input_quantum: usize, encoded_quantum: usize, padded: bool) {
    let mut lengths: Vec<usize> = (0..=16).collect();
    for boundary in [
        usize::MAX / 8,
        usize::MAX / 4,
        isize::MAX as usize,
        (usize::MAX / encoded_quantum) * input_quantum,
        usize::MAX,
    ] {
        for delta in 0..=encoded_quantum {
            lengths.extend(boundary.checked_sub(delta));
            lengths.extend(boundary.checked_add(delta));
        }
    }
    lengths.sort_unstable();
    lengths.dedup();

    // A wider integer makes the direct mathematical formulas independent of
    // the checked and quotient/remainder implementations under test.
    let raw = input_quantum as u128;
    let encoded = encoded_quantum as u128;
    let encoded_len = |length: u128| {
        if padded {
            length.div_ceil(raw) * encoded
        } else {
            (length * encoded).div_ceil(raw)
        }
    };

    for &length in &lengths {
        assert_eq!(
            C::encoded_len(length),
            representable(encoded_len(length as u128)),
            "{} encoded length for {length}",
            C::name(),
        );
        let capacity = C::max_decoded_len(length) as u128;
        let max_payload = length as u128 * raw / encoded;
        // An impossible wire length has no payload to reserve space for.
        if encoded_len(max_payload) == length as u128 {
            assert!(
                capacity >= max_payload,
                "{} decoded bound for {length}",
                C::name()
            );
        }
    }

    for pending in 0..input_quantum {
        let mut encoder = C::encoder();
        encoder.update(&[0_u8; 5][..pending], &mut []).unwrap();
        assert_eq!(
            encoder.finish_output_len() as u128,
            encoded_len(pending as u128)
        );

        for &length in &lengths {
            let expected = (length as u128 + pending as u128) / raw * encoded;
            assert_eq!(
                encoder.update_output_len(length),
                representable(expected),
                "{} encoder length for {length} with {pending} pending bytes",
                C::name(),
            );
        }
    }

    for pending in 0..encoded_quantum {
        let mut decoder = C::decoder();
        decoder.update(&[b'A'; 8][..pending], &mut []).unwrap();

        for &length in &lengths {
            let expected = (length as u128 + pending as u128) / encoded * raw;
            assert!(
                decoder.max_update_output_len(length) as u128 >= expected,
                "{} decoder bound for {length} with {pending} pending symbols",
                C::name(),
            );
        }
    }
}

fn representable(length: u128) -> Result<usize, EncodeError> {
    usize::try_from(length).map_err(|_| EncodeError::LengthOverflow)
}
