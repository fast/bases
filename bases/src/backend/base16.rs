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

//! Base16 (hexadecimal) execution backend.

use core::mem::MaybeUninit;

use crate::DecodeError;
use crate::backend;
use crate::backend::DecodePlan;

#[allow(unsafe_code)]
mod arch;

const MIXED_DECODE_LUT: [u8; 256] = make_decode_lut(true, true);
const UPPER_DECODE_LUT: [u8; 256] = make_decode_lut(true, false);
const LOWER_DECODE_LUT: [u8; 256] = make_decode_lut(false, true);

const fn make_decode_lut(allow_upper: bool, allow_lower: bool) -> [u8; 256] {
    let mut lut = [0xff_u8; 256];
    let mut byte = b'0';
    while byte <= b'9' {
        lut[byte as usize] = byte - b'0';
        byte += 1;
    }
    if allow_upper {
        let mut byte = b'A';
        while byte <= b'F' {
            lut[byte as usize] = byte - b'A' + 10;
            byte += 1;
        }
    }
    if allow_lower {
        let mut byte = b'a';
        while byte <= b'f' {
            lut[byte as usize] = byte - b'a' + 10;
            byte += 1;
        }
    }
    lut
}

const fn select_decode_lut<const UPPERCASE: bool, const STRICT_CASE: bool>() -> &'static [u8; 256] {
    if !STRICT_CASE {
        &MIXED_DECODE_LUT
    } else if UPPERCASE {
        &UPPER_DECODE_LUT
    } else {
        &LOWER_DECODE_LUT
    }
}

/// Computes the exact Base16 encoded length for `input_len`.
#[inline]
pub const fn encoded_len(input_len: usize) -> Option<usize> {
    input_len.checked_mul(2)
}

/// Computes an upper bound on the Base16 decoded length for `input_len`.
#[inline]
pub const fn max_decoded_len(input_len: usize) -> usize {
    input_len / 2
}

/// Validates Base16 input length and returns the exact decode plan.
#[inline]
pub fn decode_plan(input: &[u8]) -> Result<DecodePlan, DecodeError> {
    if input.len() % 2 != 0 {
        return Err(DecodeError::InvalidLength {
            length: input.len(),
        });
    }
    Ok(DecodePlan {
        required: input.len() / 2,
        data_len: input.len(),
        complete_len: input.len(),
        padding: 0,
    })
}

/// Checks alphabet symbols after input length has been validated.
#[inline]
pub fn check_with_plan<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    plan: DecodePlan,
) -> Result<(), DecodeError> {
    debug_assert_eq!(plan.data_len, input.len());
    let lut = select_decode_lut::<UPPERCASE, STRICT_CASE>();
    if input.len() < 32 {
        for (index, &byte) in input.iter().enumerate() {
            decode_value(byte, index, lut)?;
        }
    } else if let Some(index) = backend::find_invalid(input, lut) {
        return Err(DecodeError::InvalidByte {
            index,
            byte: input[index],
        });
    }
    Ok(())
}

/// Encodes `input` into `output` using Base16, returning the number of bytes written.
#[inline(always)]
pub fn encode<const UPPERCASE: bool>(input: &[u8], output: &mut [MaybeUninit<u8>]) -> usize {
    let required = encoded_len(input.len()).expect("encoded length overflow");
    assert!(output.len() >= required, "output buffer too small");

    let (input_index, output_index) = arch::encode::<UPPERCASE>(input, output);
    encode_scalar_from::<UPPERCASE>(input, output, input_index, output_index)
}

/// Encodes a single 1-byte Base16 quantum into `output`.
#[inline]
pub fn encode_quantum<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> usize {
    debug_assert_eq!(input.len(), 1);
    debug_assert!(output.len() >= 2);
    output[0].write(encode_nibble::<UPPERCASE>(input[0] >> 4));
    output[1].write(encode_nibble::<UPPERCASE>(input[0] & 0x0f));
    2
}

/// Decodes Base16 `input` into `output` using `plan`, returning the number of bytes written.
#[inline]
pub fn decode_with_plan<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    plan: DecodePlan,
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    debug_assert_eq!(plan.data_len, input.len());
    debug_assert_eq!(plan.complete_len, input.len());
    debug_assert_eq!(plan.padding, 0);
    debug_assert_eq!(plan.required, input.len() / 2);
    assert!(output.len() >= plan.required, "output buffer too small");
    let out = &mut output[..plan.required];

    let (input_index, output_index) = arch::decode::<UPPERCASE, STRICT_CASE>(input, out);
    decode_scalar_from::<UPPERCASE, STRICT_CASE>(input, out, input_index, output_index)
}

/// Decodes a single 2-byte Base16 quantum into `output`.
#[inline]
pub fn decode_quantum<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    debug_assert_eq!(input.len(), 2);
    debug_assert!(!output.is_empty());
    let lut = select_decode_lut::<UPPERCASE, STRICT_CASE>();
    output[0].write(decode_pair(lut, input[0], input[1], 0)?);
    Ok(1)
}

/// Decodes Base16 `buf` in place, returning the number of decoded bytes.
#[inline]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> Result<usize, DecodeError> {
    let plan = decode_plan(buf)?;
    let (input_index, output_index) = arch::decode_in_place::<UPPERCASE, STRICT_CASE>(buf);
    let written =
        decode_in_place_scalar_from::<UPPERCASE, STRICT_CASE>(buf, input_index, output_index)?;
    debug_assert_eq!(written, plan.required);
    Ok(written)
}

#[inline]
fn encode_scalar_from<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    input_index: usize,
    output_index: usize,
) -> usize {
    for (&byte, pair) in input[input_index..]
        .iter()
        .zip(output[output_index..].as_chunks_mut::<2>().0)
    {
        pair[0].write(encode_nibble::<UPPERCASE>(byte >> 4));
        pair[1].write(encode_nibble::<UPPERCASE>(byte & 0x0f));
    }
    output_index + (input.len() - input_index) * 2
}

#[inline]
fn encode_nibble<const UPPERCASE: bool>(nibble: u8) -> u8 {
    let letter_offset = if UPPERCASE {
        b'A' - b'0' - 10
    } else {
        b'a' - b'0' - 10
    };
    b'0' + nibble + u8::from(nibble >= 10) * letter_offset
}

#[inline]
fn decode_chunk_8(lut: &[u8; 256], chunk: &[u8; 8]) -> Option<[u8; 4]> {
    let d0 = lut[usize::from(chunk[0])];
    let d1 = lut[usize::from(chunk[1])];
    let d2 = lut[usize::from(chunk[2])];
    let d3 = lut[usize::from(chunk[3])];
    let d4 = lut[usize::from(chunk[4])];
    let d5 = lut[usize::from(chunk[5])];
    let d6 = lut[usize::from(chunk[6])];
    let d7 = lut[usize::from(chunk[7])];
    if (d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7) & 0xf0 != 0 {
        return None;
    }
    Some([
        (d0 << 4) | d1,
        (d2 << 4) | d3,
        (d4 << 4) | d5,
        (d6 << 4) | d7,
    ])
}

#[inline]
fn decode_pair(lut: &[u8; 256], b0: u8, b1: u8, index: usize) -> Result<u8, DecodeError> {
    let d0 = decode_value(b0, index, lut)?;
    let d1 = decode_value(b1, index + 1, lut)?;
    Ok((d0 << 4) | d1)
}

#[inline]
fn decode_value(byte: u8, index: usize, lut: &[u8; 256]) -> Result<u8, DecodeError> {
    let value = lut[usize::from(byte)];
    if value & 0xf0 == 0 {
        Ok(value)
    } else {
        Err(DecodeError::InvalidByte { index, byte })
    }
}

#[inline]
fn decode_scalar_from<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<UPPERCASE, STRICT_CASE>();

    while input_index + 8 <= input.len() {
        let chunk: &[u8; 8] = input[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let Some(decoded) = decode_chunk_8(lut, chunk) else {
            break;
        };
        backend::write_bytes(&mut output[output_index..], decoded);
        input_index += 8;
        output_index += 4;
    }

    while input_index + 2 <= input.len() {
        output[output_index].write(decode_pair(
            lut,
            input[input_index],
            input[input_index + 1],
            input_index,
        )?);
        input_index += 2;
        output_index += 1;
    }

    Ok(output_index)
}

#[inline]
fn decode_in_place_scalar_from<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<UPPERCASE, STRICT_CASE>();

    while input_index + 8 <= buf.len() {
        let chunk: [u8; 8] = buf[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let Some(decoded) = decode_chunk_8(lut, &chunk) else {
            break;
        };
        buf[output_index..output_index + 4].copy_from_slice(&decoded);
        input_index += 8;
        output_index += 4;
    }

    while input_index + 2 <= buf.len() {
        buf[output_index] = decode_pair(lut, buf[input_index], buf[input_index + 1], input_index)?;
        input_index += 2;
        output_index += 1;
    }

    Ok(output_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec;

    fn decoded_len(input: &[u8]) -> Result<usize, DecodeError> {
        decode_plan(input).map(|plan| plan.required)
    }

    fn encode_init<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> usize {
        encode::<UPPERCASE>(input, codec::as_uninit_slice(output))
    }

    fn decode_init<const UPPERCASE: bool, const STRICT_CASE: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan(input)?;
        decode_with_plan::<UPPERCASE, STRICT_CASE>(input, plan, codec::as_uninit_slice(output))
    }

    fn decode_scalar_init<const UPPERCASE: bool, const STRICT_CASE: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        decode_scalar_from::<UPPERCASE, STRICT_CASE>(input, codec::as_uninit_slice(output), 0, 0)
    }

    fn decode_in_place_scalar<const UPPERCASE: bool, const STRICT_CASE: bool>(
        buf: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let _ = decoded_len(buf)?;
        decode_in_place_scalar_from::<UPPERCASE, STRICT_CASE>(buf, 0, 0)
    }

    fn fill_pattern(buf: &mut [u8], seed: usize) {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = ((i.wrapping_mul(131).wrapping_add(seed.wrapping_mul(17))) & 0xff) as u8;
        }
    }

    #[test]
    fn length_calculations_and_bounds() {
        assert_eq!(encoded_len(0), Some(0));
        assert_eq!(encoded_len(usize::MAX / 2), Some((usize::MAX / 2) * 2));
        assert_eq!(encoded_len(usize::MAX / 2 + 1), None);
        assert_eq!(max_decoded_len(0), 0);
        assert_eq!(max_decoded_len(1), 0);
        assert_eq!(max_decoded_len(usize::MAX), usize::MAX / 2);

        assert_eq!(decoded_len(b""), Ok(0));
        assert_eq!(decoded_len(b"0a"), Ok(1));
        assert_eq!(
            decoded_len(b"0"),
            Err(DecodeError::InvalidLength { length: 1 })
        );
        assert_eq!(
            decoded_len(b"0ab"),
            Err(DecodeError::InvalidLength { length: 3 })
        );
    }

    #[test]
    fn roundtrip_and_arch_parity_lengths_0_to_260() {
        let mut raw_buf = [0_u8; 260];
        let mut enc_buf = [0_u8; 520];
        let mut scalar_enc_buf = [0_u8; 520];
        let mut dec_buf = [0_u8; 260];
        let mut scalar_dec_buf = [0_u8; 260];
        let mut in_place_buf = [0_u8; 520];
        let mut scalar_in_place_buf = [0_u8; 520];

        for len in 0..=260 {
            let raw = &mut raw_buf[..len];
            fill_pattern(raw, len);

            // Uppercase profile
            let enc = &mut enc_buf[..len * 2];
            let scalar_enc = &mut scalar_enc_buf[..len * 2];
            assert_eq!(encode_init::<true>(raw, enc), len * 2);
            assert_eq!(
                encode_scalar_from::<true>(raw, codec::as_uninit_slice(scalar_enc), 0, 0),
                len * 2
            );
            assert_eq!(enc, scalar_enc);

            let dec = &mut dec_buf[..len];
            let scalar_dec = &mut scalar_dec_buf[..len];
            assert_eq!(decode_init::<true, true>(enc, dec), Ok(len));
            assert_eq!(decode_scalar_init::<true, true>(enc, scalar_dec), Ok(len));
            assert_eq!(dec, raw);
            assert_eq!(dec, scalar_dec);

            let ip = &mut in_place_buf[..len * 2];
            let sip = &mut scalar_in_place_buf[..len * 2];
            ip.copy_from_slice(enc);
            sip.copy_from_slice(enc);
            assert_eq!(decode_in_place::<true, true>(ip), Ok(len));
            assert_eq!(decode_in_place_scalar::<true, true>(sip), Ok(len));
            assert_eq!(&ip[..len], raw);
            assert_eq!(&sip[..len], raw);

            // Lowercase profile
            assert_eq!(encode_init::<false>(raw, enc), len * 2);
            assert_eq!(
                encode_scalar_from::<false>(raw, codec::as_uninit_slice(scalar_enc), 0, 0),
                len * 2
            );
            assert_eq!(enc, scalar_enc);

            assert_eq!(decode_init::<false, true>(enc, dec), Ok(len));
            assert_eq!(decode_scalar_init::<false, true>(enc, scalar_dec), Ok(len));
            assert_eq!(dec, raw);
            assert_eq!(dec, scalar_dec);

            ip.copy_from_slice(enc);
            sip.copy_from_slice(enc);
            assert_eq!(decode_in_place::<false, true>(ip), Ok(len));
            assert_eq!(decode_in_place_scalar::<false, true>(sip), Ok(len));
            assert_eq!(&ip[..len], raw);
            assert_eq!(&sip[..len], raw);

            // Mixed case accepted when STRICT_CASE == false
            for (idx, b) in enc.iter_mut().enumerate() {
                if idx % 3 == 0 && b.is_ascii_lowercase() {
                    *b = b.to_ascii_uppercase();
                }
            }
            assert_eq!(decode_init::<false, false>(enc, dec), Ok(len));
            assert_eq!(decode_init::<true, false>(enc, dec), Ok(len));
            assert_eq!(dec, raw);
            ip.copy_from_slice(enc);
            assert_eq!(decode_in_place::<false, false>(ip), Ok(len));
            assert_eq!(&ip[..len], raw);
        }
    }

    fn verify_all_bytes_at_offsets<const UPPERCASE: bool, const STRICT_CASE: bool>(
        raw_lens: &[usize],
    ) {
        let mut raw_buf = [0_u8; 132];
        let mut clean_enc_buf = [0_u8; 264];
        let mut mutated_buf = [0_u8; 264];
        let mut dec_buf = [0_u8; 132];
        let mut scalar_dec_buf = [0_u8; 132];
        let mut scalar_ip_buf = [0_u8; 264];

        for &raw_len in raw_lens {
            let raw = &mut raw_buf[..raw_len];
            fill_pattern(raw, raw_len + 3);
            let enc_len = raw_len * 2;
            let clean_enc = &mut clean_enc_buf[..enc_len];
            encode_init::<UPPERCASE>(raw, clean_enc);

            for pos in 0..enc_len {
                for byte in 0_u8..=u8::MAX {
                    let mutated = &mut mutated_buf[..enc_len];
                    mutated.copy_from_slice(clean_enc);
                    mutated[pos] = byte;

                    let actual =
                        decode_init::<UPPERCASE, STRICT_CASE>(mutated, &mut dec_buf[..raw_len]);
                    let expected = decode_scalar_init::<UPPERCASE, STRICT_CASE>(
                        mutated,
                        &mut scalar_dec_buf[..raw_len],
                    );
                    assert_eq!(
                        actual, expected,
                        "decode mismatch at raw_len={raw_len}, pos={pos}, byte={byte:#x}"
                    );
                    if actual.is_ok() {
                        assert_eq!(&dec_buf[..raw_len], &scalar_dec_buf[..raw_len]);
                    }

                    let sip = &mut scalar_ip_buf[..enc_len];
                    sip.copy_from_slice(mutated);
                    let ip_expected = decode_in_place_scalar::<UPPERCASE, STRICT_CASE>(sip);
                    let ip_actual = decode_in_place::<UPPERCASE, STRICT_CASE>(mutated);
                    assert_eq!(
                        ip_actual, ip_expected,
                        "decode_in_place mismatch at raw_len={raw_len}, pos={pos}, byte={byte:#x}"
                    );
                    if let Ok(n) = ip_actual {
                        assert_eq!(&mutated[..n], &sip[..n]);
                    }
                }
            }
        }
    }

    #[test]
    fn exhaustive_invalid_bytes_across_chunk_boundaries() {
        // Encoded lengths 2, 8, 16, 18, 30, 32, 34, 62, 64, 66 cross all
        // scalar/NEON/SSSE3/AVX2/wasm128 boundaries.
        let raw_lens = [1, 4, 8, 9, 15, 16, 17, 31, 32, 33];
        verify_all_bytes_at_offsets::<false, false>(&raw_lens);
        verify_all_bytes_at_offsets::<false, true>(&raw_lens);
        verify_all_bytes_at_offsets::<true, false>(&raw_lens);
        verify_all_bytes_at_offsets::<true, true>(&raw_lens);
    }

    #[test]
    fn strict_vs_mixed_case_and_odd_length_errors() {
        let mut out = [0_u8; 64];
        // Wrong case inside large SIMD chunk (offset 35) and at start (offset 0)
        let mut mixed_in_upper = [b'A'; 66];
        mixed_in_upper[0] = b'a';
        assert_eq!(
            decode_init::<true, true>(&mixed_in_upper, &mut out[..33]),
            Err(DecodeError::InvalidByte {
                index: 0,
                byte: b'a'
            })
        );
        mixed_in_upper[0] = b'A';
        mixed_in_upper[35] = b'f';
        assert_eq!(
            decode_init::<true, true>(&mixed_in_upper, &mut out[..33]),
            Err(DecodeError::InvalidByte {
                index: 35,
                byte: b'f'
            })
        );
        assert_eq!(
            decode_init::<true, false>(&mixed_in_upper, &mut out[..33]),
            Ok(33)
        );

        mixed_in_upper[35] = b'=';
        assert_eq!(
            decode_init::<false, false>(&mixed_in_upper, &mut out[..33]),
            Err(DecodeError::InvalidByte {
                index: 35,
                byte: b'='
            })
        );

        let mut odd = [b'a'; 3];
        assert_eq!(
            decode_init::<false, false>(&odd, &mut out),
            Err(DecodeError::InvalidLength { length: 3 })
        );
        assert_eq!(
            decode_in_place::<false, false>(&mut odd),
            Err(DecodeError::InvalidLength { length: 3 })
        );
    }
}
