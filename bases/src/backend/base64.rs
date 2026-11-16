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

//! Base64 execution backend.

use crate::DecodeError;
use crate::backend;

#[allow(unsafe_code)]
mod arch;

const STANDARD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const URL_SAFE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

const STD_DECODE_LUTS: [[i32; 256]; 4] = make_decode_luts(false);
const URL_DECODE_LUTS: [[i32; 256]; 4] = make_decode_luts(true);

const fn make_decode_luts(url_safe: bool) -> [[i32; 256]; 4] {
    let mut luts = [[-1_i32; 256]; 4];
    let alphabet = if url_safe { URL_SAFE } else { STANDARD };
    let mut idx = 0;
    while idx < 64 {
        let byte = alphabet[idx] as usize;
        let v = idx as i32;
        luts[0][byte] = v << 2;
        luts[1][byte] = (v >> 4) | ((v & 0x0f) << 12);
        luts[2][byte] = ((v >> 2) << 8) | ((v & 0x03) << 22);
        luts[3][byte] = v << 16;
        idx += 1;
    }
    luts
}

/// Computes the exact Base64 encoded length for `input_len`.
#[inline]
pub const fn encoded_len<const PADDED: bool>(input_len: usize) -> Option<usize> {
    let complete = match (input_len / 3).checked_mul(4) {
        Some(length) => length,
        None => return None,
    };
    let trailing = match (input_len % 3, PADDED) {
        (0, _) => 0,
        (_, true) => 4,
        (1, false) => 2,
        (2, false) => 3,
        _ => 0,
    };
    complete.checked_add(trailing)
}

/// Computes an upper bound on the Base64 decoded length for `input_len`.
#[inline]
pub const fn max_decoded_len(input_len: usize) -> usize {
    let complete = (input_len / 4) * 3;
    let trailing = match input_len % 4 {
        2 => 1,
        3 => 2,
        _ => 0,
    };
    complete + trailing
}

/// Validates Base64 input length and padding shape, returning the exact decoded length.
#[inline]
pub fn decoded_len<const PADDED: bool>(input: &[u8]) -> Result<usize, DecodeError> {
    decoded_len_and_padding::<PADDED>(input).map(|(len, _)| len)
}

#[inline]
fn decoded_len_and_padding<const PADDED: bool>(
    input: &[u8],
) -> Result<(usize, usize), DecodeError> {
    if PADDED {
        let len = input.len();
        if len % 4 != 0 {
            return Err(DecodeError::InvalidLength { length: len });
        }
        if len == 0 {
            return Ok((0, 0));
        }

        let padding = if input[len - 1] != b'=' {
            0
        } else if input[len - 2] != b'=' {
            1
        } else if input[len - 3] != b'=' {
            2
        } else {
            backend::trailing_padding(input)
        };
        if padding > 2 {
            return Err(DecodeError::InvalidPadding {
                index: len - padding,
            });
        }
        Ok((len / 4 * 3 - padding, padding))
    } else {
        let trailing = match input.len() % 4 {
            0 => 0,
            2 => 1,
            3 => 2,
            _ => {
                return Err(DecodeError::InvalidLength {
                    length: input.len(),
                });
            }
        };
        Ok((input.len() / 4 * 3 + trailing, 0))
    }
}

/// Encodes `input` into `output` using Base64, returning the number of bytes written.
#[inline]
pub fn encode<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    input: &[u8],
    output: &mut [u8],
) -> usize {
    let required = encoded_len::<PADDED>(input.len()).expect("encoded length overflow");
    assert!(output.len() >= required, "output buffer too small");

    let (input_index, output_index) = arch::encode::<URL_SAFE_ALPHABET>(input, output);
    encode_scalar_from::<URL_SAFE_ALPHABET, PADDED>(input, output, input_index, output_index)
}

#[inline]
fn decode_complete_len<const PADDED: bool>(input: &[u8]) -> Result<usize, DecodeError> {
    if PADDED {
        if input.len() % 4 != 0 {
            return Err(DecodeError::InvalidLength {
                length: input.len(),
            });
        }
        if input[input.len() - 1] == b'=' {
            Ok(input.len() - 4)
        } else {
            Ok(input.len())
        }
    } else {
        match input.len() % 4 {
            0 | 2 | 3 => Ok((input.len() / 4) * 4),
            _ => Err(DecodeError::InvalidLength {
                length: input.len(),
            }),
        }
    }
}

/// Decodes Base64 `input` into `output`, returning the number of bytes written.
#[inline]
pub fn decode<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    input: &[u8],
    output: &mut [u8],
) -> Result<usize, DecodeError> {
    if input.is_empty() {
        return Ok(0);
    }

    let complete_len = decode_complete_len::<PADDED>(input)?;
    assert!(
        output.len() >= (complete_len / 4) * 3,
        "output buffer too small"
    );

    let (input_index, output_index, had_invalid) =
        arch::decode::<URL_SAFE_ALPHABET>(&input[..complete_len], output);
    decode_from::<URL_SAFE_ALPHABET, PADDED>(
        input,
        output,
        complete_len,
        input_index,
        output_index,
        had_invalid,
    )
}

#[inline]
fn decode_from<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    input: &[u8],
    output: &mut [u8],
    complete_len: usize,
    input_index: usize,
    output_index: usize,
    had_invalid: bool,
) -> Result<usize, DecodeError> {
    if had_invalid {
        let lut = if URL_SAFE_ALPHABET {
            &URL_DECODE_LUTS[3]
        } else {
            &STD_DECODE_LUTS[3]
        };
        return Err(find_invalid_in_slice(
            &input[input_index..complete_len],
            input_index,
            lut,
        ));
    }

    let mut output_index = decode_scalar_complete_from::<URL_SAFE_ALPHABET>(
        &input[..complete_len],
        output,
        input_index,
        output_index,
    )?;

    if PADDED {
        if complete_len != input.len() {
            let last = &input[complete_len..];
            let padding = if last[2] != b'=' {
                1
            } else if last[1] != b'=' {
                2
            } else if last[0] != b'=' {
                return Err(DecodeError::InvalidPadding {
                    index: input.len() - 3,
                });
            } else {
                return Err(DecodeError::InvalidPadding {
                    index: input.len() - 4,
                });
            };
            output_index += decode_tail::<URL_SAFE_ALPHABET>(
                &last[..4 - padding],
                complete_len,
                &mut output[output_index..],
            )?;
        }
    } else if complete_len != input.len() {
        output_index += decode_tail::<URL_SAFE_ALPHABET>(
            &input[complete_len..],
            complete_len,
            &mut output[output_index..],
        )?;
    }

    Ok(output_index)
}

/// Decodes Base64 `buf` in place, returning the number of decoded bytes.
#[inline]
pub fn decode_in_place<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    buf: &mut [u8],
) -> Result<usize, DecodeError> {
    let (required, padding) = decoded_len_and_padding::<PADDED>(buf)?;
    let data_len = buf.len() - padding;
    let complete_len = (data_len / 4) * 4;

    let (input_index, output_index, had_invalid) =
        arch::decode_in_place::<URL_SAFE_ALPHABET>(buf, complete_len);
    let written = decode_in_place_from::<URL_SAFE_ALPHABET>(
        buf,
        data_len,
        complete_len,
        input_index,
        output_index,
        had_invalid,
    )?;

    debug_assert_eq!(written, required);
    Ok(written)
}

#[inline]
fn decode_in_place_from<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    data_len: usize,
    complete_len: usize,
    input_index: usize,
    output_index: usize,
    had_invalid: bool,
) -> Result<usize, DecodeError> {
    if had_invalid {
        let lut = if URL_SAFE_ALPHABET {
            &URL_DECODE_LUTS[3]
        } else {
            &STD_DECODE_LUTS[3]
        };
        return Err(find_invalid_in_slice(
            &buf[input_index..complete_len],
            input_index,
            lut,
        ));
    }

    let mut output_index = decode_in_place_scalar_complete_from::<URL_SAFE_ALPHABET>(
        buf,
        complete_len,
        input_index,
        output_index,
    )?;

    if complete_len != data_len {
        let mut tail = [0_u8; 3];
        let tail_len = data_len - complete_len;
        tail[..tail_len].copy_from_slice(&buf[complete_len..data_len]);
        output_index += decode_tail::<URL_SAFE_ALPHABET>(
            &tail[..tail_len],
            complete_len,
            &mut buf[output_index..],
        )?;
    }

    Ok(output_index)
}

#[inline]
fn encode_quantum(alphabet: &[u8; 64], w: u32) -> [u8; 4] {
    [
        alphabet[((w >> 18) & 0x3f) as usize],
        alphabet[((w >> 12) & 0x3f) as usize],
        alphabet[((w >> 6) & 0x3f) as usize],
        alphabet[(w & 0x3f) as usize],
    ]
}

#[inline]
fn encode_scalar_from<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    input: &[u8],
    output: &mut [u8],
    mut input_index: usize,
    mut output_index: usize,
) -> usize {
    let alphabet = if URL_SAFE_ALPHABET {
        URL_SAFE
    } else {
        STANDARD
    };

    while input_index + 12 <= input.len() {
        let s = &input[input_index..input_index + 12];
        let w0 = u32::from_be_bytes([0, s[0], s[1], s[2]]);
        let w1 = u32::from_be_bytes([0, s[3], s[4], s[5]]);
        let w2 = u32::from_be_bytes([0, s[6], s[7], s[8]]);
        let w3 = u32::from_be_bytes([0, s[9], s[10], s[11]]);
        let e0 = encode_quantum(alphabet, w0);
        let e1 = encode_quantum(alphabet, w1);
        let e2 = encode_quantum(alphabet, w2);
        let e3 = encode_quantum(alphabet, w3);
        output[output_index..output_index + 4].copy_from_slice(&e0);
        output[output_index + 4..output_index + 8].copy_from_slice(&e1);
        output[output_index + 8..output_index + 12].copy_from_slice(&e2);
        output[output_index + 12..output_index + 16].copy_from_slice(&e3);
        input_index += 12;
        output_index += 16;
    }

    while input_index + 6 <= input.len() {
        let s = &input[input_index..input_index + 6];
        let w = u64::from_be_bytes([0, 0, s[0], s[1], s[2], s[3], s[4], s[5]]);
        output[output_index..output_index + 8].copy_from_slice(&[
            alphabet[((w >> 42) & 0x3f) as usize],
            alphabet[((w >> 36) & 0x3f) as usize],
            alphabet[((w >> 30) & 0x3f) as usize],
            alphabet[((w >> 24) & 0x3f) as usize],
            alphabet[((w >> 18) & 0x3f) as usize],
            alphabet[((w >> 12) & 0x3f) as usize],
            alphabet[((w >> 6) & 0x3f) as usize],
            alphabet[(w & 0x3f) as usize],
        ]);
        input_index += 6;
        output_index += 8;
    }

    while input_index + 3 <= input.len() {
        let s = &input[input_index..input_index + 3];
        let w = u32::from_be_bytes([0, s[0], s[1], s[2]]);
        output[output_index..output_index + 4].copy_from_slice(&encode_quantum(alphabet, w));
        input_index += 3;
        output_index += 4;
    }

    match &input[input_index..] {
        [first] => {
            output[output_index] = alphabet[usize::from(first >> 2)];
            output[output_index + 1] = alphabet[usize::from((first << 4) & 0x3f)];
            output_index += 2;
            if PADDED {
                output[output_index] = b'=';
                output[output_index + 1] = b'=';
                output_index += 2;
            }
        }
        [first, second] => {
            output[output_index] = alphabet[usize::from(first >> 2)];
            output[output_index + 1] = alphabet[usize::from(((first << 4) | (second >> 4)) & 0x3f)];
            output[output_index + 2] = alphabet[usize::from((second << 2) & 0x3f)];
            output_index += 3;
            if PADDED {
                output[output_index] = b'=';
                output_index += 1;
            }
        }
        [] => {}
        _ => unreachable!(),
    }

    output_index
}

#[inline]
fn decode_4_lut(luts: &[[i32; 256]; 4], b0: u8, b1: u8, b2: u8, b3: u8) -> i32 {
    luts[0][usize::from(b0)]
        | luts[1][usize::from(b1)]
        | luts[2][usize::from(b2)]
        | luts[3][usize::from(b3)]
}

#[inline]
fn decode_chunk_16(luts: &[[i32; 256]; 4], s: &[u8; 16]) -> Option<[u8; 12]> {
    let w0 = decode_4_lut(luts, s[0], s[1], s[2], s[3]);
    let w1 = decode_4_lut(luts, s[4], s[5], s[6], s[7]);
    let w2 = decode_4_lut(luts, s[8], s[9], s[10], s[11]);
    let w3 = decode_4_lut(luts, s[12], s[13], s[14], s[15]);
    if (w0 | w1 | w2 | w3) < 0 {
        return None;
    }
    let lo = u64::from(w0 as u32) | (u64::from(w1 as u32) << 24) | (u64::from(w2 as u32) << 48);
    let hi = ((w2 as u32) >> 16) | ((w3 as u32) << 8);
    let lo = lo.to_le_bytes();
    let hi = hi.to_le_bytes();
    Some([
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
    ])
}

#[inline]
fn decode_chunk_4(luts: &[[i32; 256]; 4], s: &[u8; 4]) -> Option<[u8; 3]> {
    let w = decode_4_lut(luts, s[0], s[1], s[2], s[3]);
    if w < 0 {
        return None;
    }
    let bytes = w.to_le_bytes();
    Some([bytes[0], bytes[1], bytes[2]])
}

#[inline]
fn decode_scalar_complete_from<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [u8],
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let luts = if URL_SAFE_ALPHABET {
        &URL_DECODE_LUTS
    } else {
        &STD_DECODE_LUTS
    };

    while input_index + 16 <= input.len() {
        let s: &[u8; 16] = input[input_index..input_index + 16]
            .try_into()
            .expect("16-byte slice fits in [u8; 16]");
        let Some(decoded) = decode_chunk_16(luts, s) else {
            return Err(find_invalid_in_slice(s, input_index, &luts[3]));
        };
        output[output_index..output_index + 12].copy_from_slice(&decoded);
        input_index += 16;
        output_index += 12;
    }

    while input_index + 4 <= input.len() {
        let s: &[u8; 4] = input[input_index..input_index + 4]
            .try_into()
            .expect("4-byte slice fits in [u8; 4]");
        let Some(decoded) = decode_chunk_4(luts, s) else {
            return Err(find_invalid_in_slice(s, input_index, &luts[3]));
        };
        output[output_index..output_index + 3].copy_from_slice(&decoded);
        input_index += 4;
        output_index += 3;
    }

    Ok(output_index)
}

#[inline]
fn decode_in_place_scalar_complete_from<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    complete_len: usize,
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let luts = if URL_SAFE_ALPHABET {
        &URL_DECODE_LUTS
    } else {
        &STD_DECODE_LUTS
    };

    while input_index + 16 <= complete_len {
        let s: [u8; 16] = buf[input_index..input_index + 16]
            .try_into()
            .expect("16-byte slice fits in [u8; 16]");
        let Some(decoded) = decode_chunk_16(luts, &s) else {
            return Err(find_invalid_in_slice(&s, input_index, &luts[3]));
        };
        buf[output_index..output_index + 12].copy_from_slice(&decoded);
        input_index += 16;
        output_index += 12;
    }

    while input_index + 4 <= complete_len {
        let s: [u8; 4] = buf[input_index..input_index + 4]
            .try_into()
            .expect("4-byte slice fits in [u8; 4]");
        let Some(decoded) = decode_chunk_4(luts, &s) else {
            return Err(find_invalid_in_slice(&s, input_index, &luts[3]));
        };
        buf[output_index..output_index + 3].copy_from_slice(&decoded);
        input_index += 4;
        output_index += 3;
    }

    Ok(output_index)
}

fn decode_tail<const URL_SAFE_ALPHABET: bool>(
    trailing: &[u8],
    base_index: usize,
    output: &mut [u8],
) -> Result<usize, DecodeError> {
    let lut = if URL_SAFE_ALPHABET {
        &URL_DECODE_LUTS[3]
    } else {
        &STD_DECODE_LUTS[3]
    };

    match trailing {
        [first, second] => {
            let v0 = decode_value(*first, base_index, lut)?;
            let v1 = decode_value(*second, base_index + 1, lut)?;
            if v1 & 0x0f != 0 {
                return Err(DecodeError::NonCanonical {
                    index: base_index + 1,
                });
            }
            output[0] = (v0 << 2) | (v1 >> 4);
            Ok(1)
        }
        [first, second, third] => {
            let v0 = decode_value(*first, base_index, lut)?;
            let v1 = decode_value(*second, base_index + 1, lut)?;
            let v2 = decode_value(*third, base_index + 2, lut)?;
            if v2 & 0x03 != 0 {
                return Err(DecodeError::NonCanonical {
                    index: base_index + 2,
                });
            }
            output[0] = (v0 << 2) | (v1 >> 4);
            output[1] = (v1 << 4) | (v2 >> 2);
            Ok(2)
        }
        _ => unreachable!(),
    }
}

#[inline]
fn find_invalid_in_slice(slice: &[u8], base_index: usize, lut3: &[i32; 256]) -> DecodeError {
    for (offset, &byte) in slice.iter().enumerate() {
        if lut3[usize::from(byte)] < 0 {
            return backend::invalid_byte_error(byte, base_index + offset);
        }
    }
    unreachable!("called only when at least one byte in slice is invalid")
}

#[inline]
fn decode_value(byte: u8, index: usize, lut3: &[i32; 256]) -> Result<u8, DecodeError> {
    let val = lut3[usize::from(byte)];
    if val >= 0 {
        Ok((val >> 16) as u8)
    } else {
        Err(backend::invalid_byte_error(byte, index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_scalar<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> usize {
        encode_scalar_from::<URL_SAFE_ALPHABET, PADDED>(input, output, 0, 0)
    }

    fn decode_scalar<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        if input.is_empty() {
            return Ok(0);
        }
        let complete_len = decode_complete_len::<PADDED>(input)?;
        decode_from::<URL_SAFE_ALPHABET, PADDED>(input, output, complete_len, 0, 0, false)
    }

    fn decode_in_place_scalar<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        buf: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let (required, padding) = decoded_len_and_padding::<PADDED>(buf)?;
        let data_len = buf.len() - padding;
        let complete_len = (data_len / 4) * 4;
        let written =
            decode_in_place_from::<URL_SAFE_ALPHABET>(buf, data_len, complete_len, 0, 0, false)?;
        debug_assert_eq!(written, required);
        Ok(written)
    }

    fn fill_pattern(buf: &mut [u8], seed: usize) {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = ((i.wrapping_mul(199).wrapping_add(seed.wrapping_mul(37))) & 0xff) as u8;
        }
    }

    #[test]
    fn length_calculations_and_bounds() {
        assert_eq!(encoded_len::<true>(0), Some(0));
        assert_eq!(encoded_len::<false>(0), Some(0));
        assert_eq!(encoded_len::<true>(1), Some(4));
        assert_eq!(encoded_len::<false>(1), Some(2));
        assert_eq!(encoded_len::<true>(2), Some(4));
        assert_eq!(encoded_len::<false>(2), Some(3));
        assert_eq!(encoded_len::<true>(3), Some(4));
        assert_eq!(encoded_len::<false>(3), Some(4));

        let max_quanta = usize::MAX / 4;
        assert_eq!(encoded_len::<true>(max_quanta * 3), Some(max_quanta * 4));
        assert_eq!(encoded_len::<true>(max_quanta * 3 + 1), None);

        assert_eq!(max_decoded_len(0), 0);
        assert_eq!(max_decoded_len(1), 0);
        assert_eq!(max_decoded_len(2), 1);
        assert_eq!(max_decoded_len(3), 2);
        assert_eq!(max_decoded_len(4), 3);
        assert_eq!(max_decoded_len(usize::MAX), (usize::MAX / 4) * 3 + 2);

        assert_eq!(decoded_len::<true>(b""), Ok(0));
        assert_eq!(decoded_len::<true>(b"AA=="), Ok(1));
        assert_eq!(decoded_len::<true>(b"AAA="), Ok(2));
        assert_eq!(decoded_len::<true>(b"AAAA"), Ok(3));
        assert_eq!(
            decoded_len::<true>(b"A==="),
            Err(DecodeError::InvalidPadding { index: 1 })
        );
        assert_eq!(
            decoded_len::<true>(b"===="),
            Err(DecodeError::InvalidPadding { index: 0 })
        );
        assert_eq!(
            decoded_len::<true>(b"AAA"),
            Err(DecodeError::InvalidLength { length: 3 })
        );
        assert_eq!(
            decoded_len::<false>(b"A"),
            Err(DecodeError::InvalidLength { length: 1 })
        );
        assert_eq!(
            decoded_len::<false>(b"AAAAA"),
            Err(DecodeError::InvalidLength { length: 5 })
        );
    }

    fn check_roundtrip_and_arch_parity<const URL_SAFE_ALPHABET: bool, const PADDED: bool>() {
        let mut raw_buf = [0_u8; 260];
        let mut enc_buf = [0_u8; 360];
        let mut scalar_enc_buf = [0_u8; 360];
        let mut dec_buf = [0_u8; 260];
        let mut scalar_dec_buf = [0_u8; 260];
        let mut ip_buf = [0_u8; 360];
        let mut sip_buf = [0_u8; 360];

        for len in 0..=260 {
            let raw = &mut raw_buf[..len];
            fill_pattern(raw, len);

            let expected_enc_len = encoded_len::<PADDED>(len).expect("within bounds");
            let enc = &mut enc_buf[..expected_enc_len];
            let scalar_enc = &mut scalar_enc_buf[..expected_enc_len];

            assert_eq!(
                encode::<URL_SAFE_ALPHABET, PADDED>(raw, enc),
                expected_enc_len
            );
            assert_eq!(
                encode_scalar::<URL_SAFE_ALPHABET, PADDED>(raw, scalar_enc),
                expected_enc_len
            );
            assert_eq!(enc, scalar_enc, "encode mismatch at len={len}");

            let dec = &mut dec_buf[..len];
            let scalar_dec = &mut scalar_dec_buf[..len];
            assert_eq!(decode::<URL_SAFE_ALPHABET, PADDED>(enc, dec), Ok(len));
            assert_eq!(
                decode_scalar::<URL_SAFE_ALPHABET, PADDED>(enc, scalar_dec),
                Ok(len)
            );
            assert_eq!(dec, raw);
            assert_eq!(dec, scalar_dec);

            let ip = &mut ip_buf[..expected_enc_len];
            let sip = &mut sip_buf[..expected_enc_len];
            ip.copy_from_slice(enc);
            sip.copy_from_slice(enc);
            assert_eq!(decode_in_place::<URL_SAFE_ALPHABET, PADDED>(ip), Ok(len));
            assert_eq!(
                decode_in_place_scalar::<URL_SAFE_ALPHABET, PADDED>(sip),
                Ok(len)
            );
            assert_eq!(&ip[..len], raw);
            assert_eq!(&sip[..len], raw);
        }
    }

    #[test]
    fn roundtrip_and_arch_parity_all_profiles_0_to_260() {
        check_roundtrip_and_arch_parity::<false, true>();
        check_roundtrip_and_arch_parity::<false, false>();
        check_roundtrip_and_arch_parity::<true, true>();
        check_roundtrip_and_arch_parity::<true, false>();

        // Explicit check covering all 64 alphabet symbols across multiple SIMD chunks (64 bytes ->
        // 48 decoded)
        let std_sym = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/\
                        ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = [0_u8; 96];
        let mut scalar_out = [0_u8; 96];
        assert_eq!(decode::<false, true>(&std_sym, &mut out), Ok(96));
        assert_eq!(
            decode_scalar::<false, true>(&std_sym, &mut scalar_out),
            Ok(96)
        );
        assert_eq!(out, scalar_out);

        let mut reencoded = [0_u8; 128];
        assert_eq!(encode::<false, true>(&out, &mut reencoded), 128);
        assert_eq!(reencoded, std_sym);

        let url_sym = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_\
                        ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        assert_eq!(decode::<true, false>(&url_sym, &mut out), Ok(96));
        assert_eq!(
            decode_scalar::<true, false>(&url_sym, &mut scalar_out),
            Ok(96)
        );
        assert_eq!(out, scalar_out);
    }

    fn verify_all_bytes_at_offsets<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        raw_lens: &[usize],
    ) {
        let mut raw_buf = [0_u8; 64];
        let mut clean_enc_buf = [0_u8; 96];
        let mut mutated_buf = [0_u8; 96];
        let mut dec_buf = [0_u8; 64];
        let mut scalar_dec_buf = [0_u8; 64];
        let mut sip_buf = [0_u8; 96];

        for &raw_len in raw_lens {
            let raw = &mut raw_buf[..raw_len];
            fill_pattern(raw, raw_len + 13);
            let enc_len = encoded_len::<PADDED>(raw_len).expect("within bounds");
            let clean_enc = &mut clean_enc_buf[..enc_len];
            encode::<URL_SAFE_ALPHABET, PADDED>(raw, clean_enc);

            for pos in 0..enc_len {
                for byte in 0_u8..=u8::MAX {
                    let mutated = &mut mutated_buf[..enc_len];
                    mutated.copy_from_slice(clean_enc);
                    mutated[pos] = byte;

                    let out_cap = max_decoded_len(enc_len);
                    let actual =
                        decode::<URL_SAFE_ALPHABET, PADDED>(mutated, &mut dec_buf[..out_cap]);
                    let expected = decode_scalar::<URL_SAFE_ALPHABET, PADDED>(
                        mutated,
                        &mut scalar_dec_buf[..out_cap],
                    );
                    assert_eq!(
                        actual, expected,
                        "decode mismatch at raw_len={raw_len}, pos={pos}, byte={byte:#x}"
                    );
                    if let Ok(n) = actual {
                        assert_eq!(&dec_buf[..n], &scalar_dec_buf[..n]);
                    }

                    let sip = &mut sip_buf[..enc_len];
                    sip.copy_from_slice(mutated);
                    let ip_expected = decode_in_place_scalar::<URL_SAFE_ALPHABET, PADDED>(sip);
                    let ip_actual = decode_in_place::<URL_SAFE_ALPHABET, PADDED>(mutated);
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
    fn exhaustive_invalid_bytes_and_padding_across_chunk_boundaries() {
        // Covers 4B scalar, 16B SSSE3/wasm128/scalar, 32B AVX2, 64B NEON, and partial tails.
        let raw_lens = [1, 2, 3, 12, 14, 24, 36, 48, 51];
        verify_all_bytes_at_offsets::<false, true>(&raw_lens);
        verify_all_bytes_at_offsets::<false, false>(&raw_lens);
        verify_all_bytes_at_offsets::<true, true>(&raw_lens);
        verify_all_bytes_at_offsets::<true, false>(&raw_lens);
    }

    #[test]
    fn interior_padding_non_canonical_and_alphabet_cross_errors() {
        let mut out = [0_u8; 64];

        for pos in [0_usize, 15, 16, 31, 32, 63, 64, 78] {
            let mut buf = [b'A'; 80];
            buf[pos] = b'=';
            assert_eq!(
                decode::<false, true>(&buf, &mut out),
                Err(DecodeError::InvalidPadding { index: pos })
            );
            assert_eq!(
                decode_in_place::<false, true>(&mut buf),
                Err(DecodeError::InvalidPadding { index: pos })
            );
        }
        for pos in [0_usize, 15, 16, 31, 32, 63, 64, 79] {
            let mut buf = [b'A'; 80];
            buf[pos] = b'=';
            assert_eq!(
                decode::<true, false>(&buf, &mut out),
                Err(DecodeError::InvalidPadding { index: pos })
            );
            assert_eq!(
                decode_in_place::<true, false>(&mut buf),
                Err(DecodeError::InvalidPadding { index: pos })
            );
        }

        // Cross-alphabet rejection inside SIMD chunk (pos = 65, after 64B NEON / 2xAVX2)
        let mut cross = [b'A'; 80];
        cross[65] = b'-';
        assert_eq!(
            decode::<false, true>(&cross, &mut out),
            Err(DecodeError::InvalidByte {
                index: 65,
                byte: b'-'
            })
        );
        cross[65] = b'+';
        assert_eq!(
            decode::<true, true>(&cross, &mut out),
            Err(DecodeError::InvalidByte {
                index: 65,
                byte: b'+'
            })
        );

        // Non-canonical 2-char tail after 64B SIMD chunk: second & 0x0f != 0 ('B' = 1)
        let mut tail2 = [b'A'; 68];
        tail2[65] = b'B';
        tail2[66] = b'=';
        tail2[67] = b'=';
        assert_eq!(
            decode::<false, true>(&tail2, &mut out),
            Err(DecodeError::NonCanonical { index: 65 })
        );
        assert_eq!(
            decode::<false, false>(&tail2[..66], &mut out),
            Err(DecodeError::NonCanonical { index: 65 })
        );
        assert_eq!(
            decode_in_place::<false, true>(&mut tail2),
            Err(DecodeError::NonCanonical { index: 65 })
        );

        // Non-canonical 3-char tail after 64B SIMD chunk: third & 0x03 != 0 ('B' = 1)
        let mut tail3 = [b'A'; 68];
        tail3[66] = b'B';
        tail3[67] = b'=';
        assert_eq!(
            decode::<false, true>(&tail3, &mut out),
            Err(DecodeError::NonCanonical { index: 66 })
        );
        assert_eq!(
            decode::<false, false>(&tail3[..67], &mut out),
            Err(DecodeError::NonCanonical { index: 66 })
        );
        assert_eq!(
            decode_in_place::<false, true>(&mut tail3),
            Err(DecodeError::NonCanonical { index: 66 })
        );
    }
}
