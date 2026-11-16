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

use core::mem::MaybeUninit;

use crate::DecodeError;
use crate::backend;
use crate::backend::DecodePlan;
use crate::codec;

#[allow(unsafe_code)]
mod arch;

const STANDARD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const URL_SAFE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

const INVALID_SYMBOL: u8 = 0xff;
const STD_DECODE_LUTS: [[i32; 256]; 4] = make_decode_luts(false);
const URL_DECODE_LUTS: [[i32; 256]; 4] = make_decode_luts(true);
const STD_DECODE_BYTE_LUT: [u8; 256] = make_decode_byte_lut(false);
const URL_DECODE_BYTE_LUT: [u8; 256] = make_decode_byte_lut(true);

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

const fn make_decode_byte_lut(url_safe: bool) -> [u8; 256] {
    let mut lut = [INVALID_SYMBOL; 256];
    let alphabet = if url_safe { URL_SAFE } else { STANDARD };
    let mut idx = 0;
    while idx < 64 {
        lut[alphabet[idx] as usize] = idx as u8;
        idx += 1;
    }
    lut
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

/// Validates Base64 input length and padding shape, returning the exact decode plan.
#[inline]
pub fn decode_plan<const PADDED: bool>(input: &[u8]) -> Result<DecodePlan, DecodeError> {
    if PADDED {
        let len = input.len();
        if len % 4 != 0 {
            return Err(DecodeError::InvalidLength { length: len });
        }
        if len == 0 {
            return Ok(DecodePlan {
                required: 0,
                data_len: 0,
                complete_len: 0,
                padding: 0,
            });
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
        let data_len = len - padding;
        let complete_len = (data_len / 4) * 4;
        Ok(DecodePlan {
            required: len / 4 * 3 - padding,
            data_len,
            complete_len,
            padding,
        })
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
        let complete_len = (input.len() / 4) * 4;
        Ok(DecodePlan {
            required: input.len() / 4 * 3 + trailing,
            data_len: input.len(),
            complete_len,
            padding: 0,
        })
    }
}

/// Encodes `input` into `output` using Base64, returning the number of bytes written.
#[inline]
pub fn encode<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> usize {
    let required = encoded_len::<PADDED>(input.len()).expect("encoded length overflow");
    assert!(output.len() >= required, "output buffer too small");

    let (input_index, output_index) = arch::encode::<URL_SAFE_ALPHABET>(input, output);
    encode_scalar_from::<URL_SAFE_ALPHABET, PADDED>(input, output, input_index, output_index)
}

/// Encodes a single 3-byte Base64 quantum into `output`.
#[inline]
pub fn encode_quantum<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> usize {
    let s: &[u8; 3] = input.try_into().expect("3-byte slice fits in [u8; 3]");
    debug_assert!(output.len() >= 4);
    let alphabet = if URL_SAFE_ALPHABET {
        URL_SAFE
    } else {
        STANDARD
    };
    let w = u32::from_be_bytes([0, s[0], s[1], s[2]]);
    backend::write_bytes(output, encode_quantum_3(alphabet, w));
    4
}

/// Decodes Base64 `input` into `output` using `plan`, returning the number of bytes written.
#[inline]
pub fn decode_with_plan<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    plan: DecodePlan,
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    if plan.data_len == 0 {
        return Ok(0);
    }
    assert!(output.len() >= plan.required, "output buffer too small");
    let out = &mut output[..plan.required];

    let (input_index, output_index, had_invalid) =
        arch::decode::<URL_SAFE_ALPHABET>(&input[..plan.complete_len], out);
    decode_from::<URL_SAFE_ALPHABET>(
        input,
        out,
        plan.data_len,
        plan.complete_len,
        input_index,
        output_index,
        had_invalid,
    )
}

/// Decodes a single complete 4-byte Base64 quantum into `output`.
#[inline]
pub fn decode_quantum<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    let lut = if URL_SAFE_ALPHABET {
        &URL_DECODE_BYTE_LUT
    } else {
        &STD_DECODE_BYTE_LUT
    };
    let s: &[u8; 4] = input.try_into().expect("4-byte slice fits in [u8; 4]");
    debug_assert!(output.len() >= 3);
    let Some(decoded) = decode_chunk_4_byte(lut, s) else {
        return Err(find_invalid_in_slice(s, 0, lut));
    };
    backend::write_bytes(output, decoded);
    Ok(3)
}

#[inline]
fn decode_from<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    data_len: usize,
    complete_len: usize,
    input_index: usize,
    output_index: usize,
    had_invalid: bool,
) -> Result<usize, DecodeError> {
    if had_invalid {
        let lut = if URL_SAFE_ALPHABET {
            &URL_DECODE_BYTE_LUT
        } else {
            &STD_DECODE_BYTE_LUT
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

    if complete_len != data_len {
        output_index += decode_tail::<URL_SAFE_ALPHABET>(
            &input[complete_len..data_len],
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
    let plan = decode_plan::<PADDED>(buf)?;

    let (input_index, output_index, had_invalid) =
        arch::decode_in_place::<URL_SAFE_ALPHABET>(buf, plan.complete_len, plan.required);
    let written = decode_in_place_from::<URL_SAFE_ALPHABET>(
        buf,
        plan.data_len,
        plan.complete_len,
        input_index,
        output_index,
        had_invalid,
    )?;

    debug_assert_eq!(written, plan.required);
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
            &URL_DECODE_BYTE_LUT
        } else {
            &STD_DECODE_BYTE_LUT
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
            codec::as_uninit_slice(&mut buf[output_index..]),
        )?;
    }

    Ok(output_index)
}

#[inline]
fn encode_quantum_3(alphabet: &[u8; 64], w: u32) -> [u8; 4] {
    [
        alphabet[((w >> 18) & 0x3f) as usize],
        alphabet[((w >> 12) & 0x3f) as usize],
        alphabet[((w >> 6) & 0x3f) as usize],
        alphabet[(w & 0x3f) as usize],
    ]
}

#[inline(always)]
fn encode_scalar_from<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    mut input_index: usize,
    mut output_index: usize,
) -> usize {
    let alphabet = if URL_SAFE_ALPHABET {
        URL_SAFE
    } else {
        STANDARD
    };

    // Read eight available bytes, but encode six so the load stays within the input.
    while input_index + 8 <= input.len() {
        let s = input[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let w = u64::from_be_bytes(s);
        backend::write_bytes(
            &mut output[output_index..],
            [
                alphabet[((w >> 58) & 0x3f) as usize],
                alphabet[((w >> 52) & 0x3f) as usize],
                alphabet[((w >> 46) & 0x3f) as usize],
                alphabet[((w >> 40) & 0x3f) as usize],
                alphabet[((w >> 34) & 0x3f) as usize],
                alphabet[((w >> 28) & 0x3f) as usize],
                alphabet[((w >> 22) & 0x3f) as usize],
                alphabet[((w >> 16) & 0x3f) as usize],
            ],
        );
        input_index += 6;
        output_index += 8;
    }

    while input_index + 3 <= input.len() {
        let s = &input[input_index..input_index + 3];
        let w = u32::from_be_bytes([0, s[0], s[1], s[2]]);
        backend::write_bytes(&mut output[output_index..], encode_quantum_3(alphabet, w));
        input_index += 3;
        output_index += 4;
    }

    match &input[input_index..] {
        [first] => {
            output[output_index].write(alphabet[usize::from(first >> 2)]);
            output[output_index + 1].write(alphabet[usize::from((first << 4) & 0x3f)]);
            output_index += 2;
            if PADDED {
                output[output_index].write(b'=');
                output[output_index + 1].write(b'=');
                output_index += 2;
            }
        }
        [first, second] => {
            output[output_index].write(alphabet[usize::from(first >> 2)]);
            output[output_index + 1]
                .write(alphabet[usize::from(((first << 4) | (second >> 4)) & 0x3f)]);
            output[output_index + 2].write(alphabet[usize::from((second << 2) & 0x3f)]);
            output_index += 3;
            if PADDED {
                output[output_index].write(b'=');
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
fn decode_chunk_8_byte(lut: &[u8; 256], s: &[u8; 8]) -> Option<[u8; 6]> {
    let v0 = lut[usize::from(s[0])];
    let v1 = lut[usize::from(s[1])];
    let v2 = lut[usize::from(s[2])];
    let v3 = lut[usize::from(s[3])];
    let v4 = lut[usize::from(s[4])];
    let v5 = lut[usize::from(s[5])];
    let v6 = lut[usize::from(s[6])];
    let v7 = lut[usize::from(s[7])];
    if (v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7) & 0x80 != 0 {
        return None;
    }
    let w = (u64::from(v0) << 42)
        | (u64::from(v1) << 36)
        | (u64::from(v2) << 30)
        | (u64::from(v3) << 24)
        | (u64::from(v4) << 18)
        | (u64::from(v5) << 12)
        | (u64::from(v6) << 6)
        | u64::from(v7);
    let b = w.to_be_bytes();
    Some([b[2], b[3], b[4], b[5], b[6], b[7]])
}

#[inline]
fn decode_chunk_4_byte(lut: &[u8; 256], s: &[u8; 4]) -> Option<[u8; 3]> {
    let v0 = lut[usize::from(s[0])];
    let v1 = lut[usize::from(s[1])];
    let v2 = lut[usize::from(s[2])];
    let v3 = lut[usize::from(s[3])];
    if (v0 | v1 | v2 | v3) & 0x80 != 0 {
        return None;
    }
    let w = (u32::from(v0) << 18) | (u32::from(v1) << 12) | (u32::from(v2) << 6) | u32::from(v3);
    let b = w.to_be_bytes();
    Some([b[1], b[2], b[3]])
}

#[inline]
fn decode_scalar_complete_from<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let byte_lut = if URL_SAFE_ALPHABET {
        &URL_DECODE_BYTE_LUT
    } else {
        &STD_DECODE_BYTE_LUT
    };

    if input_index + 16 <= input.len() {
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
                return Err(find_invalid_in_slice(s, input_index, byte_lut));
            };
            backend::write_bytes(&mut output[output_index..], decoded);
            input_index += 16;
            output_index += 12;
        }
    }

    if input_index + 8 <= input.len() {
        let s: &[u8; 8] = input[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let Some(decoded) = decode_chunk_8_byte(byte_lut, s) else {
            return Err(find_invalid_in_slice(s, input_index, byte_lut));
        };
        backend::write_bytes(&mut output[output_index..], decoded);
        input_index += 8;
        output_index += 6;
    }

    if input_index + 4 <= input.len() {
        let s: &[u8; 4] = input[input_index..input_index + 4]
            .try_into()
            .expect("4-byte slice fits in [u8; 4]");
        let Some(decoded) = decode_chunk_4_byte(byte_lut, s) else {
            return Err(find_invalid_in_slice(s, input_index, byte_lut));
        };
        backend::write_bytes(&mut output[output_index..], decoded);
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
    let byte_lut = if URL_SAFE_ALPHABET {
        &URL_DECODE_BYTE_LUT
    } else {
        &STD_DECODE_BYTE_LUT
    };

    if input_index + 16 <= complete_len {
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
                return Err(find_invalid_in_slice(&s, input_index, byte_lut));
            };
            buf[output_index..output_index + 12].copy_from_slice(&decoded);
            input_index += 16;
            output_index += 12;
        }
    }

    if input_index + 8 <= complete_len {
        let s: [u8; 8] = buf[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let Some(decoded) = decode_chunk_8_byte(byte_lut, &s) else {
            return Err(find_invalid_in_slice(&s, input_index, byte_lut));
        };
        buf[output_index..output_index + 6].copy_from_slice(&decoded);
        input_index += 8;
        output_index += 6;
    }

    if input_index + 4 <= complete_len {
        let s: [u8; 4] = buf[input_index..input_index + 4]
            .try_into()
            .expect("4-byte slice fits in [u8; 4]");
        let Some(decoded) = decode_chunk_4_byte(byte_lut, &s) else {
            return Err(find_invalid_in_slice(&s, input_index, byte_lut));
        };
        buf[output_index..output_index + 3].copy_from_slice(&decoded);
        output_index += 3;
    }

    Ok(output_index)
}

fn decode_tail<const URL_SAFE_ALPHABET: bool>(
    trailing: &[u8],
    base_index: usize,
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    let lut = if URL_SAFE_ALPHABET {
        &URL_DECODE_BYTE_LUT
    } else {
        &STD_DECODE_BYTE_LUT
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
            output[0].write((v0 << 2) | (v1 >> 4));
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
            output[0].write((v0 << 2) | (v1 >> 4));
            output[1].write((v1 << 4) | (v2 >> 2));
            Ok(2)
        }
        _ => unreachable!(),
    }
}

#[inline]
fn find_invalid_in_slice(slice: &[u8], base_index: usize, lut: &[u8; 256]) -> DecodeError {
    for (offset, &byte) in slice.iter().enumerate() {
        if lut[usize::from(byte)] == INVALID_SYMBOL {
            return backend::invalid_byte_error(byte, base_index + offset);
        }
    }
    unreachable!("called only when at least one byte in slice is invalid")
}

#[inline]
fn decode_value(byte: u8, index: usize, lut: &[u8; 256]) -> Result<u8, DecodeError> {
    let val = lut[usize::from(byte)];
    if val != INVALID_SYMBOL {
        Ok(val)
    } else {
        Err(backend::invalid_byte_error(byte, index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded_len<const PADDED: bool>(input: &[u8]) -> Result<usize, DecodeError> {
        decode_plan::<PADDED>(input).map(|plan| plan.required)
    }

    fn encode_init<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> usize {
        encode::<URL_SAFE_ALPHABET, PADDED>(input, codec::as_uninit_slice(output))
    }

    fn decode_init<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan::<PADDED>(input)?;
        decode_with_plan::<URL_SAFE_ALPHABET>(input, plan, codec::as_uninit_slice(output))
    }

    fn encode_scalar<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> usize {
        encode_scalar_from::<URL_SAFE_ALPHABET, PADDED>(input, codec::as_uninit_slice(output), 0, 0)
    }

    fn decode_scalar<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan::<PADDED>(input)?;
        if plan.data_len == 0 {
            return Ok(0);
        }
        decode_from::<URL_SAFE_ALPHABET>(
            input,
            codec::as_uninit_slice(&mut output[..plan.required]),
            plan.data_len,
            plan.complete_len,
            0,
            0,
            false,
        )
    }

    fn decode_in_place_scalar<const URL_SAFE_ALPHABET: bool, const PADDED: bool>(
        buf: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan::<PADDED>(buf)?;
        let written = decode_in_place_from::<URL_SAFE_ALPHABET>(
            buf,
            plan.data_len,
            plan.complete_len,
            0,
            0,
            false,
        )?;
        debug_assert_eq!(written, plan.required);
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
                encode_init::<URL_SAFE_ALPHABET, PADDED>(raw, enc),
                expected_enc_len
            );
            assert_eq!(
                encode_scalar::<URL_SAFE_ALPHABET, PADDED>(raw, scalar_enc),
                expected_enc_len
            );
            assert_eq!(enc, scalar_enc, "encode mismatch at len={len}");

            let dec = &mut dec_buf[..len];
            let scalar_dec = &mut scalar_dec_buf[..len];
            assert_eq!(decode_init::<URL_SAFE_ALPHABET, PADDED>(enc, dec), Ok(len));
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
        assert_eq!(decode_init::<false, true>(&std_sym, &mut out), Ok(96));
        assert_eq!(
            decode_scalar::<false, true>(&std_sym, &mut scalar_out),
            Ok(96)
        );
        assert_eq!(out, scalar_out);

        let mut reencoded = [0_u8; 128];
        assert_eq!(encode_init::<false, true>(&out, &mut reencoded), 128);
        assert_eq!(reencoded, std_sym);

        let url_sym = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_\
                        ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        assert_eq!(decode_init::<true, false>(&url_sym, &mut out), Ok(96));
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
            encode_init::<URL_SAFE_ALPHABET, PADDED>(raw, clean_enc);

            for pos in 0..enc_len {
                for byte in 0_u8..=u8::MAX {
                    let mutated = &mut mutated_buf[..enc_len];
                    mutated.copy_from_slice(clean_enc);
                    mutated[pos] = byte;

                    let out_cap = max_decoded_len(enc_len);
                    let actual =
                        decode_init::<URL_SAFE_ALPHABET, PADDED>(mutated, &mut dec_buf[..out_cap]);
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
                decode_init::<false, true>(&buf, &mut out),
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
                decode_init::<true, false>(&buf, &mut out),
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
            decode_init::<false, true>(&cross, &mut out),
            Err(DecodeError::InvalidByte {
                index: 65,
                byte: b'-'
            })
        );
        cross[65] = b'+';
        assert_eq!(
            decode_init::<true, true>(&cross, &mut out),
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
            decode_init::<false, true>(&tail2, &mut out),
            Err(DecodeError::NonCanonical { index: 65 })
        );
        assert_eq!(
            decode_init::<false, false>(&tail2[..66], &mut out),
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
            decode_init::<false, true>(&tail3, &mut out),
            Err(DecodeError::NonCanonical { index: 66 })
        );
        assert_eq!(
            decode_init::<false, false>(&tail3[..67], &mut out),
            Err(DecodeError::NonCanonical { index: 66 })
        );
        assert_eq!(
            decode_in_place::<false, true>(&mut tail3),
            Err(DecodeError::NonCanonical { index: 66 })
        );
    }

    #[test]
    fn bounded_wide_store_and_chunk_error_offsets() {
        fn check_bounds<const URL_SAFE_ALPHABET: bool, const PADDED: bool>() {
            let mut raw_buf = [0_u8; 96];
            let mut enc_buf = [0_u8; 136];
            let mut sentinel = [0_u8; 160];

            for raw_len in 0..=96 {
                let raw = &mut raw_buf[..raw_len];
                fill_pattern(raw, raw_len + 53);
                let enc_len = encoded_len::<PADDED>(raw_len).expect("within bounds");
                let enc = &mut enc_buf[..enc_len];
                encode_init::<URL_SAFE_ALPHABET, PADDED>(raw, enc);

                let plan = decode_plan::<PADDED>(enc).unwrap();
                assert_eq!(plan.required, raw_len);

                sentinel.fill(0xa5);
                let written = decode_with_plan::<URL_SAFE_ALPHABET>(
                    enc,
                    plan,
                    codec::as_uninit_slice(&mut sentinel[..raw_len]),
                )
                .unwrap();
                assert_eq!(written, raw_len);
                assert_eq!(&sentinel[..raw_len], raw);
                assert!(sentinel[raw_len..].iter().all(|&b| b == 0xa5));

                sentinel.fill(0x5a);
                let written_oversized = decode_with_plan::<URL_SAFE_ALPHABET>(
                    enc,
                    plan,
                    codec::as_uninit_slice(&mut sentinel),
                )
                .unwrap();
                assert_eq!(written_oversized, raw_len);
                assert_eq!(&sentinel[..raw_len], raw);
                assert!(sentinel[raw_len..].iter().all(|&b| b == 0x5a));
            }
        }

        // Cover 12B/16B, 24B/28B AVX2, and 64B NEON + 16B tail wide-store thresholds.
        check_bounds::<false, true>();
        check_bounds::<true, false>();

        // Verify every offset across a 64B NEON / 2x32B AVX2 chunk + 16B tail (including byte 63
        // inside the first 64B chunk and trailing bytes in the 16B/4B remainder).
        let mut raw_64 = [0_u8; 64];
        fill_pattern(&mut raw_64, 91);
        let mut clean_enc = [0_u8; 88];
        encode_init::<false, true>(&raw_64, &mut clean_enc);
        let mut out = [0_u8; 64];
        for pos in 0..86 {
            let mut mutated = clean_enc;
            mutated[pos] = b'!';
            assert_eq!(
                decode_init::<false, true>(&mutated, &mut out),
                Err(DecodeError::InvalidByte {
                    index: pos,
                    byte: b'!'
                }),
                "out-of-place mismatch at pos={pos}"
            );
            assert_eq!(
                decode_in_place::<false, true>(&mut mutated),
                Err(DecodeError::InvalidByte {
                    index: pos,
                    byte: b'!'
                }),
                "in-place mismatch at pos={pos}"
            );
        }
    }
}
