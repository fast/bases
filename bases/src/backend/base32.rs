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

//! Base32 execution backend.

use core::mem::MaybeUninit;

use crate::DecodeError;
use crate::backend;
use crate::backend::DecodePlan;
use crate::codec;

#[allow(unsafe_code)]
mod arch;

const SCALAR_BULK_QUANTA: usize = 32;

const STANDARD_UPPER: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const STANDARD_LOWER: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
const HEX_UPPER: &[u8; 32] = b"0123456789ABCDEFGHIJKLMNOPQRSTUV";
const HEX_LOWER: &[u8; 32] = b"0123456789abcdefghijklmnopqrstuv";

const STD_DECODE_MIXED: [u8; 256] = make_decode_lut(false, true, true);
const STD_DECODE_UPPER: [u8; 256] = make_decode_lut(false, true, false);
const STD_DECODE_LOWER: [u8; 256] = make_decode_lut(false, false, true);
const HEX_DECODE_MIXED: [u8; 256] = make_decode_lut(true, true, true);
const HEX_DECODE_UPPER: [u8; 256] = make_decode_lut(true, true, false);
const HEX_DECODE_LOWER: [u8; 256] = make_decode_lut(true, false, true);

const fn make_decode_lut(extended_hex: bool, allow_upper: bool, allow_lower: bool) -> [u8; 256] {
    let mut lut = [0xff_u8; 256];
    if extended_hex {
        let mut byte = b'0';
        while byte <= b'9' {
            lut[byte as usize] = byte - b'0';
            byte += 1;
        }
        if allow_upper {
            let mut byte = b'A';
            while byte <= b'V' {
                lut[byte as usize] = byte - b'A' + 10;
                byte += 1;
            }
        }
        if allow_lower {
            let mut byte = b'a';
            while byte <= b'v' {
                lut[byte as usize] = byte - b'a' + 10;
                byte += 1;
            }
        }
    } else {
        let mut byte = b'2';
        while byte <= b'7' {
            lut[byte as usize] = byte - b'2' + 26;
            byte += 1;
        }
        if allow_upper {
            let mut byte = b'A';
            while byte <= b'Z' {
                lut[byte as usize] = byte - b'A';
                byte += 1;
            }
        }
        if allow_lower {
            let mut byte = b'a';
            while byte <= b'z' {
                lut[byte as usize] = byte - b'a';
                byte += 1;
            }
        }
    }
    lut
}

const fn select_alphabet<const EXTENDED_HEX: bool, const UPPERCASE: bool>() -> &'static [u8; 32] {
    match (EXTENDED_HEX, UPPERCASE) {
        (false, true) => STANDARD_UPPER,
        (false, false) => STANDARD_LOWER,
        (true, true) => HEX_UPPER,
        (true, false) => HEX_LOWER,
    }
}

const fn select_decode_lut<
    const EXTENDED_HEX: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>() -> &'static [u8; 256] {
    if !STRICT_CASE {
        if EXTENDED_HEX {
            &HEX_DECODE_MIXED
        } else {
            &STD_DECODE_MIXED
        }
    } else if EXTENDED_HEX {
        if UPPERCASE {
            &HEX_DECODE_UPPER
        } else {
            &HEX_DECODE_LOWER
        }
    } else if UPPERCASE {
        &STD_DECODE_UPPER
    } else {
        &STD_DECODE_LOWER
    }
}

/// Computes the exact Base32 encoded length for `input_len`.
#[inline]
pub const fn encoded_len<const PADDED: bool>(input_len: usize) -> Option<usize> {
    let complete = match (input_len / 5).checked_mul(8) {
        Some(length) => length,
        None => return None,
    };
    let trailing = match (input_len % 5, PADDED) {
        (0, _) => 0,
        (_, true) => 8,
        (1, false) => 2,
        (2, false) => 4,
        (3, false) => 5,
        (4, false) => 7,
        _ => 0,
    };
    complete.checked_add(trailing)
}

/// Computes an upper bound on the Base32 decoded length for `input_len`.
#[inline]
pub const fn max_decoded_len(input_len: usize) -> usize {
    let complete = (input_len / 8) * 5;
    let trailing = (input_len % 8) * 5 / 8;
    complete + trailing
}

/// Validates Base32 input length and padding shape, returning the exact decode plan.
#[inline]
pub fn decode_plan<const PADDED: bool>(input: &[u8]) -> Result<DecodePlan, DecodeError> {
    if PADDED {
        if input.len() % 8 != 0 {
            return Err(DecodeError::InvalidLength {
                length: input.len(),
            });
        }
        if input.is_empty() {
            return Ok(DecodePlan {
                required: 0,
                data_len: 0,
                complete_len: 0,
                padding: 0,
            });
        }

        let padding = if input[input.len() - 1] == b'=' {
            backend::trailing_padding(input)
        } else {
            0
        };
        let reduction = match padding {
            0 => 0,
            1 => 1,
            3 => 2,
            4 => 3,
            6 => 4,
            _ => {
                return Err(DecodeError::InvalidPadding {
                    index: input.len() - padding,
                });
            }
        };
        let data_len = input.len() - padding;
        let complete_len = (data_len / 8) * 8;
        Ok(DecodePlan {
            required: input.len() / 8 * 5 - reduction,
            data_len,
            complete_len,
            padding,
        })
    } else {
        let trailing = match input.len() % 8 {
            0 => 0,
            2 => 1,
            4 => 2,
            5 => 3,
            7 => 4,
            _ => {
                return Err(DecodeError::InvalidLength {
                    length: input.len(),
                });
            }
        };
        let complete_len = (input.len() / 8) * 8;
        Ok(DecodePlan {
            required: input.len() / 8 * 5 + trailing,
            data_len: input.len(),
            complete_len,
            padding: 0,
        })
    }
}

/// Encodes `input` into `output` using Base32, returning the number of bytes written.
#[inline]
pub fn encode<const EXTENDED_HEX: bool, const PADDED: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> usize {
    let required = encoded_len::<PADDED>(input.len()).expect("encoded length overflow");
    assert!(output.len() >= required, "output buffer too small");

    let (input_index, output_index) = match arch::encode::<EXTENDED_HEX, UPPERCASE>(input, output) {
        Some(progress) => progress,
        None if input.len() >= SCALAR_BULK_QUANTA * 5 => {
            return encode_scalar_bulk::<EXTENDED_HEX, PADDED, UPPERCASE>(input, output);
        }
        None => (0, 0),
    };
    encode_scalar_from::<EXTENDED_HEX, PADDED, UPPERCASE>(input, output, input_index, output_index)
}

/// Encodes a single 5-byte Base32 quantum into `output`.
#[inline]
pub fn encode_quantum<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> usize {
    let chunk: &[u8; 5] = input.try_into().expect("5-byte slice fits in [u8; 5]");
    debug_assert!(output.len() >= 8);
    let alphabet = select_alphabet::<EXTENDED_HEX, UPPERCASE>();
    backend::write_bytes(output, encode_quantum_5(alphabet, chunk));
    8
}

/// Decodes Base32 `input` into `output` using `plan`, returning the number of bytes written.
#[inline]
pub fn decode_with_plan<
    const EXTENDED_HEX: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>(
    input: &[u8],
    plan: DecodePlan,
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    if plan.data_len == 0 {
        return Ok(0);
    }
    assert!(output.len() >= plan.required, "output buffer too small");
    let out = &mut output[..plan.required];

    let (input_index, output_index) = match arch::decode::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
        &input[..plan.complete_len],
        out,
    ) {
        Some(progress) => progress,
        None if plan.complete_len >= SCALAR_BULK_QUANTA * 8 => {
            return decode_scalar_bulk::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(input, plan, out);
        }
        None => (0, 0),
    };
    decode_from::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
        input,
        out,
        plan.data_len,
        plan.complete_len,
        input_index,
        output_index,
    )
}

/// Decodes a single complete 8-byte Base32 quantum into `output`.
#[inline]
pub fn decode_quantum<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();
    let chunk: &[u8; 8] = input.try_into().expect("8-byte slice fits in [u8; 8]");
    debug_assert!(output.len() >= 5);
    let Some(decoded) = decode_chunk_8(lut, chunk) else {
        return Err(find_invalid_in_quantum(chunk, 0, lut));
    };
    backend::write_bytes(output, decoded);
    Ok(5)
}

#[inline]
fn decode_from<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    data_len: usize,
    complete_len: usize,
    input_index: usize,
    output_index: usize,
) -> Result<usize, DecodeError> {
    let mut output_index = decode_scalar_complete_from::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
        &input[..complete_len],
        output,
        input_index,
        output_index,
    )?;

    if complete_len != data_len {
        output_index += decode_tail::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            &input[complete_len..data_len],
            complete_len,
            &mut output[output_index..],
        )?;
    }

    Ok(output_index)
}

/// Decodes Base32 `buf` in place, returning the number of decoded bytes.
#[inline]
pub fn decode_in_place<
    const EXTENDED_HEX: bool,
    const PADDED: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>(
    buf: &mut [u8],
) -> Result<usize, DecodeError> {
    let plan = decode_plan::<PADDED>(buf)?;

    let (input_index, output_index) = arch::decode_in_place::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
        buf,
        plan.complete_len,
        plan.required,
    );
    let written = decode_in_place_from::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
        buf,
        plan.data_len,
        plan.complete_len,
        input_index,
        output_index,
    )?;

    debug_assert_eq!(written, plan.required);
    Ok(written)
}

#[inline]
fn decode_in_place_from<
    const EXTENDED_HEX: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>(
    buf: &mut [u8],
    data_len: usize,
    complete_len: usize,
    input_index: usize,
    output_index: usize,
) -> Result<usize, DecodeError> {
    let mut output_index = decode_in_place_scalar_complete_from::<
        EXTENDED_HEX,
        UPPERCASE,
        STRICT_CASE,
    >(buf, complete_len, input_index, output_index)?;

    if complete_len != data_len {
        let mut tail = [0_u8; 7];
        let tail_len = data_len - complete_len;
        tail[..tail_len].copy_from_slice(&buf[complete_len..data_len]);
        output_index += decode_tail::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            &tail[..tail_len],
            complete_len,
            codec::as_uninit_slice(&mut buf[output_index..]),
        )?;
    }

    Ok(output_index)
}

#[inline]
fn encode_quantum_5(alphabet: &[u8; 32], b: &[u8; 5]) -> [u8; 8] {
    let w = u64::from_be_bytes([0, 0, 0, b[0], b[1], b[2], b[3], b[4]]);
    [
        alphabet[((w >> 35) & 0x1f) as usize],
        alphabet[((w >> 30) & 0x1f) as usize],
        alphabet[((w >> 25) & 0x1f) as usize],
        alphabet[((w >> 20) & 0x1f) as usize],
        alphabet[((w >> 15) & 0x1f) as usize],
        alphabet[((w >> 10) & 0x1f) as usize],
        alphabet[((w >> 5) & 0x1f) as usize],
        alphabet[(w & 0x1f) as usize],
    ]
}

#[inline(never)]
fn encode_scalar_bulk<const EXTENDED_HEX: bool, const PADDED: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> usize {
    let alphabet = select_alphabet::<EXTENDED_HEX, UPPERCASE>();
    let (quanta, _) = input.as_chunks::<5>();
    let complete_output_len = quanta.len() * 8;
    let complete_output = &mut output[..complete_output_len];
    for (src, dst) in quanta.iter().zip(complete_output.as_chunks_mut::<8>().0) {
        backend::write_bytes(dst, encode_quantum_5(alphabet, src));
    }
    encode_scalar_from::<EXTENDED_HEX, PADDED, UPPERCASE>(
        input,
        output,
        quanta.len() * 5,
        complete_output_len,
    )
}

fn encode_scalar_from<const EXTENDED_HEX: bool, const PADDED: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    mut input_index: usize,
    mut output_index: usize,
) -> usize {
    let alphabet = select_alphabet::<EXTENDED_HEX, UPPERCASE>();

    while input_index + 5 <= input.len() {
        let b: &[u8; 5] = input[input_index..input_index + 5]
            .try_into()
            .expect("5-byte slice fits in [u8; 5]");
        backend::write_bytes(&mut output[output_index..], encode_quantum_5(alphabet, b));
        input_index += 5;
        output_index += 8;
    }

    let tail_len = input.len() - input_index;
    if tail_len != 0 {
        let rem = &input[input_index..];
        let (written_syms, pad_len) = match rem {
            [b0] => {
                output[output_index].write(alphabet[usize::from(b0 >> 3)]);
                output[output_index + 1].write(alphabet[usize::from((b0 << 2) & 0x1f)]);
                (2, 6)
            }
            [b0, b1] => {
                output[output_index].write(alphabet[usize::from(b0 >> 3)]);
                output[output_index + 1]
                    .write(alphabet[usize::from(((b0 << 2) | (b1 >> 6)) & 0x1f)]);
                output[output_index + 2].write(alphabet[usize::from((b1 >> 1) & 0x1f)]);
                output[output_index + 3].write(alphabet[usize::from((b1 << 4) & 0x1f)]);
                (4, 4)
            }
            [b0, b1, b2] => {
                output[output_index].write(alphabet[usize::from(b0 >> 3)]);
                output[output_index + 1]
                    .write(alphabet[usize::from(((b0 << 2) | (b1 >> 6)) & 0x1f)]);
                output[output_index + 2].write(alphabet[usize::from((b1 >> 1) & 0x1f)]);
                output[output_index + 3]
                    .write(alphabet[usize::from(((b1 << 4) | (b2 >> 4)) & 0x1f)]);
                output[output_index + 4].write(alphabet[usize::from((b2 << 1) & 0x1f)]);
                (5, 3)
            }
            [b0, b1, b2, b3] => {
                output[output_index].write(alphabet[usize::from(b0 >> 3)]);
                output[output_index + 1]
                    .write(alphabet[usize::from(((b0 << 2) | (b1 >> 6)) & 0x1f)]);
                output[output_index + 2].write(alphabet[usize::from((b1 >> 1) & 0x1f)]);
                output[output_index + 3]
                    .write(alphabet[usize::from(((b1 << 4) | (b2 >> 4)) & 0x1f)]);
                output[output_index + 4]
                    .write(alphabet[usize::from(((b2 << 1) | (b3 >> 7)) & 0x1f)]);
                output[output_index + 5].write(alphabet[usize::from((b3 >> 2) & 0x1f)]);
                output[output_index + 6].write(alphabet[usize::from((b3 << 3) & 0x1f)]);
                (7, 1)
            }
            _ => unreachable!(),
        };
        output_index += written_syms;
        if PADDED {
            output[output_index..output_index + pad_len].fill(MaybeUninit::new(b'='));
            output_index += pad_len;
        }
    }

    output_index
}

#[inline]
fn decode_chunk_8(lut: &[u8; 256], s: &[u8; 8]) -> Option<[u8; 5]> {
    let v0 = lut[usize::from(s[0])];
    let v1 = lut[usize::from(s[1])];
    let v2 = lut[usize::from(s[2])];
    let v3 = lut[usize::from(s[3])];
    let v4 = lut[usize::from(s[4])];
    let v5 = lut[usize::from(s[5])];
    let v6 = lut[usize::from(s[6])];
    let v7 = lut[usize::from(s[7])];
    if (v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7) & 0xe0 != 0 {
        return None;
    }
    let packed = (u64::from(v0) << 35)
        | (u64::from(v1) << 30)
        | (u64::from(v2) << 25)
        | (u64::from(v3) << 20)
        | (u64::from(v4) << 15)
        | (u64::from(v5) << 10)
        | (u64::from(v6) << 5)
        | u64::from(v7);
    let [b0, b1, b2, b3] = ((packed >> 8) as u32).to_be_bytes();
    Some([b0, b1, b2, b3, packed as u8])
}

#[inline(never)]
fn decode_scalar_bulk<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    plan: DecodePlan,
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();
    let (quanta, _) = input[..plan.complete_len].as_chunks::<8>();
    let complete_output_len = quanta.len() * 5;
    let complete_output = &mut output[..complete_output_len];
    for (chunk_index, (src, dst)) in quanta
        .iter()
        .zip(complete_output.as_chunks_mut::<5>().0)
        .enumerate()
    {
        let Some(decoded) = decode_chunk_8(lut, src) else {
            return Err(find_invalid_in_quantum(src, chunk_index * 8, lut));
        };
        backend::write_bytes(dst, decoded);
    }
    decode_from::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
        input,
        output,
        plan.data_len,
        plan.complete_len,
        plan.complete_len,
        complete_output_len,
    )
}

fn decode_scalar_complete_from<
    const EXTENDED_HEX: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();

    while input_index + 8 <= input.len() {
        let s: &[u8; 8] = input[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let Some(decoded) = decode_chunk_8(lut, s) else {
            return Err(find_invalid_in_quantum(s, input_index, lut));
        };
        backend::write_bytes(&mut output[output_index..], decoded);
        input_index += 8;
        output_index += 5;
    }

    Ok(output_index)
}

fn decode_in_place_scalar_complete_from<
    const EXTENDED_HEX: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>(
    buf: &mut [u8],
    complete_len: usize,
    mut input_index: usize,
    mut output_index: usize,
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();

    while input_index + 8 <= complete_len {
        let s: [u8; 8] = buf[input_index..input_index + 8]
            .try_into()
            .expect("8-byte slice fits in [u8; 8]");
        let Some(decoded) = decode_chunk_8(lut, &s) else {
            return Err(find_invalid_in_quantum(&s, input_index, lut));
        };
        buf[output_index..output_index + 5].copy_from_slice(&decoded);
        input_index += 8;
        output_index += 5;
    }

    Ok(output_index)
}

fn decode_tail<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    trailing: &[u8],
    base_index: usize,
    output: &mut [MaybeUninit<u8>],
) -> Result<usize, DecodeError> {
    let lut = select_decode_lut::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();
    match trailing {
        [b0, b1] => {
            let v0 = decode_value(*b0, base_index, lut)?;
            let v1 = decode_value(*b1, base_index + 1, lut)?;
            if v1 & 0x03 != 0 {
                return Err(DecodeError::NonCanonical {
                    index: base_index + 1,
                });
            }
            output[0].write((v0 << 3) | (v1 >> 2));
            Ok(1)
        }
        [b0, b1, b2, b3] => {
            let v0 = decode_value(*b0, base_index, lut)?;
            let v1 = decode_value(*b1, base_index + 1, lut)?;
            let v2 = decode_value(*b2, base_index + 2, lut)?;
            let v3 = decode_value(*b3, base_index + 3, lut)?;
            if v3 & 0x0f != 0 {
                return Err(DecodeError::NonCanonical {
                    index: base_index + 3,
                });
            }
            output[0].write((v0 << 3) | (v1 >> 2));
            output[1].write((v1 << 6) | (v2 << 1) | (v3 >> 4));
            Ok(2)
        }
        [b0, b1, b2, b3, b4] => {
            let v0 = decode_value(*b0, base_index, lut)?;
            let v1 = decode_value(*b1, base_index + 1, lut)?;
            let v2 = decode_value(*b2, base_index + 2, lut)?;
            let v3 = decode_value(*b3, base_index + 3, lut)?;
            let v4 = decode_value(*b4, base_index + 4, lut)?;
            if v4 & 0x01 != 0 {
                return Err(DecodeError::NonCanonical {
                    index: base_index + 4,
                });
            }
            output[0].write((v0 << 3) | (v1 >> 2));
            output[1].write((v1 << 6) | (v2 << 1) | (v3 >> 4));
            output[2].write((v3 << 4) | (v4 >> 1));
            Ok(3)
        }
        [b0, b1, b2, b3, b4, b5, b6] => {
            let v0 = decode_value(*b0, base_index, lut)?;
            let v1 = decode_value(*b1, base_index + 1, lut)?;
            let v2 = decode_value(*b2, base_index + 2, lut)?;
            let v3 = decode_value(*b3, base_index + 3, lut)?;
            let v4 = decode_value(*b4, base_index + 4, lut)?;
            let v5 = decode_value(*b5, base_index + 5, lut)?;
            let v6 = decode_value(*b6, base_index + 6, lut)?;
            if v6 & 0x07 != 0 {
                return Err(DecodeError::NonCanonical {
                    index: base_index + 6,
                });
            }
            output[0].write((v0 << 3) | (v1 >> 2));
            output[1].write((v1 << 6) | (v2 << 1) | (v3 >> 4));
            output[2].write((v3 << 4) | (v4 >> 1));
            output[3].write((v4 << 7) | (v5 << 2) | (v6 >> 3));
            Ok(4)
        }
        _ => unreachable!(),
    }
}

fn find_invalid_in_quantum(quantum: &[u8], base_index: usize, lut: &[u8; 256]) -> DecodeError {
    for (offset, &byte) in quantum.iter().enumerate() {
        if lut[usize::from(byte)] & 0xe0 != 0 {
            return backend::invalid_byte_error(byte, base_index + offset);
        }
    }
    unreachable!("called only when at least one byte in quantum is invalid")
}

#[inline]
fn decode_value(byte: u8, index: usize, lut: &[u8; 256]) -> Result<u8, DecodeError> {
    let value = lut[usize::from(byte)];
    if value & 0xe0 == 0 {
        Ok(value)
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

    fn encode_init<const EXTENDED_HEX: bool, const PADDED: bool, const UPPERCASE: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> usize {
        encode::<EXTENDED_HEX, PADDED, UPPERCASE>(input, codec::as_uninit_slice(output))
    }

    fn decode_init<
        const EXTENDED_HEX: bool,
        const PADDED: bool,
        const UPPERCASE: bool,
        const STRICT_CASE: bool,
    >(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan::<PADDED>(input)?;
        decode_with_plan::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            input,
            plan,
            codec::as_uninit_slice(output),
        )
    }

    fn encode_scalar<const EXTENDED_HEX: bool, const PADDED: bool, const UPPERCASE: bool>(
        input: &[u8],
        output: &mut [u8],
    ) -> usize {
        encode_scalar_from::<EXTENDED_HEX, PADDED, UPPERCASE>(
            input,
            codec::as_uninit_slice(output),
            0,
            0,
        )
    }

    fn decode_scalar<
        const EXTENDED_HEX: bool,
        const PADDED: bool,
        const UPPERCASE: bool,
        const STRICT_CASE: bool,
    >(
        input: &[u8],
        output: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan::<PADDED>(input)?;
        if plan.data_len == 0 {
            return Ok(0);
        }
        decode_from::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            input,
            codec::as_uninit_slice(&mut output[..plan.required]),
            plan.data_len,
            plan.complete_len,
            0,
            0,
        )
    }

    fn decode_in_place_scalar<
        const EXTENDED_HEX: bool,
        const PADDED: bool,
        const UPPERCASE: bool,
        const STRICT_CASE: bool,
    >(
        buf: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let plan = decode_plan::<PADDED>(buf)?;
        let written = decode_in_place_from::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            buf,
            plan.data_len,
            plan.complete_len,
            0,
            0,
        )?;
        debug_assert_eq!(written, plan.required);
        Ok(written)
    }

    fn fill_pattern(buf: &mut [u8], seed: usize) {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = ((i.wrapping_mul(173).wrapping_add(seed.wrapping_mul(29))) & 0xff) as u8;
        }
    }

    #[test]
    fn length_calculations_and_bounds() {
        assert_eq!(encoded_len::<true>(0), Some(0));
        assert_eq!(encoded_len::<false>(0), Some(0));
        assert_eq!(encoded_len::<true>(1), Some(8));
        assert_eq!(encoded_len::<false>(1), Some(2));
        assert_eq!(encoded_len::<false>(2), Some(4));
        assert_eq!(encoded_len::<false>(3), Some(5));
        assert_eq!(encoded_len::<false>(4), Some(7));
        assert_eq!(encoded_len::<false>(5), Some(8));

        let max_quanta = usize::MAX / 8;
        assert_eq!(encoded_len::<true>(max_quanta * 5), Some(max_quanta * 8));
        assert_eq!(encoded_len::<true>(max_quanta * 5 + 1), None);

        assert_eq!(max_decoded_len(0), 0);
        assert_eq!(max_decoded_len(7), 4);
        assert_eq!(max_decoded_len(8), 5);
        assert_eq!(max_decoded_len(usize::MAX), (usize::MAX / 8) * 5 + 4);

        assert_eq!(decoded_len::<true>(b""), Ok(0));
        assert_eq!(decoded_len::<true>(b"AA======"), Ok(1));
        assert_eq!(decoded_len::<true>(b"AAAA===="), Ok(2));
        assert_eq!(decoded_len::<true>(b"AAAAA==="), Ok(3));
        assert_eq!(decoded_len::<true>(b"AAAAAAA="), Ok(4));
        assert_eq!(decoded_len::<true>(b"AAAAAAAA"), Ok(5));
        assert_eq!(
            decoded_len::<true>(b"A======="),
            Err(DecodeError::InvalidPadding { index: 1 })
        );
        assert_eq!(
            decoded_len::<true>(b"========"),
            Err(DecodeError::InvalidPadding { index: 0 })
        );
        assert_eq!(
            decoded_len::<true>(b"AAA====="),
            Err(DecodeError::InvalidPadding { index: 3 })
        );
        assert_eq!(
            decoded_len::<true>(b"AAAAAA=="),
            Err(DecodeError::InvalidPadding { index: 6 })
        );
        assert_eq!(
            decoded_len::<true>(b"AAAAAAA"),
            Err(DecodeError::InvalidLength { length: 7 })
        );

        for bad_rem in [1, 3, 6] {
            let buf = [b'A'; 16];
            assert_eq!(
                decoded_len::<false>(&buf[..8 + bad_rem]),
                Err(DecodeError::InvalidLength {
                    length: 8 + bad_rem
                })
            );
        }
    }

    fn check_roundtrip_and_arch_parity<
        const EXTENDED_HEX: bool,
        const PADDED: bool,
        const UPPERCASE: bool,
        const STRICT_CASE: bool,
    >() {
        let mut raw_buf = [0_u8; 260];
        let mut enc_buf = [0_u8; 424];
        let mut scalar_enc_buf = [0_u8; 424];
        let mut dec_buf = [0_u8; 260];
        let mut scalar_dec_buf = [0_u8; 260];
        let mut ip_buf = [0_u8; 424];
        let mut sip_buf = [0_u8; 424];

        for len in 0..=260 {
            let raw = &mut raw_buf[..len];
            fill_pattern(raw, len);

            let expected_enc_len = encoded_len::<PADDED>(len).expect("within bounds");
            let enc = &mut enc_buf[..expected_enc_len];
            let scalar_enc = &mut scalar_enc_buf[..expected_enc_len];

            assert_eq!(
                encode_init::<EXTENDED_HEX, PADDED, UPPERCASE>(raw, enc),
                expected_enc_len
            );
            assert_eq!(
                encode_scalar::<EXTENDED_HEX, PADDED, UPPERCASE>(raw, scalar_enc),
                expected_enc_len
            );
            assert_eq!(enc, scalar_enc, "encode mismatch at len={len}");

            let dec = &mut dec_buf[..len];
            let scalar_dec = &mut scalar_dec_buf[..len];
            assert_eq!(
                decode_init::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(enc, dec),
                Ok(len)
            );
            assert_eq!(
                decode_scalar::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(enc, scalar_dec),
                Ok(len)
            );
            assert_eq!(dec, raw);
            assert_eq!(dec, scalar_dec);

            let ip = &mut ip_buf[..expected_enc_len];
            let sip = &mut sip_buf[..expected_enc_len];
            ip.copy_from_slice(enc);
            sip.copy_from_slice(enc);
            assert_eq!(
                decode_in_place::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(ip),
                Ok(len)
            );
            assert_eq!(
                decode_in_place_scalar::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(sip),
                Ok(len)
            );
            assert_eq!(&ip[..len], raw);
            assert_eq!(&sip[..len], raw);

            if !STRICT_CASE && !enc.is_empty() {
                for (idx, b) in enc.iter_mut().enumerate() {
                    if idx % 2 == 0 {
                        if b.is_ascii_uppercase() {
                            *b = b.to_ascii_lowercase();
                        } else if b.is_ascii_lowercase() {
                            *b = b.to_ascii_uppercase();
                        }
                    }
                }
                assert_eq!(
                    decode_init::<EXTENDED_HEX, PADDED, UPPERCASE, false>(enc, dec),
                    Ok(len)
                );
                assert_eq!(dec, raw);
                ip.copy_from_slice(enc);
                assert_eq!(
                    decode_in_place::<EXTENDED_HEX, PADDED, UPPERCASE, false>(ip),
                    Ok(len)
                );
                assert_eq!(&ip[..len], raw);
            }
        }
    }

    #[test]
    fn roundtrip_and_arch_parity_all_profiles_0_to_260() {
        check_roundtrip_and_arch_parity::<false, true, true, true>();
        check_roundtrip_and_arch_parity::<false, true, true, false>();
        check_roundtrip_and_arch_parity::<false, true, false, true>();
        check_roundtrip_and_arch_parity::<false, true, false, false>();
        check_roundtrip_and_arch_parity::<false, false, true, true>();
        check_roundtrip_and_arch_parity::<false, false, true, false>();
        check_roundtrip_and_arch_parity::<false, false, false, true>();
        check_roundtrip_and_arch_parity::<false, false, false, false>();

        check_roundtrip_and_arch_parity::<true, true, true, true>();
        check_roundtrip_and_arch_parity::<true, true, true, false>();
        check_roundtrip_and_arch_parity::<true, true, false, true>();
        check_roundtrip_and_arch_parity::<true, true, false, false>();
        check_roundtrip_and_arch_parity::<true, false, true, true>();
        check_roundtrip_and_arch_parity::<true, false, true, false>();
        check_roundtrip_and_arch_parity::<true, false, false, true>();
        check_roundtrip_and_arch_parity::<true, false, false, false>();
    }

    #[test]
    fn scalar_bulk_reports_global_error_offsets() {
        // 32 complete quanta followed by the RFC 4648 encoding of "f".
        let mut input = [b'A'; 264];
        input[256..].copy_from_slice(b"MY======");
        let plan = decode_plan::<true>(&input).unwrap();
        let mut output = [MaybeUninit::uninit(); 161];

        input[255] = b'!';
        assert_eq!(
            decode_scalar_bulk::<false, true, true>(&input, plan, &mut output),
            Err(DecodeError::InvalidByte {
                index: 255,
                byte: b'!',
            })
        );

        input[255] = b'A';
        input[257] = b'Z';
        assert_eq!(
            decode_scalar_bulk::<false, true, true>(&input, plan, &mut output),
            Err(DecodeError::NonCanonical { index: 257 })
        );
    }

    fn verify_all_bytes_at_offsets<
        const EXTENDED_HEX: bool,
        const PADDED: bool,
        const UPPERCASE: bool,
        const STRICT_CASE: bool,
    >(
        raw_lens: &[usize],
    ) {
        let mut raw_buf = [0_u8; 64];
        let mut clean_enc_buf = [0_u8; 104];
        let mut mutated_buf = [0_u8; 104];
        let mut dec_buf = [0_u8; 64];
        let mut scalar_dec_buf = [0_u8; 64];
        let mut sip_buf = [0_u8; 104];

        for &raw_len in raw_lens {
            let raw = &mut raw_buf[..raw_len];
            fill_pattern(raw, raw_len + 11);
            let enc_len = encoded_len::<PADDED>(raw_len).expect("within bounds");
            let clean_enc = &mut clean_enc_buf[..enc_len];
            encode_init::<EXTENDED_HEX, PADDED, UPPERCASE>(raw, clean_enc);

            for pos in 0..enc_len {
                for byte in 0_u8..=u8::MAX {
                    let mutated = &mut mutated_buf[..enc_len];
                    mutated.copy_from_slice(clean_enc);
                    mutated[pos] = byte;

                    let out_cap = max_decoded_len(enc_len);
                    let actual = decode_init::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(
                        mutated,
                        &mut dec_buf[..out_cap],
                    );
                    let expected = decode_scalar::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(
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
                    let ip_expected =
                        decode_in_place_scalar::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(sip);
                    let ip_actual =
                        decode_in_place::<EXTENDED_HEX, PADDED, UPPERCASE, STRICT_CASE>(mutated);
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
        // Covers 1-quantum (8B), NEON/SSSE3 16B, 24B, AVX2 32B, 40B, 64B, and tail boundaries.
        let raw_lens = [1, 5, 10, 11, 20, 24, 40];
        verify_all_bytes_at_offsets::<false, true, true, true>(&raw_lens);
        verify_all_bytes_at_offsets::<false, false, false, true>(&raw_lens);
        verify_all_bytes_at_offsets::<true, true, true, false>(&raw_lens);
        verify_all_bytes_at_offsets::<true, false, false, false>(&raw_lens);

        let strict_lens = [10, 20, 40];
        verify_all_bytes_at_offsets::<false, true, false, true>(&strict_lens);
        verify_all_bytes_at_offsets::<true, true, true, true>(&strict_lens);
        verify_all_bytes_at_offsets::<true, false, false, true>(&strict_lens);
    }

    #[test]
    fn interior_padding_non_canonical_and_strict_case_errors() {
        let mut out = [0_u8; 64];

        // Interior padding inside SIMD chunks for padded & unpadded
        for pos in [0_usize, 15, 16, 31, 32, 46] {
            let mut buf = [b'A'; 48];
            buf[pos] = b'=';
            assert_eq!(
                decode_init::<false, true, true, true>(&buf, &mut out),
                Err(DecodeError::InvalidPadding { index: pos })
            );
            assert_eq!(
                decode_in_place::<false, true, true, true>(&mut buf),
                Err(DecodeError::InvalidPadding { index: pos })
            );
        }
        for pos in [0_usize, 15, 16, 31, 32, 47] {
            let mut buf = [b'a'; 48];
            buf[pos] = b'=';
            assert_eq!(
                decode_init::<false, false, false, true>(&buf, &mut out),
                Err(DecodeError::InvalidPadding { index: pos })
            );
            assert_eq!(
                decode_in_place::<false, false, false, true>(&mut buf),
                Err(DecodeError::InvalidPadding { index: pos })
            );
        }

        // Non-canonical trailing bits after multiple SIMD chunks (32B complete + tail)
        // 2-char tail: low 2 bits of byte[1] must be 0 ('A'=0, 'B'=1)
        let mut tail2 = *b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB======";
        assert_eq!(
            decode_init::<false, true, true, true>(&tail2, &mut out),
            Err(DecodeError::NonCanonical { index: 33 })
        );
        assert_eq!(
            decode_init::<false, false, true, true>(&tail2[..34], &mut out),
            Err(DecodeError::NonCanonical { index: 33 })
        );
        assert_eq!(
            decode_in_place::<false, true, true, true>(&mut tail2),
            Err(DecodeError::NonCanonical { index: 33 })
        );

        // 4-char tail: low 4 bits of byte[3] must be 0
        let mut tail4 = *b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB====";
        assert_eq!(
            decode_init::<false, true, true, true>(&tail4, &mut out),
            Err(DecodeError::NonCanonical { index: 35 })
        );
        assert_eq!(
            decode_in_place::<false, true, true, true>(&mut tail4),
            Err(DecodeError::NonCanonical { index: 35 })
        );

        // 5-char tail: low 1 bit of byte[4] must be 0
        let mut tail5 = *b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB===";
        assert_eq!(
            decode_init::<false, true, true, true>(&tail5, &mut out),
            Err(DecodeError::NonCanonical { index: 36 })
        );
        assert_eq!(
            decode_in_place::<false, true, true, true>(&mut tail5),
            Err(DecodeError::NonCanonical { index: 36 })
        );

        // 7-char tail: low 3 bits of byte[6] must be 0
        let mut tail7 = *b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB=";
        assert_eq!(
            decode_init::<false, true, true, true>(&tail7, &mut out),
            Err(DecodeError::NonCanonical { index: 38 })
        );
        assert_eq!(
            decode_in_place::<false, true, true, true>(&mut tail7),
            Err(DecodeError::NonCanonical { index: 38 })
        );

        // Strict case rejects opposite case inside SIMD chunks for Standard & ExtendedHex
        let mut upper_with_lower = [b'B'; 40];
        upper_with_lower[23] = b'b';
        assert_eq!(
            decode_init::<false, true, true, true>(&upper_with_lower, &mut out),
            Err(DecodeError::InvalidByte {
                index: 23,
                byte: b'b'
            })
        );
        assert_eq!(
            decode_init::<false, true, true, false>(&upper_with_lower, &mut out),
            Ok(25)
        );

        let mut lower_hex_with_upper = [b'v'; 40];
        lower_hex_with_upper[17] = b'V';
        assert_eq!(
            decode_init::<true, true, false, true>(&lower_hex_with_upper, &mut out),
            Err(DecodeError::InvalidByte {
                index: 17,
                byte: b'V'
            })
        );
        assert_eq!(
            decode_init::<true, true, false, false>(&lower_hex_with_upper, &mut out),
            Ok(25)
        );
    }

    #[test]
    fn bounded_wide_store_respects_exact_output_and_chunk_errors() {
        fn check_bounds<const PADDED: bool>() {
            let mut raw_buf = [0_u8; 80];
            let mut enc_buf = [0_u8; 136];
            let mut sentinel = [0_u8; 128];

            for raw_len in 0..=64 {
                let raw = &mut raw_buf[..raw_len];
                fill_pattern(raw, raw_len + 71);
                let enc_len = encoded_len::<PADDED>(raw_len).expect("within bounds");
                let enc = &mut enc_buf[..enc_len];
                encode_init::<false, PADDED, true>(raw, enc);

                let plan = decode_plan::<PADDED>(enc).unwrap();
                assert_eq!(plan.required, raw_len);

                sentinel.fill(0xa5);
                let written = decode_with_plan::<false, true, false>(
                    enc,
                    plan,
                    codec::as_uninit_slice(&mut sentinel[..raw_len]),
                )
                .unwrap();
                assert_eq!(written, raw_len);
                assert_eq!(&sentinel[..raw_len], raw);
                assert!(sentinel[raw_len..].iter().all(|&b| b == 0xa5));

                sentinel.fill(0x5a);
                let written_oversized = decode_with_plan::<false, true, false>(
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

        // Cover single/paired 16-char SIMD bounds (10B, 16B, 20B, 25B, 26B, 30B, 40B, 64B).
        check_bounds::<true>();
        check_bounds::<false>();

        // Reject invalid bytes across paired 32-char and trailing 16-char chunks without mutating
        // unvalidated output offsets.
        let mut sentinel = [0_u8; 40];
        let mut bad_buf = [b'A'; 64];
        for pos in [0_usize, 15, 16, 31, 32, 47, 48, 63] {
            let mut input = bad_buf;
            input[pos] = b'!';
            assert_eq!(
                decode_init::<false, true, true, false>(&input, &mut sentinel),
                Err(DecodeError::InvalidByte {
                    index: pos,
                    byte: b'!'
                })
            );
            assert_eq!(
                decode_in_place::<false, true, true, false>(&mut input),
                Err(DecodeError::InvalidByte {
                    index: pos,
                    byte: b'!'
                })
            );
        }
        bad_buf[63] = b'8';
        assert_eq!(
            decode_in_place::<false, true, true, false>(&mut bad_buf),
            Err(DecodeError::InvalidByte {
                index: 63,
                byte: b'8'
            })
        );
    }
}
