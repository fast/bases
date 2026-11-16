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

//! AArch64 NEON Base32 kernels.

use core::arch::aarch64::int16x8_t;
use core::arch::aarch64::uint8x16_t;
use core::arch::aarch64::uint8x16x2_t;
use core::arch::aarch64::uint16x8_t;
use core::arch::aarch64::uint32x4_t;
use core::arch::aarch64::vaddq_u8;
use core::arch::aarch64::vandq_u8;
use core::arch::aarch64::vandq_u16;
use core::arch::aarch64::vandq_u32;
use core::arch::aarch64::vdupq_n_u8;
use core::arch::aarch64::vdupq_n_u16;
use core::arch::aarch64::vdupq_n_u32;
use core::arch::aarch64::vget_low_u8;
use core::arch::aarch64::vgetq_lane_u16;
use core::arch::aarch64::vld1q_s16;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vminvq_u8;
use core::arch::aarch64::vorrq_u8;
use core::arch::aarch64::vorrq_u16;
use core::arch::aarch64::vorrq_u32;
use core::arch::aarch64::vorrq_u64;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vqtbl2q_u8;
use core::arch::aarch64::vreinterpretq_u8_u16;
use core::arch::aarch64::vreinterpretq_u8_u64;
use core::arch::aarch64::vreinterpretq_u16_u8;
use core::arch::aarch64::vreinterpretq_u32_u16;
use core::arch::aarch64::vreinterpretq_u64_u32;
use core::arch::aarch64::vshlq_n_u16;
use core::arch::aarch64::vshlq_n_u32;
use core::arch::aarch64::vshlq_n_u64;
use core::arch::aarch64::vshlq_u16;
use core::arch::aarch64::vshrq_n_u8;
use core::arch::aarch64::vshrq_n_u16;
use core::arch::aarch64::vshrq_n_u32;
use core::arch::aarch64::vshrq_n_u64;
use core::arch::aarch64::vst1_u8;
use core::arch::aarch64::vst1q_u8;
use core::ptr;

const STANDARD_UPPER: [u8; 32] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const STANDARD_LOWER: [u8; 32] = *b"abcdefghijklmnopqrstuvwxyz234567";
const HEX_UPPER: [u8; 32] = *b"0123456789ABCDEFGHIJKLMNOPQRSTUV";
const HEX_LOWER: [u8; 32] = *b"0123456789abcdefghijklmnopqrstuv";

const W_SHUF: [u8; 16] = [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 9, 8];
const EVEN_SHIFTS: [i16; 8] = [-11, -9, -7, -5, -11, -9, -7, -5];
const ODD_SHIFTS: [i16; 8] = [2, 4, 6, 8, 2, 4, 6, 8];

const HI_MASK_MIXED: [u8; 16] = [0, 0, 0, 1, 2, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_UPPER: [u8; 16] = [0, 0, 0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_LOWER: [u8; 16] = [0, 0, 0, 1, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0];

const LO_MASK_STD: [u8; 16] = [4, 6, 7, 7, 7, 7, 7, 7, 6, 6, 6, 2, 2, 2, 2, 2];
const LO_MASK_HEX: [u8; 16] = [5, 7, 7, 7, 7, 7, 7, 3, 3, 3, 2, 2, 2, 2, 2, 2];

const ROLL_STD: [u8; 16] = [
    0, 0, 0, 0xe8, 0xbf, 0xbf, 0x9f, 0x9f, 0, 0, 0, 0, 0, 0, 0, 0,
];
const ROLL_HEX: [u8; 16] = [
    0, 0, 0, 0xd0, 0xc9, 0xc9, 0xa9, 0xa9, 0, 0, 0, 0, 0, 0, 0, 0,
];

const PACK_SHUF: [u8; 16] = [4, 3, 2, 1, 0, 12, 11, 10, 9, 8, 0, 0, 0, 0, 0, 0];

const fn select_alphabet<const EXTENDED_HEX: bool, const UPPERCASE: bool>() -> &'static [u8; 32] {
    match (EXTENDED_HEX, UPPERCASE) {
        (false, true) => &STANDARD_UPPER,
        (false, false) => &STANDARD_LOWER,
        (true, true) => &HEX_UPPER,
        (true, false) => &HEX_LOWER,
    }
}

const fn select_hi_mask<const UPPERCASE: bool, const STRICT_CASE: bool>() -> &'static [u8; 16] {
    if !STRICT_CASE {
        &HI_MASK_MIXED
    } else if UPPERCASE {
        &HI_MASK_UPPER
    } else {
        &HI_MASK_LOWER
    }
}

#[inline]
pub fn encode<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    if input.len() < 16 {
        return (0, 0);
    }
    let chunks = ((input.len() - 6) / 10).min(output.len() / 16);
    if chunks == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks > 0` implies `input.len() >= (chunks - 1) * 10 + 16` and
    // `output.len() >= chunks * 16`. NEON is baseline on aarch64.
    unsafe {
        encode_neon::<EXTENDED_HEX, UPPERCASE>(input.as_ptr(), output.as_mut_ptr(), chunks);
    }
    (chunks * 10, chunks * 16)
}

#[inline]
pub fn decode<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks = (input.len() / 16).min(output.len() / 10);
    if chunks == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks * 16 <= input.len()` and `chunks * 10 <= output.len()`, and NEON is baseline.
    let done = unsafe {
        decode_neon::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            input.as_ptr(),
            output.as_mut_ptr(),
            chunks,
        )
    };
    (done * 16, done * 10)
}

#[inline]
pub fn decode_in_place<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
    complete_len: usize,
) -> (usize, usize) {
    let chunks = complete_len.min(buf.len()) / 16;
    if chunks == 0 {
        return (0, 0);
    }

    let ptr = buf.as_mut_ptr();
    // SAFETY: `chunks * 16 <= buf.len()`. Every iteration loads 16 bytes before storing 10 bytes,
    // and `out_ptr` advances by 10 while `in_ptr` advances by 16.
    let done = unsafe {
        decode_neon::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(ptr.cast_const(), ptr, chunks)
    };
    (done * 16, done * 10)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `(chunks - 1) * 10 + 16` reads (`chunks > 0`) and
/// `out_ptr` is valid for `chunks * 16` non-overlapping writes, with NEON enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn encode_neon<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
) {
    let alpha_ptr = select_alphabet::<EXTENDED_HEX, UPPERCASE>().as_ptr();
    // SAFETY: Caller guarantees sufficient readable/writable bytes and `alpha_ptr` points to 32
    // bytes.
    unsafe {
        let alphabet_tbl = uint8x16x2_t(vld1q_u8(alpha_ptr), vld1q_u8(alpha_ptr.add(16)));
        let w_shuf = vld1q_u8(W_SHUF.as_ptr());
        let even_shifts: int16x8_t = vld1q_s16(EVEN_SHIFTS.as_ptr());
        let odd_shifts: int16x8_t = vld1q_s16(ODD_SHIFTS.as_ptr());
        let even_mask = vdupq_n_u16(0x001f);
        let odd_mask = vdupq_n_u16(0x1f00);

        for _ in 0..chunks {
            let raw = vld1q_u8(in_ptr);
            let w = vreinterpretq_u16_u8(vqtbl1q_u8(raw, w_shuf));
            let even = vandq_u16(vshlq_u16(w, even_shifts), even_mask);
            let odd = vandq_u16(vshlq_u16(w, odd_shifts), odd_mask);
            let indices = vorrq_u8(vreinterpretq_u8_u16(even), vreinterpretq_u8_u16(odd));
            let ascii = vqtbl2q_u8(alphabet_tbl, indices);
            vst1q_u8(out_ptr, ascii);
            in_ptr = in_ptr.add(10);
            out_ptr = out_ptr.add(16);
        }
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads, `out_ptr` is valid for
/// `chunks * 10` writes, and NEON is enabled. `out_ptr` must either not overlap `in_ptr` or equal
/// `in_ptr` (each iteration loads 16 input bytes before writing 10 output bytes, so `out_ptr` never
/// overtakes unread input).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_neon<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
) -> usize {
    let hi_mask_ptr = select_hi_mask::<UPPERCASE, STRICT_CASE>().as_ptr();
    let lo_mask_ptr = if EXTENDED_HEX {
        &LO_MASK_HEX
    } else {
        &LO_MASK_STD
    }
    .as_ptr();
    let roll_ptr = if EXTENDED_HEX { &ROLL_HEX } else { &ROLL_STD }.as_ptr();
    let mut done = 0;

    // SAFETY: Caller guarantees `chunks * 16` reads at `in_ptr`, `chunks * 10` writes at `out_ptr`,
    // non-overtaking pointer advancement for in-place decoding, and 16-byte static lookup tables.
    unsafe {
        let hi_mask_tbl = vld1q_u8(hi_mask_ptr);
        let lo_mask_tbl = vld1q_u8(lo_mask_ptr);
        let roll_tbl = vld1q_u8(roll_ptr);
        let pack_shuf = vld1q_u8(PACK_SHUF.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        let mask_001f = vdupq_n_u16(0x001f);
        let mask_03ff = vdupq_n_u32(0x03ff);

        while done < chunks {
            let v = vld1q_u8(in_ptr);
            let Some(res) = decode_chunk_16_neon(
                v,
                hi_mask_tbl,
                lo_mask_tbl,
                roll_tbl,
                pack_shuf,
                mask_0f,
                mask_001f,
                mask_03ff,
            ) else {
                break;
            };
            vst1_u8(out_ptr, vget_low_u8(res));
            ptr::write_unaligned(
                out_ptr.add(8).cast::<u16>(),
                vgetq_lane_u16::<4>(vreinterpretq_u16_u8(res)),
            );
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(10);
            done += 1;
        }
    }
    done
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_16_neon(
    v: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    roll_tbl: uint8x16_t,
    pack_shuf: uint8x16_t,
    mask_0f: uint8x16_t,
    mask_001f: uint16x8_t,
    mask_03ff: uint32x4_t,
) -> Option<uint8x16_t> {
    let hi = vshrq_n_u8::<4>(v);
    let lo = vandq_u8(v, mask_0f);
    let chk = vandq_u8(vqtbl1q_u8(hi_mask_tbl, hi), vqtbl1q_u8(lo_mask_tbl, lo));
    if vminvq_u8(chk) == 0 {
        return None;
    }

    let vals = vaddq_u8(v, vqtbl1q_u8(roll_tbl, hi));
    let vals_u16 = vreinterpretq_u16_u8(vals);
    let lo_byte = vandq_u16(vals_u16, mask_001f);
    let hi_byte = vshrq_n_u16::<8>(vals_u16);
    let p10 = vorrq_u16(vshlq_n_u16::<5>(lo_byte), hi_byte);

    let p10_u32 = vreinterpretq_u32_u16(p10);
    let p20 = vorrq_u32(
        vshlq_n_u32::<10>(vandq_u32(p10_u32, mask_03ff)),
        vshrq_n_u32::<16>(p10_u32),
    );

    let p20_u64 = vreinterpretq_u64_u32(p20);
    let q = vorrq_u64(vshlq_n_u64::<20>(p20_u64), vshrq_n_u64::<32>(p20_u64));
    Some(vqtbl1q_u8(vreinterpretq_u8_u64(q), pack_shuf))
}
