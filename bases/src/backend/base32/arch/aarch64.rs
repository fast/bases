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
use core::arch::aarch64::vdupq_n_u64;
use core::arch::aarch64::vget_low_u8;
use core::arch::aarch64::vgetq_lane_u16;
use core::arch::aarch64::vld1q_s16;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vminq_u8;
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
use core::arch::aarch64::vsetq_lane_u64;
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
use core::mem::MaybeUninit;
use core::ptr;

use crate::backend::base32::LO_MASK_HEX;
use crate::backend::base32::LO_MASK_STD;
use crate::backend::base32::select_alphabet;
use crate::backend::base32::select_hi_mask;

const W_SHUF: [u8; 16] = [1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 9, 8];
const EVEN_SHIFTS: [i16; 8] = [-11, -9, -7, -5, -11, -9, -7, -5];
const ODD_SHIFTS: [i16; 8] = [2, 4, 6, 8, 2, 4, 6, 8];

const ROLL_STD: [u8; 16] = [
    0, 0, 0, 0xe8, 0xbf, 0xbf, 0x9f, 0x9f, 0, 0, 0, 0, 0, 0, 0, 0,
];
const ROLL_HEX: [u8; 16] = [
    0, 0, 0, 0xd0, 0xc9, 0xc9, 0xa9, 0xa9, 0, 0, 0, 0, 0, 0, 0, 0,
];

const PACK_SHUF: [u8; 16] = [4, 3, 2, 1, 0, 12, 11, 10, 9, 8, 0, 0, 0, 0, 0, 0];

#[inline]
pub fn encode<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    let chunks = (input.len() / 10).min(output.len() / 16);
    if chunks == 0 {
        return (0, 0);
    }

    // SAFETY: `input.len() >= chunks * 10`, `output.len() >= chunks * 16`,
    // `MaybeUninit<u8>` has the same layout as `u8`, and NEON is baseline on aarch64.
    unsafe {
        encode_neon::<EXTENDED_HEX, UPPERCASE>(
            input.as_ptr(),
            output.as_mut_ptr().cast::<u8>(),
            chunks,
            input.len(),
        );
    }
    (chunks * 10, chunks * 16)
}

#[inline]
pub fn decode<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    let chunks = (input.len() / 16).min(output.len() / 10);
    if chunks == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks * 16 <= input.len()`, `chunks * 10 <= output.len()`, non-final wide stores
    // stay within `output.len()`, `MaybeUninit<u8>` has the same layout as `u8`, and NEON is
    // baseline.
    let done = unsafe {
        decode_neon::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
            input.as_ptr(),
            output.as_mut_ptr().cast::<u8>(),
            chunks,
            output.len(),
        )
    };
    (done * 16, done * 10)
}

#[inline]
pub fn decode_in_place<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
    complete_len: usize,
    required: usize,
) -> (usize, usize) {
    let chunks = complete_len.min(buf.len()) / 16;
    if chunks == 0 {
        return (0, 0);
    }

    let out_len = required.min(buf.len());
    let ptr = buf.as_mut_ptr();
    // SAFETY: `chunks * 16 <= buf.len()`. Each iteration or pair loads all 16 or 32 input bytes
    // before storing, bounded wide stores stay within `out_len`, and `out_ptr` advances strictly
    // slower than `in_ptr` (`10k + 16 <= 16(k + 1)` and `10k + 26 <= 16(k + 2)`).
    let done = unsafe {
        decode_neon::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(ptr.cast_const(), ptr, chunks, out_len)
    };
    (done * 16, done * 10)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `input_len >= chunks * 10` reads (`chunks > 0`) and
/// `out_ptr` is valid for `chunks * 16` non-overlapping writes, with NEON enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn encode_neon<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
    input_len: usize,
) {
    let alpha_ptr = select_alphabet::<EXTENDED_HEX, UPPERCASE>().as_ptr();
    // SAFETY: Caller guarantees sufficient readable/writable bytes and `alpha_ptr` points to 32
    // bytes.
    unsafe {
        let alphabet_tbl = uint8x16x2_t(vld1q_u8(alpha_ptr), vld1q_u8(alpha_ptr.add(16)));
        let w_shuf = vld1q_u8(W_SHUF.as_ptr());
        let even_shifts: int16x8_t = vld1q_s16(EVEN_SHIFTS.as_ptr());
        let odd_shifts: int16x8_t = vld1q_s16(ODD_SHIFTS.as_ptr());
        let wide_chunks = ((input_len - 6) / 10).min(chunks);
        for _ in 0..wide_chunks {
            let raw = vld1q_u8(in_ptr);
            let ascii = encode_chunk_neon(raw, alphabet_tbl, w_shuf, even_shifts, odd_shifts);
            vst1q_u8(out_ptr, ascii);
            in_ptr = in_ptr.add(10);
            out_ptr = out_ptr.add(16);
        }
        if wide_chunks < chunks {
            // The last two quanta need only ten readable bytes, even at an allocation boundary.
            let low = ptr::read_unaligned(in_ptr.cast::<u64>());
            let high = u64::from(ptr::read_unaligned(in_ptr.add(8).cast::<u16>()));
            let raw = vreinterpretq_u8_u64(vsetq_lane_u64::<1>(high, vdupq_n_u64(low)));
            let ascii = encode_chunk_neon(raw, alphabet_tbl, w_shuf, even_shifts, odd_shifts);
            vst1q_u8(out_ptr, ascii);
        }
    }
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn encode_chunk_neon(
    raw: uint8x16_t,
    alphabet: uint8x16x2_t,
    shuffle: uint8x16_t,
    even_shifts: int16x8_t,
    odd_shifts: int16x8_t,
) -> uint8x16_t {
    let words = vreinterpretq_u16_u8(vqtbl1q_u8(raw, shuffle));
    let even = vandq_u16(vshlq_u16(words, even_shifts), vdupq_n_u16(0x001f));
    let odd = vandq_u16(vshlq_u16(words, odd_shifts), vdupq_n_u16(0x1f00));
    let indices = vorrq_u8(vreinterpretq_u8_u16(even), vreinterpretq_u8_u16(odd));
    vqtbl2q_u8(alphabet, indices)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads, `out_ptr` is valid for
/// `out_len >= chunks * 10` writes, and NEON is enabled. `out_ptr` must either not overlap
/// `in_ptr` or equal `in_ptr` (every iteration or pair loads all its input before storing,
/// wide 16-byte stores require `10 * k + 16 <= out_len`, and `out_ptr` never overtakes unread
/// input).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_neon<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
    out_len: usize,
) -> usize {
    let hi_mask_ptr = select_hi_mask::<UPPERCASE, STRICT_CASE>().as_ptr();
    let lo_mask_ptr = if EXTENDED_HEX {
        &LO_MASK_HEX
    } else {
        &LO_MASK_STD
    }
    .as_ptr();
    let roll_ptr = if EXTENDED_HEX { &ROLL_HEX } else { &ROLL_STD }.as_ptr();
    let wide_chunks = (out_len.saturating_sub(6) / 10).min(chunks);
    let mut done = 0;

    // SAFETY: Caller guarantees `chunks * 16` reads at `in_ptr`, `out_len >= chunks * 10` writes at
    // `out_ptr`, non-overtaking pointer advancement for in-place decoding, and 16-byte static
    // lookup tables.
    unsafe {
        let hi_mask_tbl = vld1q_u8(hi_mask_ptr);
        let lo_mask_tbl = vld1q_u8(lo_mask_ptr);
        let roll_tbl = vld1q_u8(roll_ptr);
        let pack_shuf = vld1q_u8(PACK_SHUF.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        let mask_001f = vdupq_n_u16(0x001f);
        let mask_03ff = vdupq_n_u32(0x03ff);

        while done + 2 <= wide_chunks {
            let v0 = vld1q_u8(in_ptr);
            let v1 = vld1q_u8(in_ptr.add(16));
            let (hi0, chk0) = classify_chunk_16_neon(v0, hi_mask_tbl, lo_mask_tbl, mask_0f);
            let (hi1, chk1) = classify_chunk_16_neon(v1, hi_mask_tbl, lo_mask_tbl, mask_0f);
            if vminvq_u8(vminq_u8(chk0, chk1)) == 0 {
                return done;
            }
            let res0 = pack_chunk_16_neon(v0, hi0, roll_tbl, pack_shuf, mask_001f, mask_03ff);
            let res1 = pack_chunk_16_neon(v1, hi1, roll_tbl, pack_shuf, mask_001f, mask_03ff);
            vst1q_u8(out_ptr, res0);
            vst1q_u8(out_ptr.add(10), res1);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(20);
            done += 2;
        }

        while done < chunks {
            let v = vld1q_u8(in_ptr);
            let (hi, chk) = classify_chunk_16_neon(v, hi_mask_tbl, lo_mask_tbl, mask_0f);
            if vminvq_u8(chk) == 0 {
                return done;
            }
            let res = pack_chunk_16_neon(v, hi, roll_tbl, pack_shuf, mask_001f, mask_03ff);
            if done < wide_chunks {
                vst1q_u8(out_ptr, res);
            } else {
                vst1_u8(out_ptr, vget_low_u8(res));
                ptr::write_unaligned(
                    out_ptr.add(8).cast::<u16>(),
                    vgetq_lane_u16::<4>(vreinterpretq_u16_u8(res)),
                );
            }
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
unsafe fn classify_chunk_16_neon(
    v: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let hi = vshrq_n_u8::<4>(v);
    let lo = vandq_u8(v, mask_0f);
    let chk = vandq_u8(vqtbl1q_u8(hi_mask_tbl, hi), vqtbl1q_u8(lo_mask_tbl, lo));
    (hi, chk)
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pack_chunk_16_neon(
    v: uint8x16_t,
    hi: uint8x16_t,
    roll_tbl: uint8x16_t,
    pack_shuf: uint8x16_t,
    mask_001f: uint16x8_t,
    mask_03ff: uint32x4_t,
) -> uint8x16_t {
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
    vqtbl1q_u8(vreinterpretq_u8_u64(q), pack_shuf)
}
