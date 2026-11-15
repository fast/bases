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

//! AArch64 NEON Base16 kernels.

use core::arch::aarch64::uint8x8_t;
use core::arch::aarch64::uint8x16_t;
use core::arch::aarch64::uint8x16x2_t;
use core::arch::aarch64::vaddq_u8;
use core::arch::aarch64::vandq_u8;
use core::arch::aarch64::vcombine_u8;
use core::arch::aarch64::vdupq_n_u8;
use core::arch::aarch64::vget_high_u8;
use core::arch::aarch64::vget_low_u8;
use core::arch::aarch64::vld1_u8;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vld2q_u8;
use core::arch::aarch64::vminq_u8;
use core::arch::aarch64::vminvq_u8;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vshrq_n_u8;
use core::arch::aarch64::vsli_n_u8;
use core::arch::aarch64::vsliq_n_u8;
use core::arch::aarch64::vst1_u8;
use core::arch::aarch64::vst1q_u8;
use core::arch::aarch64::vuzp1_u8;
use core::arch::aarch64::vuzp2_u8;
use core::arch::aarch64::vzip1q_u8;
use core::arch::aarch64::vzip2q_u8;

const UPPER_HEX: [u8; 16] = *b"0123456789ABCDEF";
const LOWER_HEX: [u8; 16] = *b"0123456789abcdef";

const HI_MASK_MIXED: [u8; 16] = [0, 0, 0, 1, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_UPPER: [u8; 16] = [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_LOWER: [u8; 16] = [0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const LO_MASK: [u8; 16] = [1, 7, 7, 7, 7, 7, 7, 1, 1, 1, 0, 0, 0, 0, 0, 0];
const HI_OFFSET: [u8; 16] = [0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0];

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
pub fn encode<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    let consumed = input.len().min(output.len() / 2) & !7;
    if consumed == 0 {
        return (0, 0);
    }

    // SAFETY: `consumed <= input.len()`, `consumed * 2 <= output.len()`, and NEON is baseline on
    // aarch64.
    unsafe { encode_neon::<UPPERCASE>(input.as_ptr(), output.as_mut_ptr(), consumed) };
    (consumed, consumed * 2)
}

#[inline]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks_16 = (input.len() / 16).min(output.len() / 8);
    if chunks_16 == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks_16 * 16 <= input.len()` and `chunks_16 * 8 <= output.len()`, and NEON is
    // baseline on aarch64.
    let done_16 = unsafe {
        decode_neon::<UPPERCASE, STRICT_CASE>(input.as_ptr(), output.as_mut_ptr(), chunks_16)
    };
    (done_16 * 16, done_16 * 8)
}

#[inline]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    let chunks_16 = buf.len() / 16;
    if chunks_16 == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks_16 * 16 <= buf.len()`, each iteration loads 32 or 16 bytes before storing
    // 16 or 8 bytes, and NEON is baseline on aarch64.
    let done_16 =
        unsafe { decode_in_place_neon::<UPPERCASE, STRICT_CASE>(buf.as_mut_ptr(), chunks_16) };
    (done_16 * 16, done_16 * 8)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads and `out_ptr` is valid for
/// `chunks * 32` non-overlapping writes, with NEON enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn encode_neon<const UPPERCASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    len: usize,
) {
    let table = if UPPERCASE { &UPPER_HEX } else { &LOWER_HEX };
    // SAFETY: Caller guarantees `len` readable bytes at `in_ptr`, `len * 2` writable bytes at
    // `out_ptr`, `len` is a multiple of 8, and `table` is a 16-byte static array.
    unsafe {
        let hex_tbl = vld1q_u8(table.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        let mut rem = len;

        while rem >= 32 {
            let v0 = vld1q_u8(in_ptr);
            let v1 = vld1q_u8(in_ptr.add(16));

            let hi0 = vqtbl1q_u8(hex_tbl, vshrq_n_u8::<4>(v0));
            let lo0 = vqtbl1q_u8(hex_tbl, vandq_u8(v0, mask_0f));
            let out0 = vzip1q_u8(hi0, lo0);
            let out1 = vzip2q_u8(hi0, lo0);

            let hi1 = vqtbl1q_u8(hex_tbl, vshrq_n_u8::<4>(v1));
            let lo1 = vqtbl1q_u8(hex_tbl, vandq_u8(v1, mask_0f));
            let out2 = vzip1q_u8(hi1, lo1);
            let out3 = vzip2q_u8(hi1, lo1);

            vst1q_u8(out_ptr, out0);
            vst1q_u8(out_ptr.add(16), out1);
            vst1q_u8(out_ptr.add(32), out2);
            vst1q_u8(out_ptr.add(48), out3);

            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(64);
            rem -= 32;
        }

        if rem >= 16 {
            let v = vld1q_u8(in_ptr);
            let hi = vqtbl1q_u8(hex_tbl, vshrq_n_u8::<4>(v));
            let lo = vqtbl1q_u8(hex_tbl, vandq_u8(v, mask_0f));
            vst1q_u8(out_ptr, vzip1q_u8(hi, lo));
            vst1q_u8(out_ptr.add(16), vzip2q_u8(hi, lo));

            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(32);
            rem -= 16;
        }

        if rem >= 8 {
            let v8 = vld1_u8(in_ptr);
            let v = vcombine_u8(v8, v8);
            let hi = vqtbl1q_u8(hex_tbl, vshrq_n_u8::<4>(v));
            let lo = vqtbl1q_u8(hex_tbl, vandq_u8(v, mask_0f));
            vst1q_u8(out_ptr, vzip1q_u8(hi, lo));
        }
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_16 * 16` reads and `out_ptr` is valid for
/// `chunks_16 * 8` non-overlapping writes, with NEON enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_neon<const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_16: usize,
) -> usize {
    let hi_mask_ptr = select_hi_mask::<UPPERCASE, STRICT_CASE>().as_ptr();
    let mut done_16 = 0;
    // SAFETY: Caller guarantees `chunks_16 * 16` readable bytes at `in_ptr`, `chunks_16 * 8`
    // writable bytes at `out_ptr`, and table pointers point to 16-byte static arrays.
    unsafe {
        let hi_mask_tbl = vld1q_u8(hi_mask_ptr);
        let lo_mask_tbl = vld1q_u8(LO_MASK.as_ptr());
        let hi_off_tbl = vld1q_u8(HI_OFFSET.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        let mut saw_invalid = false;

        while done_16 + 2 <= chunks_16 {
            let uint8x16x2_t(even, odd) = vld2q_u8(in_ptr);
            let Some(packed) =
                decode_pair_neon(even, odd, hi_mask_tbl, lo_mask_tbl, hi_off_tbl, mask_0f)
            else {
                saw_invalid = true;
                break;
            };
            vst1q_u8(out_ptr, packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
            done_16 += 2;
        }

        if !saw_invalid
            && done_16 < chunks_16
            && let Some(packed) = decode_16_neon(
                vld1q_u8(in_ptr),
                hi_mask_tbl,
                lo_mask_tbl,
                hi_off_tbl,
                mask_0f,
            )
        {
            vst1_u8(out_ptr, packed);
            done_16 += 1;
        }
    }
    done_16
}

/// # Safety
///
/// Caller must ensure `ptr` is valid for `chunks_16 * 16` reads and writes, with NEON enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_in_place_neon<const UPPERCASE: bool, const STRICT_CASE: bool>(
    ptr: *mut u8,
    chunks_16: usize,
) -> usize {
    let hi_mask_ptr = select_hi_mask::<UPPERCASE, STRICT_CASE>().as_ptr();
    let mut in_ptr: *const u8 = ptr;
    let mut out_ptr: *mut u8 = ptr;
    let mut done_16 = 0;
    // SAFETY: Caller guarantees `chunks_16 * 16` readable/writable bytes from `ptr`. Each iteration
    // loads `in_ptr[0..32]` or `in_ptr[0..16]` completely before writing `out_ptr[0..16]` or
    // `out_ptr[0..8]`, and `out_ptr` lags behind `in_ptr` by at least `8 * done_16` bytes so future
    // reads are never overwritten.
    unsafe {
        let hi_mask_tbl = vld1q_u8(hi_mask_ptr);
        let lo_mask_tbl = vld1q_u8(LO_MASK.as_ptr());
        let hi_off_tbl = vld1q_u8(HI_OFFSET.as_ptr());
        let mask_0f = vdupq_n_u8(0x0f);
        let mut saw_invalid = false;

        while done_16 + 2 <= chunks_16 {
            let uint8x16x2_t(even, odd) = vld2q_u8(in_ptr);
            let Some(packed) =
                decode_pair_neon(even, odd, hi_mask_tbl, lo_mask_tbl, hi_off_tbl, mask_0f)
            else {
                saw_invalid = true;
                break;
            };
            vst1q_u8(out_ptr, packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
            done_16 += 2;
        }

        if !saw_invalid
            && done_16 < chunks_16
            && let Some(packed) = decode_16_neon(
                vld1q_u8(in_ptr),
                hi_mask_tbl,
                lo_mask_tbl,
                hi_off_tbl,
                mask_0f,
            )
        {
            vst1_u8(out_ptr, packed);
            done_16 += 1;
        }
    }
    done_16
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_16_neon(
    chunk: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    hi_off_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> Option<uint8x8_t> {
    let hi = vshrq_n_u8::<4>(chunk);
    let lo = vandq_u8(chunk, mask_0f);
    let chk = vandq_u8(vqtbl1q_u8(hi_mask_tbl, hi), vqtbl1q_u8(lo_mask_tbl, lo));
    if vminvq_u8(chk) == 0 {
        return None;
    }

    let nib = vaddq_u8(lo, vqtbl1q_u8(hi_off_tbl, hi));
    let lo_half = vget_low_u8(nib);
    let hi_half = vget_high_u8(nib);
    let even = vuzp1_u8(lo_half, hi_half);
    let odd = vuzp2_u8(lo_half, hi_half);
    Some(vsli_n_u8::<4>(odd, even))
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_pair_neon(
    even: uint8x16_t,
    odd: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    hi_off_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> Option<uint8x16_t> {
    let hi_even = vshrq_n_u8::<4>(even);
    let lo_even = vandq_u8(even, mask_0f);
    let hi_odd = vshrq_n_u8::<4>(odd);
    let lo_odd = vandq_u8(odd, mask_0f);

    let chk_even = vandq_u8(
        vqtbl1q_u8(hi_mask_tbl, hi_even),
        vqtbl1q_u8(lo_mask_tbl, lo_even),
    );
    let chk_odd = vandq_u8(
        vqtbl1q_u8(hi_mask_tbl, hi_odd),
        vqtbl1q_u8(lo_mask_tbl, lo_odd),
    );
    if vminvq_u8(vminq_u8(chk_even, chk_odd)) == 0 {
        return None;
    }

    let nib_even = vaddq_u8(lo_even, vqtbl1q_u8(hi_off_tbl, hi_even));
    let nib_odd = vaddq_u8(lo_odd, vqtbl1q_u8(hi_off_tbl, hi_odd));
    Some(vsliq_n_u8::<4>(nib_odd, nib_even))
}
