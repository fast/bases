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
use core::mem::MaybeUninit;

use crate::backend::base16::LO_MASK;
use crate::backend::base16::select_hi_mask;
use crate::backend::simd::NeonToken;
use crate::backend::simd::simd_fn;

const UPPER_HEX: &[u8; 16] = b"0123456789ABCDEF";
const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
const HI_OFFSET: [u8; 16] = [0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline]
pub fn encode<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    encode_neon::<UPPERCASE>(NeonToken::summon(), input, output)
}

#[inline]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    let chunks_16 = (input.len() / 16).min(output.len() / 8);
    if chunks_16 == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks_16 * 16 <= input.len()`, `chunks_16 * 8 <= output.len()`, `MaybeUninit<u8>`
    // has the same layout as `u8`, and NEON is baseline on aarch64.
    let done_16 = unsafe {
        decode_neon::<UPPERCASE, STRICT_CASE>(
            input.as_ptr(),
            output.as_mut_ptr().cast::<u8>(),
            chunks_16,
        )
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

    let ptr = buf.as_mut_ptr();
    // SAFETY: `chunks_16 * 16 <= buf.len()`, each iteration loads 32 or 16 bytes before storing
    // 16 or 8 bytes, and NEON is baseline on aarch64.
    let done_16 =
        unsafe { decode_neon::<UPPERCASE, STRICT_CASE>(ptr.cast_const(), ptr, chunks_16) };
    (done_16 * 16, done_16 * 8)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_16(bytes: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `bytes` contains 16 initialized readable bytes.
    unsafe { vld1q_u8(bytes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn encode_16_neon(
    v: uint8x16_t,
    hex_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let hi = vqtbl1q_u8(hex_tbl, vshrq_n_u8::<4>(v));
    let lo = vqtbl1q_u8(hex_tbl, vandq_u8(v, mask_0f));
    (vzip1q_u8(hi, lo), vzip2q_u8(hi, lo))
}

simd_fn! {
    fn encode_neon<const UPPERCASE: bool>(
        _token: NeonToken,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> (usize, usize) {
        let consumed = input.len().min(output.len() / 2) & !7;
        if consumed == 0 {
            return (0, 0);
        }

        let hex_tbl = load_16(if UPPERCASE { UPPER_HEX } else { LOWER_HEX });
        let mask_0f = vdupq_n_u8(0x0f);
        let mut in_ptr = input.as_ptr();
        let mut out_ptr = output.as_mut_ptr().cast::<u8>();
        let mut rem = consumed;

        while rem >= 32 {
            // SAFETY: `rem <= consumed <= input.len()` guarantees 32 readable bytes at `in_ptr`.
            let (v0, v1) = unsafe { (vld1q_u8(in_ptr), vld1q_u8(in_ptr.add(16))) };
            let (out0, out1) = encode_16_neon(v0, hex_tbl, mask_0f);
            let (out2, out3) = encode_16_neon(v1, hex_tbl, mask_0f);

            // SAFETY: `rem * 2 <= consumed * 2 <= output.len()` guarantees 64 writable bytes at
            // `out_ptr`, and advancing by 32 and 64 bytes stays within both buffers.
            unsafe {
                vst1q_u8(out_ptr, out0);
                vst1q_u8(out_ptr.add(16), out1);
                vst1q_u8(out_ptr.add(32), out2);
                vst1q_u8(out_ptr.add(48), out3);
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(64);
            }
            rem -= 32;
        }

        if rem >= 16 {
            // SAFETY: `rem >= 16` guarantees 16 readable bytes at `in_ptr`.
            let v = unsafe { vld1q_u8(in_ptr) };
            let (out0, out1) = encode_16_neon(v, hex_tbl, mask_0f);

            // SAFETY: `rem * 2 >= 32` guarantees 32 writable bytes at `out_ptr`, and advancing by 16
            // and 32 bytes stays within both buffers.
            unsafe {
                vst1q_u8(out_ptr, out0);
                vst1q_u8(out_ptr.add(16), out1);
                in_ptr = in_ptr.add(16);
                out_ptr = out_ptr.add(32);
            }
            rem -= 16;
        }

        if rem >= 8 {
            // SAFETY: `rem >= 8` guarantees 8 readable bytes at `in_ptr`.
            let v8 = unsafe { vld1_u8(in_ptr) };
            let (out, _) = encode_16_neon(vcombine_u8(v8, v8), hex_tbl, mask_0f);
            // SAFETY: `rem * 2 >= 16` guarantees 16 writable bytes at `out_ptr`.
            unsafe { vst1q_u8(out_ptr, out) };
        }

        (consumed, consumed * 2)
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_16 * 16` reads, `out_ptr` is valid for
/// `chunks_16 * 8` writes, and NEON is enabled. `out_ptr` must either not overlap `in_ptr` or
/// equal `in_ptr` (each iteration loads 32 or 16 input bytes before writing 16 or 8 output bytes,
/// so `out_ptr` never overtakes unread input).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_neon<const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_16: usize,
) -> usize {
    let hi_mask_tbl = load_16(select_hi_mask::<UPPERCASE, STRICT_CASE>());
    let lo_mask_tbl = load_16(&LO_MASK);
    let hi_off_tbl = load_16(&HI_OFFSET);
    let mask_0f = vdupq_n_u8(0x0f);
    let mut done_16 = 0;
    let mut saw_invalid = false;

    while done_16 + 2 <= chunks_16 {
        // SAFETY: `done_16 + 2 <= chunks_16` guarantees 32 readable bytes at `in_ptr`.
        let uint8x16x2_t(even, odd) = unsafe { vld2q_u8(in_ptr) };
        let Some(packed) =
            decode_pair_neon(even, odd, hi_mask_tbl, lo_mask_tbl, hi_off_tbl, mask_0f)
        else {
            saw_invalid = true;
            break;
        };
        // SAFETY: `done_16 + 2 <= chunks_16` guarantees 16 writable bytes at `out_ptr`, in-place
        // writes lag behind `in_ptr`, and advancing `in_ptr`/`out_ptr` by 32 and 16 bytes stays
        // within both buffers.
        unsafe {
            vst1q_u8(out_ptr, packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
        }
        done_16 += 2;
    }

    if !saw_invalid && done_16 < chunks_16 {
        // SAFETY: `done_16 < chunks_16` guarantees 16 readable bytes at `in_ptr`.
        let chunk = unsafe { vld1q_u8(in_ptr) };
        if let Some(packed) = decode_16_neon(chunk, hi_mask_tbl, lo_mask_tbl, hi_off_tbl, mask_0f) {
            // SAFETY: `done_16 < chunks_16` guarantees 8 writable bytes at `out_ptr`, and this
            // chunk's input was already loaded so in-place writes do not overwrite unread input.
            unsafe { vst1_u8(out_ptr, packed) };
            done_16 += 1;
        }
    }
    done_16
}

#[inline]
#[target_feature(enable = "neon")]
fn decode_16_neon(
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

#[inline]
#[target_feature(enable = "neon")]
fn decode_pair_neon(
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
