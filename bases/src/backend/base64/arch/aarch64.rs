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

//! AArch64 NEON Base64 kernels.

use core::arch::aarch64::int8x16_t;
use core::arch::aarch64::uint8x16_t;
use core::arch::aarch64::uint8x16x3_t;
use core::arch::aarch64::uint8x16x4_t;
use core::arch::aarch64::vaddq_u8;
use core::arch::aarch64::vandq_u8;
use core::arch::aarch64::vceqq_u8;
use core::arch::aarch64::vcombine_u8;
use core::arch::aarch64::vdup_n_u32;
use core::arch::aarch64::vdupq_n_u8;
use core::arch::aarch64::vextq_u8;
use core::arch::aarch64::vget_low_u8;
use core::arch::aarch64::vgetq_lane_u32;
use core::arch::aarch64::vld1_u8;
use core::arch::aarch64::vld1q_s8;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vld3q_u8;
use core::arch::aarch64::vld4q_u8;
use core::arch::aarch64::vminq_u8;
use core::arch::aarch64::vminvq_u8;
use core::arch::aarch64::vorrq_u8;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vqtbl4q_u8;
use core::arch::aarch64::vreinterpret_u8_u32;
use core::arch::aarch64::vreinterpretq_u32_u8;
use core::arch::aarch64::vset_lane_u32;
use core::arch::aarch64::vshlq_n_u8;
use core::arch::aarch64::vshlq_u8;
use core::arch::aarch64::vshrq_n_u8;
use core::arch::aarch64::vst1_u8;
use core::arch::aarch64::vst1q_u8;
use core::arch::aarch64::vst3q_u8;
use core::arch::aarch64::vst4q_u8;
use core::arch::aarch64::vsubq_u8;
use core::ptr;

const STANDARD: [u8; 64] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const URL_SAFE: [u8; 64] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

const ENC_SHUF_A: [u8; 16] = [
    0xff, 0, 1, 0xff, 0xff, 3, 4, 0xff, 0xff, 6, 7, 0xff, 0xff, 9, 10, 0xff,
];
const ENC_SHL_A: [i8; 16] = [0, 4, 2, 0, 0, 4, 2, 0, 0, 4, 2, 0, 0, 4, 2, 0];
const ENC_SHUF_B: [u8; 16] = [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11];
const ENC_SHR_B: [i8; 16] = [-2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0];

const DEC_SHL: [i8; 16] = [2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0];
const DEC_SHR: [i8; 16] = [-4, -2, 0, 0, -4, -2, 0, 0, -4, -2, 0, 0, -4, -2, 0, 0];
const DEC_PACK: [u8; 16] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0xff, 0xff, 0xff, 0xff,
];

const HI_MASK: [u8; 16] = [0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0];
const LO_MASK_STD: [u8; 16] = [
    0x1a, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1c, 0x05, 0x04, 0x04, 0x04, 0x05,
];
const LO_MASK_URL: [u8; 16] = [
    0x1a, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1c, 0x04, 0x04, 0x05, 0x04, 0x0c,
];

const ROLL_STD: [u8; 16] = [0, 0, 19, 4, 0xbf, 0xbf, 0xb9, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0];
const ROLL_URL: [u8; 16] = [0, 0, 17, 4, 0xbf, 0xbf, 0xb9, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline]
pub fn encode<const URL_SAFE_ALPHABET: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    let chunks_12 = (input.len() / 12).min(output.len() / 16);
    if chunks_12 == 0 {
        return (0, 0);
    }

    let consumed_in = chunks_12 * 12;
    let consumed_out = chunks_12 * 16;
    // SAFETY: `consumed_in <= input.len()`, `consumed_out <= output.len()`, and NEON is baseline.
    unsafe {
        encode_neon::<URL_SAFE_ALPHABET>(
            input.as_ptr(),
            output.as_mut_ptr(),
            consumed_in,
            input.len(),
        );
    }
    (consumed_in, consumed_out)
}

#[inline]
pub fn decode<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize, bool) {
    let limit_16 = (input.len() / 16).min(output.len() / 12);
    if limit_16 == 0 {
        return (0, 0, false);
    }

    // SAFETY: `limit_16 * 16 <= input.len()` and `limit_16 * 12 <= output.len()`, and NEON is
    // baseline.
    let (done_16, had_invalid) =
        unsafe { decode_neon::<URL_SAFE_ALPHABET>(input.as_ptr(), output.as_mut_ptr(), limit_16) };
    (done_16 * 16, done_16 * 12, had_invalid)
}

#[inline]
pub fn decode_in_place<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    complete_len: usize,
) -> (usize, usize, bool) {
    let limit_16 = complete_len.min(buf.len()) / 16;
    if limit_16 == 0 {
        return (0, 0, false);
    }

    let ptr = buf.as_mut_ptr();
    // SAFETY: `limit_16 * 16 <= buf.len()`. Every iteration loads 64 or 16 input bytes before
    // storing 48 or 12 decoded bytes, and `out_ptr` advances strictly slower than `in_ptr`.
    let (done_16, had_invalid) =
        unsafe { decode_neon::<URL_SAFE_ALPHABET>(ptr.cast_const(), ptr, limit_16) };
    (done_16 * 16, done_16 * 12, had_invalid)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `rem_total_in >= 12` reads and NEON is enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_12_neon(in_ptr: *const u8, rem_total_in: usize) -> uint8x16_t {
    // SAFETY: Caller guarantees at least `rem_total_in >= 12` readable bytes at `in_ptr`.
    unsafe {
        if rem_total_in >= 16 {
            vld1q_u8(in_ptr)
        } else {
            let lo = vld1_u8(in_ptr);
            let hi32 = ptr::read_unaligned(in_ptr.add(8).cast::<u32>());
            let hi = vreinterpret_u8_u32(vset_lane_u32::<0>(hi32, vdup_n_u32(0)));
            vcombine_u8(lo, hi)
        }
    }
}

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 12 writes and NEON is enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_12_neon(out_ptr: *mut u8, packed: uint8x16_t) {
    // SAFETY: Caller guarantees `out_ptr` is valid for 12 writes.
    unsafe {
        vst1_u8(out_ptr, vget_low_u8(packed));
        ptr::write_unaligned(
            out_ptr.add(8).cast::<u32>(),
            vgetq_lane_u32::<2>(vreinterpretq_u32_u8(packed)),
        );
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `total_in >= consumed_in` reads (multiple of 12) and
/// `out_ptr` is valid for `(consumed_in / 12) * 16` non-overlapping writes, with NEON enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn encode_neon<const URL_SAFE_ALPHABET: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    mut rem_in: usize,
    mut rem_total_in: usize,
) {
    let alpha_ptr = if URL_SAFE_ALPHABET {
        &URL_SAFE
    } else {
        &STANDARD
    }
    .as_ptr();
    // SAFETY: Caller guarantees sufficient readable/writable bytes and static tables are valid.
    unsafe {
        let alphabet_tbl = uint8x16x4_t(
            vld1q_u8(alpha_ptr),
            vld1q_u8(alpha_ptr.add(16)),
            vld1q_u8(alpha_ptr.add(32)),
            vld1q_u8(alpha_ptr.add(48)),
        );
        let mask_3f = vdupq_n_u8(0x3f);

        while rem_in >= 48 {
            let uint8x16x3_t(b0, b1, b2) = vld3q_u8(in_ptr);
            let idx0 = vshrq_n_u8::<2>(b0);
            let idx1 = vandq_u8(vorrq_u8(vshlq_n_u8::<4>(b0), vshrq_n_u8::<4>(b1)), mask_3f);
            let idx2 = vandq_u8(vorrq_u8(vshlq_n_u8::<2>(b1), vshrq_n_u8::<6>(b2)), mask_3f);
            let idx3 = vandq_u8(b2, mask_3f);

            let out0 = vqtbl4q_u8(alphabet_tbl, idx0);
            let out1 = vqtbl4q_u8(alphabet_tbl, idx1);
            let out2 = vqtbl4q_u8(alphabet_tbl, idx2);
            let out3 = vqtbl4q_u8(alphabet_tbl, idx3);
            vst4q_u8(out_ptr, uint8x16x4_t(out0, out1, out2, out3));

            in_ptr = in_ptr.add(48);
            out_ptr = out_ptr.add(64);
            rem_in -= 48;
            rem_total_in -= 48;
        }

        if rem_in >= 12 {
            let shuf_a = vld1q_u8(ENC_SHUF_A.as_ptr());
            let shl_a = vld1q_s8(ENC_SHL_A.as_ptr());
            let shuf_b = vld1q_u8(ENC_SHUF_B.as_ptr());
            let shr_b = vld1q_s8(ENC_SHR_B.as_ptr());

            while rem_in >= 12 {
                let v = load_12_neon(in_ptr, rem_total_in);
                let part_a = vshlq_u8(vqtbl1q_u8(v, shuf_a), shl_a);
                let part_b = vshlq_u8(vqtbl1q_u8(v, shuf_b), shr_b);
                let idx = vandq_u8(vorrq_u8(part_a, part_b), mask_3f);
                vst1q_u8(out_ptr, vqtbl4q_u8(alphabet_tbl, idx));

                in_ptr = in_ptr.add(12);
                out_ptr = out_ptr.add(16);
                rem_in -= 12;
                rem_total_in -= 12;
            }
        }
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `limit_16 * 16` reads, `out_ptr` is valid for
/// `limit_16 * 12` writes, and NEON is enabled. `out_ptr` must either not overlap `in_ptr` or equal
/// `in_ptr` (each iteration loads 64 or 16 input bytes before writing 48 or 12 output bytes, so
/// `out_ptr` never overtakes unread input).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_neon<const URL_SAFE_ALPHABET: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    limit_16: usize,
) -> (usize, bool) {
    let lo_mask_ptr = if URL_SAFE_ALPHABET {
        &LO_MASK_URL
    } else {
        &LO_MASK_STD
    }
    .as_ptr();
    let roll_ptr = if URL_SAFE_ALPHABET {
        &ROLL_URL
    } else {
        &ROLL_STD
    }
    .as_ptr();
    let mut done_16 = 0;

    // SAFETY: Caller guarantees `limit_16 * 16` reads at `in_ptr`, `limit_16 * 12` writes at
    // `out_ptr`, non-overtaking pointer advancement for in-place decoding, and 16-byte static
    // lookup tables.
    unsafe {
        let hi_mask_tbl = vld1q_u8(HI_MASK.as_ptr());
        let lo_mask_tbl = vld1q_u8(lo_mask_ptr);
        let roll_tbl = vld1q_u8(roll_ptr);
        let mask_0f = vdupq_n_u8(0x0f);
        let special_ch = vdupq_n_u8(if URL_SAFE_ALPHABET { b'_' } else { b'/' });
        let special_adj = vdupq_n_u8(if URL_SAFE_ALPHABET { 33 } else { 3 });

        let limit_64_as_16 = (limit_16 / 4) * 4;
        while done_16 < limit_64_as_16 {
            let uint8x16x4_t(c0, c1, c2, c3) = vld4q_u8(in_ptr);
            let Some(packed) = decode_chunk_64_neon::<URL_SAFE_ALPHABET>(
                c0,
                c1,
                c2,
                c3,
                hi_mask_tbl,
                lo_mask_tbl,
                roll_tbl,
                mask_0f,
                special_ch,
                special_adj,
            ) else {
                break;
            };
            vst3q_u8(out_ptr, packed);
            in_ptr = in_ptr.add(64);
            out_ptr = out_ptr.add(48);
            done_16 += 4;
        }

        if done_16 < limit_16 {
            let dec_shl = vld1q_s8(DEC_SHL.as_ptr());
            let dec_shr = vld1q_s8(DEC_SHR.as_ptr());
            let dec_pack = vld1q_u8(DEC_PACK.as_ptr());

            while done_16 < limit_16 {
                let c = vld1q_u8(in_ptr);
                let Some(packed) = decode_chunk_16_neon::<URL_SAFE_ALPHABET>(
                    c,
                    hi_mask_tbl,
                    lo_mask_tbl,
                    roll_tbl,
                    mask_0f,
                    special_ch,
                    special_adj,
                    dec_shl,
                    dec_shr,
                    dec_pack,
                ) else {
                    return (done_16, true);
                };
                store_12_neon(out_ptr, packed);
                in_ptr = in_ptr.add(16);
                out_ptr = out_ptr.add(12);
                done_16 += 1;
            }
        }
    }
    (done_16, false)
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn decode_lane_neon<const URL_SAFE_ALPHABET: bool>(
    c: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    roll_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
    special_ch: uint8x16_t,
    special_adj: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let hi = vshrq_n_u8::<4>(c);
    let lo = vandq_u8(c, mask_0f);
    let chk = vandq_u8(vqtbl1q_u8(hi_mask_tbl, hi), vqtbl1q_u8(lo_mask_tbl, lo));
    let base = vaddq_u8(c, vqtbl1q_u8(roll_tbl, hi));
    let adj = vandq_u8(vceqq_u8(c, special_ch), special_adj);
    let val = if URL_SAFE_ALPHABET {
        vaddq_u8(base, adj)
    } else {
        vsubq_u8(base, adj)
    };
    (chk, val)
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_16_neon<const URL_SAFE_ALPHABET: bool>(
    c: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    roll_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
    special_ch: uint8x16_t,
    special_adj: uint8x16_t,
    dec_shl: int8x16_t,
    dec_shr: int8x16_t,
    dec_pack: uint8x16_t,
) -> Option<uint8x16_t> {
    // SAFETY: Caller guarantees NEON is enabled.
    unsafe {
        let (chk, v) = decode_lane_neon::<URL_SAFE_ALPHABET>(
            c,
            hi_mask_tbl,
            lo_mask_tbl,
            roll_tbl,
            mask_0f,
            special_ch,
            special_adj,
        );
        if vminvq_u8(chk) == 0 {
            return None;
        }
        let v_next = vextq_u8::<1>(v, v);
        let merged = vorrq_u8(vshlq_u8(v, dec_shl), vshlq_u8(v_next, dec_shr));
        Some(vqtbl1q_u8(merged, dec_pack))
    }
}

/// # Safety
///
/// Requires NEON target feature.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_64_neon<const URL_SAFE_ALPHABET: bool>(
    c0: uint8x16_t,
    c1: uint8x16_t,
    c2: uint8x16_t,
    c3: uint8x16_t,
    hi_mask_tbl: uint8x16_t,
    lo_mask_tbl: uint8x16_t,
    roll_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
    special_ch: uint8x16_t,
    special_adj: uint8x16_t,
) -> Option<uint8x16x3_t> {
    // SAFETY: Caller guarantees NEON is enabled.
    unsafe {
        let (chk0, v0) = decode_lane_neon::<URL_SAFE_ALPHABET>(
            c0,
            hi_mask_tbl,
            lo_mask_tbl,
            roll_tbl,
            mask_0f,
            special_ch,
            special_adj,
        );
        let (chk1, v1) = decode_lane_neon::<URL_SAFE_ALPHABET>(
            c1,
            hi_mask_tbl,
            lo_mask_tbl,
            roll_tbl,
            mask_0f,
            special_ch,
            special_adj,
        );
        let (chk2, v2) = decode_lane_neon::<URL_SAFE_ALPHABET>(
            c2,
            hi_mask_tbl,
            lo_mask_tbl,
            roll_tbl,
            mask_0f,
            special_ch,
            special_adj,
        );
        let (chk3, v3) = decode_lane_neon::<URL_SAFE_ALPHABET>(
            c3,
            hi_mask_tbl,
            lo_mask_tbl,
            roll_tbl,
            mask_0f,
            special_ch,
            special_adj,
        );

        let min_chk = vminq_u8(vminq_u8(chk0, chk1), vminq_u8(chk2, chk3));
        if vminvq_u8(min_chk) == 0 {
            return None;
        }

        let out0 = vorrq_u8(vshlq_n_u8::<2>(v0), vshrq_n_u8::<4>(v1));
        let out1 = vorrq_u8(vshlq_n_u8::<4>(v1), vshrq_n_u8::<2>(v2));
        let out2 = vorrq_u8(vshlq_n_u8::<6>(v2), v3);
        Some(uint8x16x3_t(out0, out1, out2))
    }
}
