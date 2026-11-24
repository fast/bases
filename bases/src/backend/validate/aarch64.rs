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

//! AArch64 NEON nibble-lut validation kernels.

use core::arch::aarch64::uint8x16_t;
use core::arch::aarch64::vandq_u8;
use core::arch::aarch64::vceqq_u8;
use core::arch::aarch64::vdupq_n_u8;
use core::arch::aarch64::vget_lane_u64;
use core::arch::aarch64::vld1q_u8;
use core::arch::aarch64::vminq_u8;
use core::arch::aarch64::vminvq_u8;
use core::arch::aarch64::vqtbl1q_u8;
use core::arch::aarch64::vreinterpret_u64_u8;
use core::arch::aarch64::vreinterpretq_u16_u8;
use core::arch::aarch64::vshrn_n_u16;
use core::arch::aarch64::vshrq_n_u8;

use crate::backend::simd::NeonToken;
use crate::backend::simd::simd_fn;

const BATCH_LEN: usize = 128;

#[inline(always)]
pub fn validate(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    if input.len() < 16 {
        return Ok(0);
    }
    validate_neon(NeonToken::summon(), input, hi_mask, lo_mask)
}

simd_fn! {
    fn validate_neon(
        _token: NeonToken,
        input: &[u8],
        hi_mask: &[u8; 16],
        lo_mask: &[u8; 16],
    ) -> Result<usize, usize> {
        let len = input.len();
        if len >= BATCH_LEN {
            return validate_bulk(input, hi_mask, lo_mask);
        }

        let hi_tbl = load_16(hi_mask);
        let lo_tbl = load_16(lo_mask);
        let mask_0f = vdupq_n_u8(0x0f);
        let agg = if len <= 32 {
            let first = input
                .first_chunk::<16>()
                .expect("input has at least 16 bytes");
            let c0 = classify_16(first, hi_tbl, lo_tbl, mask_0f);
            if len == 16 {
                c0
            } else {
                let last = input[len - 16..]
                    .first_chunk::<16>()
                    .expect("input has at least 16 bytes");
                vminq_u8(c0, classify_16(last, hi_tbl, lo_tbl, mask_0f))
            }
        } else if len <= 64 {
            let first = input
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            vminq_u8(
                classify_32(first, hi_tbl, lo_tbl, mask_0f),
                classify_32(last, hi_tbl, lo_tbl, mask_0f),
            )
        } else {
            let first = input
                .first_chunk::<64>()
                .expect("input has at least 64 bytes");
            let last = input[len - 64..]
                .first_chunk::<64>()
                .expect("input has at least 64 bytes");
            vminq_u8(
                classify_64(first, hi_tbl, lo_tbl, mask_0f),
                classify_64(last, hi_tbl, lo_tbl, mask_0f),
            )
        };

        if vminvq_u8(agg) == 0 {
            return find_invalid_from(input, 0, hi_mask, lo_mask);
        }
        Ok(len)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_16(chunk: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `chunk` contains 16 initialized readable bytes.
    unsafe { vld1q_u8(chunk.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn classify_16(
    chunk: &[u8; 16],
    hi_tbl: uint8x16_t,
    lo_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> uint8x16_t {
    let v = load_16(chunk);
    let hi = vshrq_n_u8::<4>(v);
    let lo = vandq_u8(v, mask_0f);
    vandq_u8(vqtbl1q_u8(hi_tbl, hi), vqtbl1q_u8(lo_tbl, lo))
}

#[inline]
#[target_feature(enable = "neon")]
fn classify_32(
    chunk: &[u8; 32],
    hi_tbl: uint8x16_t,
    lo_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> uint8x16_t {
    let blocks = chunk.as_chunks::<16>().0;
    vminq_u8(
        classify_16(&blocks[0], hi_tbl, lo_tbl, mask_0f),
        classify_16(&blocks[1], hi_tbl, lo_tbl, mask_0f),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn classify_64(
    chunk: &[u8; 64],
    hi_tbl: uint8x16_t,
    lo_tbl: uint8x16_t,
    mask_0f: uint8x16_t,
) -> uint8x16_t {
    let halves = chunk.as_chunks::<32>().0;
    vminq_u8(
        classify_32(&halves[0], hi_tbl, lo_tbl, mask_0f),
        classify_32(&halves[1], hi_tbl, lo_tbl, mask_0f),
    )
}

#[target_feature(enable = "neon")]
#[inline(never)]
fn validate_bulk(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    let len = input.len();
    let hi_tbl = load_16(hi_mask);
    let lo_tbl = load_16(lo_mask);
    let mask_0f = vdupq_n_u8(0x0f);

    // Check the first 32 bytes before advancing to a 16-byte-aligned batch cursor.
    let head = input
        .first_chunk::<32>()
        .expect("bulk input has at least 128 bytes");
    if vminvq_u8(classify_32(head, hi_tbl, lo_tbl, mask_0f)) == 0 {
        return find_invalid_from(input, 0, hi_mask, lo_mask);
    }

    let align = input.as_ptr().align_offset(16);
    let mut checked = if (1..16).contains(&align) {
        16 + align
    } else {
        32
    };

    for batch in input[checked..].as_chunks::<BATCH_LEN>().0 {
        let halves = batch.as_chunks::<64>().0;
        let agg = vminq_u8(
            classify_64(&halves[0], hi_tbl, lo_tbl, mask_0f),
            classify_64(&halves[1], hi_tbl, lo_tbl, mask_0f),
        );
        if vminvq_u8(agg) == 0 {
            return find_invalid_from(input, checked, hi_mask, lo_mask);
        }
        checked += BATCH_LEN;
    }

    let rem = len - checked;
    if rem != 0 {
        let agg = if rem <= 16 {
            let last = input[len - 16..]
                .first_chunk::<16>()
                .expect("input has at least 16 bytes");
            classify_16(last, hi_tbl, lo_tbl, mask_0f)
        } else if rem <= 32 {
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            classify_32(last, hi_tbl, lo_tbl, mask_0f)
        } else if rem <= 64 {
            let last = input[len - 64..]
                .first_chunk::<64>()
                .expect("input has at least 64 bytes");
            classify_64(last, hi_tbl, lo_tbl, mask_0f)
        } else {
            let tail_head = input[checked..]
                .first_chunk::<64>()
                .expect("remaining tail has at least 64 bytes");
            let last = input[len - 64..]
                .first_chunk::<64>()
                .expect("input has at least 64 bytes");
            vminq_u8(
                classify_64(tail_head, hi_tbl, lo_tbl, mask_0f),
                classify_64(last, hi_tbl, lo_tbl, mask_0f),
            )
        };
        if vminvq_u8(agg) == 0 {
            return find_invalid_from(input, checked, hi_mask, lo_mask);
        }
    }

    Ok(len)
}

#[cold]
#[inline(never)]
#[target_feature(enable = "neon")]
fn find_invalid_from(
    input: &[u8],
    mut offset: usize,
    hi_mask: &[u8; 16],
    lo_mask: &[u8; 16],
) -> Result<usize, usize> {
    let hi_tbl = load_16(hi_mask);
    let lo_tbl = load_16(lo_mask);
    let mask_0f = vdupq_n_u8(0x0f);
    for block in input[offset..].as_chunks::<16>().0 {
        let chk = classify_16(block, hi_tbl, lo_tbl, mask_0f);
        if vminvq_u8(chk) == 0 {
            return Err(offset + first_zero_lane(chk));
        }
        offset += 16;
    }
    // Every byte before `offset` is valid, so an overlapping load preserves the first error.
    let start = input.len() - 16;
    let last = input[start..]
        .first_chunk::<16>()
        .expect("input has at least 16 bytes");
    let chk = classify_16(last, hi_tbl, lo_tbl, mask_0f);
    debug_assert_eq!(vminvq_u8(chk), 0);
    Err(start + first_zero_lane(chk))
}

#[inline]
#[target_feature(enable = "neon")]
fn first_zero_lane(chk: uint8x16_t) -> usize {
    let eq_zero = vceqq_u8(chk, vdupq_n_u8(0));
    let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq_zero));
    let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed));
    debug_assert_ne!(mask, 0);
    (mask.trailing_zeros() >> 2) as usize
}
