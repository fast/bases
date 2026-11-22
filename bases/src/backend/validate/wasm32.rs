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

//! WebAssembly SIMD128 nibble-lut validation kernels.

use core::arch::wasm32::u8x16_all_true;
use core::arch::wasm32::u8x16_bitmask;
use core::arch::wasm32::u8x16_eq;
use core::arch::wasm32::u8x16_min;
use core::arch::wasm32::u8x16_shr;
use core::arch::wasm32::u8x16_splat;
use core::arch::wasm32::u8x16_swizzle;
use core::arch::wasm32::v128;
use core::arch::wasm32::v128_and;
use core::arch::wasm32::v128_load;

const BATCH_LEN: usize = 64;

#[inline(always)]
pub fn validate(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    let len = input.len();
    if len < 16 {
        return Ok(0);
    }

    let hi_tbl = load_16(hi_mask);
    let lo_tbl = load_16(lo_mask);
    let mask_0f = u8x16_splat(0x0f);
    let mut checked = 0;

    for batch in input.as_chunks::<BATCH_LEN>().0 {
        if !u8x16_all_true(classify_64(batch, hi_tbl, lo_tbl, mask_0f)) {
            return find_invalid_from(input, checked, hi_mask, lo_mask);
        }
        checked += BATCH_LEN;
    }

    let rem = len - checked;
    if rem != 0 {
        let agg = if len < 32 {
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
                u8x16_min(c0, classify_16(last, hi_tbl, lo_tbl, mask_0f))
            }
        } else if rem <= 16 {
            let last = input[len - 16..]
                .first_chunk::<16>()
                .expect("input has at least 16 bytes");
            classify_16(last, hi_tbl, lo_tbl, mask_0f)
        } else if rem <= 32 {
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            classify_32(last, hi_tbl, lo_tbl, mask_0f)
        } else {
            let head = input[checked..]
                .first_chunk::<32>()
                .expect("remaining tail has at least 32 bytes");
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            u8x16_min(
                classify_32(head, hi_tbl, lo_tbl, mask_0f),
                classify_32(last, hi_tbl, lo_tbl, mask_0f),
            )
        };
        if !u8x16_all_true(agg) {
            return find_invalid_from(input, checked, hi_mask, lo_mask);
        }
    }

    Ok(len)
}

#[cold]
#[inline(never)]
fn find_invalid_from(
    input: &[u8],
    mut offset: usize,
    hi_mask: &[u8; 16],
    lo_mask: &[u8; 16],
) -> Result<usize, usize> {
    let hi_tbl = load_16(hi_mask);
    let lo_tbl = load_16(lo_mask);
    let mask_0f = u8x16_splat(0x0f);
    let zero = u8x16_splat(0);
    for block in input[offset..].as_chunks::<16>().0 {
        let invalid = u8x16_bitmask(u8x16_eq(classify_16(block, hi_tbl, lo_tbl, mask_0f), zero));
        if invalid != 0 {
            return Err(offset + invalid.trailing_zeros() as usize);
        }
        offset += 16;
    }
    let start = input.len() - 16;
    let last = input[start..]
        .first_chunk::<16>()
        .expect("input has at least 16 bytes");
    let invalid = u8x16_bitmask(u8x16_eq(classify_16(last, hi_tbl, lo_tbl, mask_0f), zero));
    debug_assert_ne!(invalid, 0);
    Err(start + invalid.trailing_zeros() as usize)
}

#[inline(always)]
fn load_16(chunk: &[u8; 16]) -> v128 {
    // SAFETY: `chunk` contains 16 initialized readable bytes.
    unsafe { v128_load(chunk.as_ptr().cast()) }
}

#[inline(always)]
fn classify_16(chunk: &[u8; 16], hi_tbl: v128, lo_tbl: v128, mask_0f: v128) -> v128 {
    let v = load_16(chunk);
    let hi = u8x16_shr(v, 4);
    let lo = v128_and(v, mask_0f);
    v128_and(u8x16_swizzle(hi_tbl, hi), u8x16_swizzle(lo_tbl, lo))
}

#[inline(always)]
fn classify_32(chunk: &[u8; 32], hi_tbl: v128, lo_tbl: v128, mask_0f: v128) -> v128 {
    let blocks = chunk.as_chunks::<16>().0;
    u8x16_min(
        classify_16(&blocks[0], hi_tbl, lo_tbl, mask_0f),
        classify_16(&blocks[1], hi_tbl, lo_tbl, mask_0f),
    )
}

#[inline(always)]
fn classify_64(chunk: &[u8; 64], hi_tbl: v128, lo_tbl: v128, mask_0f: v128) -> v128 {
    let halves = chunk.as_chunks::<32>().0;
    u8x16_min(
        classify_32(&halves[0], hi_tbl, lo_tbl, mask_0f),
        classify_32(&halves[1], hi_tbl, lo_tbl, mask_0f),
    )
}
