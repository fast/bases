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

//! WebAssembly SIMD128 Base64 kernels.

use core::arch::wasm32::i8x16_add;
use core::arch::wasm32::i8x16_gt;
use core::arch::wasm32::i8x16_sub;
use core::arch::wasm32::i16x8;
use core::arch::wasm32::i32x4_dot_i16x8;
use core::arch::wasm32::u8x16;
use core::arch::wasm32::u8x16_all_true;
use core::arch::wasm32::u8x16_eq;
use core::arch::wasm32::u8x16_shr;
use core::arch::wasm32::u8x16_shuffle;
use core::arch::wasm32::u8x16_splat;
use core::arch::wasm32::u8x16_sub_sat;
use core::arch::wasm32::u8x16_swizzle;
use core::arch::wasm32::u16x8_add;
use core::arch::wasm32::u16x8_shl;
use core::arch::wasm32::u16x8_shr;
use core::arch::wasm32::u16x8_splat;
use core::arch::wasm32::u32x4_extract_lane;
use core::arch::wasm32::u32x4_shl;
use core::arch::wasm32::u32x4_shr;
use core::arch::wasm32::u32x4_splat;
use core::arch::wasm32::u64x2_extract_lane;
use core::arch::wasm32::v128;
use core::arch::wasm32::v128_and;
use core::arch::wasm32::v128_load;
use core::arch::wasm32::v128_or;
use core::arch::wasm32::v128_store;

const SHUF_12: v128 = u8x16(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10);

const OFFSET_STD: v128 = u8x16(
    71, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 237, 240, 65, 0, 0,
);
const OFFSET_URL: v128 = u8x16(
    71, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 239, 32, 65, 0, 0,
);

const HI_MASK: v128 = u8x16(0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0);
const LO_MASK_STD: v128 = u8x16(
    0x1a, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1c, 0x05, 0x04, 0x04, 0x04, 0x05,
);
const LO_MASK_URL: v128 = u8x16(
    0x1a, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1c, 0x04, 0x04, 0x05, 0x04, 0x0c,
);

const ROLL_STD: v128 = u8x16(0, 0, 19, 4, 0xbf, 0xbf, 0xb9, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0);
const ROLL_URL: v128 = u8x16(0, 0, 17, 4, 0xbf, 0xbf, 0xb9, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0);

#[inline]
pub fn encode<const URL_SAFE_ALPHABET: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    if input.len() < 16 {
        return (0, 0);
    }
    let chunks = ((input.len() - 4) / 12).min(output.len() / 16);
    if chunks == 0 {
        return (0, 0);
    }

    let offset_lut = if URL_SAFE_ALPHABET {
        OFFSET_URL
    } else {
        OFFSET_STD
    };
    let mask_idx0 = u32x4_splat(0x0000_FC00);
    let mask_idx1 = u32x4_splat(0x0000_03F0);
    let mask_idx2 = u32x4_splat(0x0FC0_0000);
    let mask_idx3 = u32x4_splat(0x003F_0000);
    let c51 = u8x16_splat(51);
    let c26 = u8x16_splat(26);
    let c13 = u8x16_splat(13);

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = output.as_mut_ptr();

    for _ in 0..chunks {
        // SAFETY: `chunks > 0` implies `input.len() >= (chunks - 1) * 12 + 16`.
        let raw = unsafe { v128_load(in_ptr.cast()) };
        let w = u8x16_swizzle(raw, SHUF_12);
        let idx0 = u32x4_shr(v128_and(w, mask_idx0), 10);
        let idx1 = u32x4_shl(v128_and(w, mask_idx1), 4);
        let idx2 = u32x4_shr(v128_and(w, mask_idx2), 6);
        let idx3 = u32x4_shl(v128_and(w, mask_idx3), 8);
        let indices = v128_or(v128_or(idx0, idx1), v128_or(idx2, idx3));
        let result = u8x16_sub_sat(indices, c51);
        let less = i8x16_gt(c26, indices);
        let key = i8x16_add(result, v128_and(less, c13));
        let ascii = i8x16_add(indices, u8x16_swizzle(offset_lut, key));
        // SAFETY: `chunks * 16 <= output.len()`, so `out_ptr` is valid for a 16-byte store.
        unsafe {
            v128_store(out_ptr.cast(), ascii);
            in_ptr = in_ptr.add(12);
            out_ptr = out_ptr.add(16);
        }
    }

    (chunks * 12, chunks * 16)
}

#[inline]
pub fn decode<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize, bool) {
    let chunks = (input.len() / 16).min(output.len() / 12);
    if chunks == 0 {
        return (0, 0, false);
    }

    let mut input_index = 0;
    let mut output_index = 0;
    let mut done = 0;
    let mut had_invalid = false;

    while done < chunks {
        // SAFETY: `chunks * 16 <= input.len()`, so `input_index + 16 <= input.len()`.
        let v = unsafe { v128_load(input.as_ptr().add(input_index).cast()) };
        let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(v) else {
            had_invalid = true;
            break;
        };
        let lo = u64x2_extract_lane::<0>(res);
        let hi = u32x4_extract_lane::<2>(res);
        output[output_index..output_index + 8].copy_from_slice(&lo.to_le_bytes());
        output[output_index + 8..output_index + 12].copy_from_slice(&hi.to_le_bytes());
        input_index += 16;
        output_index += 12;
        done += 1;
    }

    (input_index, output_index, had_invalid)
}

#[inline]
pub fn decode_in_place<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    complete_len: usize,
) -> (usize, usize, bool) {
    let chunks = complete_len.min(buf.len()) / 16;
    if chunks == 0 {
        return (0, 0, false);
    }

    let mut input_index = 0;
    let mut output_index = 0;
    let mut done = 0;
    let mut had_invalid = false;

    while done < chunks {
        // SAFETY: `chunks * 16 <= buf.len()`. `v128_load` reads `buf[input_index..input_index +
        // 16]` completely before writing `buf[output_index..output_index + 12]`.
        let v = unsafe { v128_load(buf.as_ptr().add(input_index).cast()) };
        let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(v) else {
            had_invalid = true;
            break;
        };
        let lo = u64x2_extract_lane::<0>(res);
        let hi = u32x4_extract_lane::<2>(res);
        buf[output_index..output_index + 8].copy_from_slice(&lo.to_le_bytes());
        buf[output_index + 8..output_index + 12].copy_from_slice(&hi.to_le_bytes());
        input_index += 16;
        output_index += 12;
        done += 1;
    }

    (input_index, output_index, had_invalid)
}

#[inline]
fn decode_chunk_16<const URL_SAFE_ALPHABET: bool>(v: v128) -> Option<v128> {
    let lo_mask = if URL_SAFE_ALPHABET {
        LO_MASK_URL
    } else {
        LO_MASK_STD
    };
    let roll_tbl = if URL_SAFE_ALPHABET {
        ROLL_URL
    } else {
        ROLL_STD
    };
    let mask_0f = u8x16_splat(0x0f);
    let hi = u8x16_shr(v, 4);
    let lo = v128_and(v, mask_0f);
    let chk = v128_and(u8x16_swizzle(HI_MASK, hi), u8x16_swizzle(lo_mask, lo));
    if !u8x16_all_true(chk) {
        return None;
    }

    let base = i8x16_add(v, u8x16_swizzle(roll_tbl, hi));
    let special_ch = u8x16_splat(if URL_SAFE_ALPHABET { b'_' } else { b'/' });
    let special_adj = u8x16_splat(if URL_SAFE_ALPHABET { 33 } else { 3 });
    let adj = v128_and(u8x16_eq(v, special_ch), special_adj);
    let vals = if URL_SAFE_ALPHABET {
        i8x16_add(base, adj)
    } else {
        i8x16_sub(base, adj)
    };

    let merged = u16x8_add(
        u16x8_shl(v128_and(vals, u16x8_splat(0x00ff)), 6),
        u16x8_shr(vals, 8),
    );
    let packed = i32x4_dot_i16x8(merged, i16x8(4096, 1, 4096, 1, 4096, 1, 4096, 1));
    Some(u8x16_shuffle::<
        2,
        1,
        0,
        6,
        5,
        4,
        10,
        9,
        8,
        14,
        13,
        12,
        3,
        7,
        11,
        15,
    >(packed, packed))
}
