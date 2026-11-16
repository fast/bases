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

//! WebAssembly SIMD128 Base16 kernels.

use core::arch::wasm32::i8x16_add;
use core::arch::wasm32::u8x16;
use core::arch::wasm32::u8x16_all_true;
use core::arch::wasm32::u8x16_min;
use core::arch::wasm32::u8x16_shl;
use core::arch::wasm32::u8x16_shr;
use core::arch::wasm32::u8x16_shuffle;
use core::arch::wasm32::u8x16_splat;
use core::arch::wasm32::u8x16_swizzle;
use core::arch::wasm32::v128;
use core::arch::wasm32::v128_and;
use core::arch::wasm32::v128_load;
use core::arch::wasm32::v128_or;
use core::arch::wasm32::v128_store;

const UPPER_HEX: v128 = u8x16(
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
);
const LOWER_HEX: v128 = u8x16(
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
);

const HI_MASK_MIXED: v128 = u8x16(0, 0, 0, 1, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0);
const HI_MASK_UPPER: v128 = u8x16(0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
const HI_MASK_LOWER: v128 = u8x16(0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0);
const LO_MASK: v128 = u8x16(1, 7, 7, 7, 7, 7, 7, 1, 1, 1, 0, 0, 0, 0, 0, 0);
const HI_OFFSET: v128 = u8x16(0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0);

const fn select_hi_mask<const UPPERCASE: bool, const STRICT_CASE: bool>() -> v128 {
    if !STRICT_CASE {
        HI_MASK_MIXED
    } else if UPPERCASE {
        HI_MASK_UPPER
    } else {
        HI_MASK_LOWER
    }
}

pub fn encode<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    let chunks = (input.len() / 16).min(output.len() / 32);
    if chunks == 0 {
        return (0, 0);
    }

    let hex_tbl = if UPPERCASE { UPPER_HEX } else { LOWER_HEX };
    let mask_0f = u8x16_splat(0x0f);
    let mut in_ptr = input.as_ptr();
    let mut out_ptr = output.as_mut_ptr();

    for _ in 0..chunks {
        // SAFETY: `chunks * 16 <= input.len()`, so `in_ptr` is valid for a 16-byte load.
        let v = unsafe { v128_load(in_ptr.cast()) };
        let hi = u8x16_shr(v, 4);
        let lo = v128_and(v, mask_0f);
        let hi_ascii = u8x16_swizzle(hex_tbl, hi);
        let lo_ascii = u8x16_swizzle(hex_tbl, lo);
        let out0 = u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
            hi_ascii, lo_ascii,
        );
        let out1 = u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
            hi_ascii, lo_ascii,
        );
        // SAFETY: `chunks * 32 <= output.len()`, so `out_ptr` is valid for two 16-byte stores.
        unsafe {
            v128_store(out_ptr.cast(), out0);
            v128_store(out_ptr.add(16).cast(), out1);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(32);
        }
    }

    (chunks * 16, chunks * 32)
}

#[inline]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks = (input.len() / 32).min(output.len() / 16);
    if chunks == 0 {
        return (0, 0);
    }

    // SAFETY: `chunks * 32 <= input.len()` and `chunks * 16 <= output.len()`.
    let done = unsafe {
        decode_wasm32::<UPPERCASE, STRICT_CASE>(input.as_ptr(), output.as_mut_ptr(), chunks)
    };
    (done * 32, done * 16)
}

#[inline]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    let chunks = buf.len() / 32;
    if chunks == 0 {
        return (0, 0);
    }

    let ptr = buf.as_mut_ptr();
    // SAFETY: `chunks * 32 <= buf.len()`, each iteration loads 32 input bytes before writing 16
    // output bytes, and `out_ptr` advances strictly slower than `in_ptr`.
    let done = unsafe { decode_wasm32::<UPPERCASE, STRICT_CASE>(ptr.cast_const(), ptr, chunks) };
    (done * 32, done * 16)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 32` reads and `out_ptr` is valid for
/// `chunks * 16` writes. `out_ptr` must either not overlap `in_ptr` or equal `in_ptr` (each
/// iteration loads 32 input bytes before writing 16 output bytes, so `out_ptr` never overtakes
/// unread input).
#[inline]
unsafe fn decode_wasm32<const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
) -> usize {
    let hi_mask = select_hi_mask::<UPPERCASE, STRICT_CASE>();
    let mask_0f = u8x16_splat(0x0f);
    let mut done = 0;

    while done < chunks {
        // SAFETY: Caller guarantees `chunks * 32` reads at `in_ptr` and `chunks * 16` writes at
        // `out_ptr` without overwriting unread input.
        let (in0, in1) = unsafe { (v128_load(in_ptr.cast()), v128_load(in_ptr.add(16).cast())) };
        let Some(packed) = decode_chunk_32(in0, in1, hi_mask, mask_0f) else {
            break;
        };
        // SAFETY: `out_ptr` is valid for a 16-byte store and does not overwrite future `in_ptr`
        // reads.
        unsafe {
            v128_store(out_ptr.cast(), packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
        }
        done += 1;
    }

    done
}

#[inline]
fn decode_chunk_32(in0: v128, in1: v128, hi_mask: v128, mask_0f: v128) -> Option<v128> {
    let hi0 = u8x16_shr(in0, 4);
    let lo0 = v128_and(in0, mask_0f);
    let chk0 = v128_and(u8x16_swizzle(hi_mask, hi0), u8x16_swizzle(LO_MASK, lo0));

    let hi1 = u8x16_shr(in1, 4);
    let lo1 = v128_and(in1, mask_0f);
    let chk1 = v128_and(u8x16_swizzle(hi_mask, hi1), u8x16_swizzle(LO_MASK, lo1));

    if !u8x16_all_true(u8x16_min(chk0, chk1)) {
        return None;
    }

    let nib0 = i8x16_add(lo0, u8x16_swizzle(HI_OFFSET, hi0));
    let nib1 = i8x16_add(lo1, u8x16_swizzle(HI_OFFSET, hi1));
    let even =
        u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(nib0, nib1);
    let odd =
        u8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(nib0, nib1);
    Some(v128_or(u8x16_shl(even, 4), odd))
}
