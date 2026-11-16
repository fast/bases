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

//! x86 / x86_64 SSSE3 and AVX2 Base64 kernels.

#[cfg(target_arch = "x86")]
use core::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86_arch;
use core::mem::MaybeUninit;
use core::ptr;
#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
use std::arch::is_x86_feature_detected;

use x86_arch::__m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::__m256i;
use x86_arch::_mm_add_epi8;
use x86_arch::_mm_and_si128;
use x86_arch::_mm_cmpeq_epi8;
use x86_arch::_mm_cmpgt_epi8;
use x86_arch::_mm_cvtsi128_si32;
use x86_arch::_mm_loadu_si128;
use x86_arch::_mm_madd_epi16;
use x86_arch::_mm_maddubs_epi16;
use x86_arch::_mm_movemask_epi8;
use x86_arch::_mm_mulhi_epu16;
use x86_arch::_mm_mullo_epi16;
use x86_arch::_mm_or_si128;
use x86_arch::_mm_set1_epi8;
use x86_arch::_mm_set1_epi32;
use x86_arch::_mm_setr_epi8;
use x86_arch::_mm_setzero_si128;
use x86_arch::_mm_shuffle_epi8;
use x86_arch::_mm_srli_epi16;
use x86_arch::_mm_srli_si128;
use x86_arch::_mm_storel_epi64;
use x86_arch::_mm_storeu_si128;
use x86_arch::_mm_sub_epi8;
use x86_arch::_mm_subs_epu8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_add_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_and_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_castsi256_si128;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_cmpeq_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_cmpgt_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_extracti128_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_loadu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_madd_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_maddubs_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_movemask_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_mulhi_epu16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_mullo_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_or_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_permutevar8x32_epi32;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set_m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi32;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_setr_epi32;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_setzero_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_shuffle_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_srli_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_storeu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_sub_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_subs_epu8;

const LO_MASK_STD: [i8; 16] = [
    0x1a, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1c, 0x05, 0x04, 0x04, 0x04, 0x05,
];
const LO_MASK_URL: [i8; 16] = [
    0x1a, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x1c, 0x04, 0x04, 0x05, 0x04, 0x0c,
];

const ROLL_STD: [i8; 16] = [0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0];
const ROLL_URL: [i8; 16] = [0, 0, 17, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline]
pub fn encode<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    if input.len() < 16 || output.len() < 16 {
        return (0, 0);
    }

    #[cfg(target_feature = "avx2")]
    {
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        unsafe { encode_avx2::<URL_SAFE_ALPHABET>(input, output) }
    }

    #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime feature detection confirmed AVX2 support.
            unsafe { encode_avx2::<URL_SAFE_ALPHABET>(input, output) }
        } else if is_x86_feature_detected!("ssse3") {
            // SAFETY: Runtime feature detection confirmed SSSE3 support.
            unsafe { encode_ssse3::<URL_SAFE_ALPHABET>(input, output) }
        } else {
            let _ = (input, output);
            (0, 0)
        }
    }

    #[cfg(all(
        not(target_feature = "avx2"),
        not(feature = "std"),
        target_feature = "ssse3"
    ))]
    {
        // SAFETY: `target_feature = "ssse3"` is enabled at compile time.
        unsafe { encode_ssse3::<URL_SAFE_ALPHABET>(input, output) }
    }
}

#[inline]
pub fn decode<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize, bool) {
    let chunks_16 = (input.len() / 16).min(output.len() / 12);
    if chunks_16 == 0 {
        return (0, 0, false);
    }

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr().cast::<u8>();
    let (done_16, had_invalid) = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time and `chunks_16` is
            // bounded by `input` and `output`.
            unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support and `chunks_16` is
                // bounded by `input` and `output`.
                unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16) }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support and `chunks_16` is
                // bounded by `input` and `output`.
                unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16) }
            } else {
                let _ = (in_ptr, out_ptr);
                (0, false)
            }
        }

        #[cfg(all(
            not(target_feature = "avx2"),
            not(feature = "std"),
            target_feature = "ssse3"
        ))]
        {
            // SAFETY: `target_feature = "ssse3"` is enabled at compile time and `chunks_16` is
            // bounded by `input` and `output`.
            unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16) }
        }
    };

    (done_16 * 16, done_16 * 12, had_invalid)
}

#[inline]
pub fn decode_in_place<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    complete_len: usize,
) -> (usize, usize, bool) {
    let chunks_16 = complete_len.min(buf.len()) / 16;
    if chunks_16 == 0 {
        return (0, 0, false);
    }

    let ptr = buf.as_mut_ptr();
    let in_ptr = ptr.cast_const();
    let (done_16, had_invalid) = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time, `chunks_16 * 16 <=
            // buf.len()`, and each step loads its full input before writing fewer output bytes.
            unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support, `chunks_16 * 16 <=
                // buf.len()`, and each step loads its full input before writing fewer output bytes.
                unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16) }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support, `chunks_16 * 16 <=
                // buf.len()`, and each step loads its full input before writing fewer output bytes.
                unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16) }
            } else {
                let _ = (in_ptr, ptr);
                (0, false)
            }
        }

        #[cfg(all(
            not(target_feature = "avx2"),
            not(feature = "std"),
            target_feature = "ssse3"
        ))]
        {
            // SAFETY: `target_feature = "ssse3"` is enabled at compile time, `chunks_16 * 16 <=
            // buf.len()`, and each step loads its full input before writing fewer output bytes.
            unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16) }
        }
    };

    (done_16 * 16, done_16 * 12, had_invalid)
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn encode_128_consts<const URL_SAFE_ALPHABET: bool>() -> (
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
) {
    let shuf = _mm_setr_epi8(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10);
    let mask0 = _mm_set1_epi32(0x0FC0_FC00);
    let mul0 = _mm_set1_epi32(0x0400_0040);
    let mask1 = _mm_set1_epi32(0x003F_03F0);
    let mul1 = _mm_set1_epi32(0x0100_0010);
    let c51 = _mm_set1_epi8(51);
    let c26 = _mm_set1_epi8(26);
    let c13 = _mm_set1_epi8(13);
    let offset_lut = if URL_SAFE_ALPHABET {
        _mm_setr_epi8(
            71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -17, 32, 65, 0, 0,
        )
    } else {
        _mm_setr_epi8(
            71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 65, 0, 0,
        )
    };
    (shuf, mask0, mul0, mask1, mul1, c51, c26, c13, offset_lut)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 16 reads and `out_ptr` for 16 writes, with SSSE3
/// enabled.
#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
unsafe fn encode_chunk_12(
    in_ptr: *const u8,
    out_ptr: *mut u8,
    shuf: __m128i,
    mask0: __m128i,
    mul0: __m128i,
    mask1: __m128i,
    mul1: __m128i,
    c51: __m128i,
    c26: __m128i,
    c13: __m128i,
    offset_lut: __m128i,
) {
    // SAFETY: Caller guarantees `in_ptr` is valid for 16 readable bytes and `out_ptr` for 16
    // writes.
    unsafe {
        let raw = _mm_loadu_si128(in_ptr.cast());
        let w = _mm_shuffle_epi8(raw, shuf);
        let t0 = _mm_mulhi_epu16(_mm_and_si128(w, mask0), mul0);
        let t1 = _mm_mullo_epi16(_mm_and_si128(w, mask1), mul1);
        let indices = _mm_or_si128(t0, t1);
        let result = _mm_subs_epu8(indices, c51);
        let less = _mm_cmpgt_epi8(c26, indices);
        let key = _mm_add_epi8(result, _mm_and_si128(less, c13));
        let ascii = _mm_add_epi8(indices, _mm_shuffle_epi8(offset_lut, key));
        _mm_storeu_si128(out_ptr.cast(), ascii);
    }
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[cfg(not(target_feature = "avx2"))]
#[target_feature(enable = "ssse3")]
unsafe fn encode_ssse3<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    if input.len() < 16 {
        return (0, 0);
    }
    let chunks = ((input.len() - 4) / 12).min(output.len() / 16);
    if chunks == 0 {
        return (0, 0);
    }

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = output.as_mut_ptr().cast::<u8>();
    // SAFETY: `chunks > 0` implies `input.len() >= (chunks - 1) * 12 + 16` and `output.len() >=
    // chunks * 16`.
    unsafe {
        let (shuf, mask0, mul0, mask1, mul1, c51, c26, c13, offset_lut) =
            encode_128_consts::<URL_SAFE_ALPHABET>();
        for _ in 0..chunks {
            encode_chunk_12(
                in_ptr, out_ptr, shuf, mask0, mul0, mask1, mul1, c51, c26, c13, offset_lut,
            );
            in_ptr = in_ptr.add(12);
            out_ptr = out_ptr.add(16);
        }
    }
    (chunks * 12, chunks * 16)
}

/// # Safety
///
/// Requires AVX2 support on the current CPU.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    let chunks = if input.len() >= 28 {
        ((input.len() - 4) / 24).min(output.len() / 32)
    } else {
        0
    };
    let mut input_index = chunks * 24;
    let mut output_index = chunks * 32;

    // SAFETY: `chunks > 0` guarantees `input.len() >= (chunks - 1) * 24 + 28` and `output.len() >=
    // chunks * 32`. The trailing 16-byte check ensures `input_index + 16 <= input.len()` and
    // `output_index + 16 <= output.len()`.
    unsafe {
        let (sh_128, m0_128, ml0_128, m1_128, ml1_128, c51_128, c26_128, c13_128, lut_128) =
            encode_128_consts::<URL_SAFE_ALPHABET>();

        if chunks > 0 {
            let shuf = _mm256_set_m128i(sh_128, sh_128);
            let mask0 = _mm256_set_m128i(m0_128, m0_128);
            let mul0 = _mm256_set_m128i(ml0_128, ml0_128);
            let mask1 = _mm256_set_m128i(m1_128, m1_128);
            let mul1 = _mm256_set_m128i(ml1_128, ml1_128);
            let c51 = _mm256_set_m128i(c51_128, c51_128);
            let c26 = _mm256_set_m128i(c26_128, c26_128);
            let c13 = _mm256_set_m128i(c13_128, c13_128);
            let offset_lut = _mm256_set_m128i(lut_128, lut_128);

            let mut in_ptr = input.as_ptr();
            let mut out_ptr = output.as_mut_ptr().cast::<u8>();
            for _ in 0..chunks {
                let raw = _mm256_set_m128i(
                    _mm_loadu_si128(in_ptr.add(12).cast()),
                    _mm_loadu_si128(in_ptr.cast()),
                );
                let w = _mm256_shuffle_epi8(raw, shuf);
                let t0 = _mm256_mulhi_epu16(_mm256_and_si256(w, mask0), mul0);
                let t1 = _mm256_mullo_epi16(_mm256_and_si256(w, mask1), mul1);
                let indices = _mm256_or_si256(t0, t1);
                let result = _mm256_subs_epu8(indices, c51);
                let less = _mm256_cmpgt_epi8(c26, indices);
                let key = _mm256_add_epi8(result, _mm256_and_si256(less, c13));
                let ascii = _mm256_add_epi8(indices, _mm256_shuffle_epi8(offset_lut, key));
                _mm256_storeu_si256(out_ptr.cast(), ascii);
                in_ptr = in_ptr.add(24);
                out_ptr = out_ptr.add(32);
            }
        }

        if input_index + 16 <= input.len() && output_index + 16 <= output.len() {
            encode_chunk_12(
                input.as_ptr().add(input_index),
                output.as_mut_ptr().cast::<u8>().add(output_index),
                sh_128,
                m0_128,
                ml0_128,
                m1_128,
                ml1_128,
                c51_128,
                c26_128,
                c13_128,
                lut_128,
            );
            input_index += 12;
            output_index += 16;
        }
    }

    (input_index, output_index)
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_128_tables<const URL_SAFE_ALPHABET: bool>()
-> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let hi_mask = _mm_setr_epi8(0, 0, 1, 2, 4, 8, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0);
    let lo = if URL_SAFE_ALPHABET {
        LO_MASK_URL
    } else {
        LO_MASK_STD
    };
    let roll = if URL_SAFE_ALPHABET {
        ROLL_URL
    } else {
        ROLL_STD
    };
    let lo_mask = _mm_setr_epi8(
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], lo[8], lo[9], lo[10], lo[11],
        lo[12], lo[13], lo[14], lo[15],
    );
    let roll_tbl = _mm_setr_epi8(
        roll[0], roll[1], roll[2], roll[3], roll[4], roll[5], roll[6], roll[7], roll[8], roll[9],
        roll[10], roll[11], roll[12], roll[13], roll[14], roll[15],
    );
    let pack_shuf = _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
    let special_ch = _mm_set1_epi8(if URL_SAFE_ALPHABET { b'_' } else { b'/' } as i8);
    let special_adj = _mm_set1_epi8(if URL_SAFE_ALPHABET { 33 } else { 3 });
    (
        hi_mask,
        lo_mask,
        roll_tbl,
        pack_shuf,
        special_ch,
        special_adj,
    )
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 16 reads and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_16<const URL_SAFE_ALPHABET: bool>(
    in_ptr: *const u8,
    hi_mask: __m128i,
    lo_mask: __m128i,
    roll_tbl: __m128i,
    pack_shuf: __m128i,
    special_ch: __m128i,
    special_adj: __m128i,
    mask_0f: __m128i,
    mul_0140: __m128i,
    mul_0001_1000: __m128i,
    zero: __m128i,
) -> Option<__m128i> {
    // SAFETY: Caller guarantees `in_ptr` is valid for 16 readable bytes and SSSE3 is enabled.
    unsafe {
        let v = _mm_loadu_si128(in_ptr.cast());
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);
        let lo = _mm_and_si128(v, mask_0f);
        let chk = _mm_and_si128(_mm_shuffle_epi8(hi_mask, hi), _mm_shuffle_epi8(lo_mask, lo));
        if _mm_movemask_epi8(_mm_cmpeq_epi8(chk, zero)) != 0 {
            return None;
        }

        let base = _mm_add_epi8(v, _mm_shuffle_epi8(roll_tbl, hi));
        let adj = _mm_and_si128(_mm_cmpeq_epi8(v, special_ch), special_adj);
        let vals = if URL_SAFE_ALPHABET {
            _mm_add_epi8(base, adj)
        } else {
            _mm_sub_epi8(base, adj)
        };
        let merged = _mm_maddubs_epi16(vals, mul_0140);
        let packed = _mm_madd_epi16(merged, mul_0001_1000);
        Some(_mm_shuffle_epi8(packed, pack_shuf))
    }
}

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 12 writes and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn store_12_ssse3(out_ptr: *mut u8, res: __m128i) {
    // SAFETY: Caller guarantees `out_ptr` is valid for 12 non-overlapping writes.
    unsafe {
        _mm_storel_epi64(out_ptr.cast(), res);
        ptr::write_unaligned(
            out_ptr.add(8).cast::<i32>(),
            _mm_cvtsi128_si32(_mm_srli_si128::<8>(res)),
        );
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads, `out_ptr` is valid for
/// `chunks * 12` writes, and SSSE3 is enabled. `out_ptr` must either not overlap `in_ptr` or equal
/// `in_ptr` (each iteration loads 16 input bytes before writing 12 output bytes, so `out_ptr` never
/// overtakes unread input).
#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const URL_SAFE_ALPHABET: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
) -> (usize, bool) {
    let mut done = 0;
    let mut had_invalid = false;
    // SAFETY: Caller guarantees `chunks * 16` reads at `in_ptr`, `chunks * 12` writes at `out_ptr`,
    // and non-overtaking pointer advancement for in-place decoding.
    unsafe {
        let (hi_mask, lo_mask, roll_tbl, pack_shuf, special_ch, special_adj) =
            decode_128_tables::<URL_SAFE_ALPHABET>();
        let mask_0f = _mm_set1_epi8(0x0f);
        let mul_0140 = _mm_set1_epi32(0x0140_0140);
        let mul_0001_1000 = _mm_set1_epi32(0x0001_1000);
        let zero = _mm_setzero_si128();

        while done < chunks {
            let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(
                in_ptr,
                hi_mask,
                lo_mask,
                roll_tbl,
                pack_shuf,
                special_ch,
                special_adj,
                mask_0f,
                mul_0140,
                mul_0001_1000,
                zero,
            ) else {
                had_invalid = true;
                break;
            };
            store_12_ssse3(out_ptr, res);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(12);
            done += 1;
        }
    }
    (done, had_invalid)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 32 reads and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_32<const URL_SAFE_ALPHABET: bool>(
    in_ptr: *const u8,
    hi_mask: __m256i,
    lo_mask: __m256i,
    roll_tbl: __m256i,
    pack_shuf: __m256i,
    lane_perm: __m256i,
    special_ch: __m256i,
    special_adj: __m256i,
    mask_0f: __m256i,
    mul_0140: __m256i,
    mul_0001_1000: __m256i,
    zero: __m256i,
) -> Option<__m256i> {
    // SAFETY: Caller guarantees `in_ptr` is valid for 32 readable bytes and AVX2 is enabled.
    unsafe {
        let v = _mm256_loadu_si256(in_ptr.cast());
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);
        let lo = _mm256_and_si256(v, mask_0f);
        let chk = _mm256_and_si256(
            _mm256_shuffle_epi8(hi_mask, hi),
            _mm256_shuffle_epi8(lo_mask, lo),
        );
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(chk, zero)) != 0 {
            return None;
        }

        let base = _mm256_add_epi8(v, _mm256_shuffle_epi8(roll_tbl, hi));
        let adj = _mm256_and_si256(_mm256_cmpeq_epi8(v, special_ch), special_adj);
        let vals = if URL_SAFE_ALPHABET {
            _mm256_add_epi8(base, adj)
        } else {
            _mm256_sub_epi8(base, adj)
        };
        let merged = _mm256_maddubs_epi16(vals, mul_0140);
        let packed = _mm256_madd_epi16(merged, mul_0001_1000);
        let shuf = _mm256_shuffle_epi8(packed, pack_shuf);
        Some(_mm256_permutevar8x32_epi32(shuf, lane_perm))
    }
}

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 24 writes and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_24_avx2(out_ptr: *mut u8, res: __m256i) {
    // SAFETY: Caller guarantees `out_ptr` is valid for 24 writes, and `res` holds bytes `0..16`
    // in its low lane and bytes `16..24` in the low 8 bytes of its high lane.
    unsafe {
        _mm_storeu_si128(out_ptr.cast(), _mm256_castsi256_si128(res));
        _mm_storel_epi64(out_ptr.add(16).cast(), _mm256_extracti128_si256::<1>(res));
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_16 * 16` reads, `out_ptr` is valid for
/// `chunks_16 * 12` writes, and AVX2 is enabled. `out_ptr` must either not overlap `in_ptr` or
/// equal `in_ptr` (each iteration loads 32 or 16 input bytes before writing 24 or 12 output bytes,
/// so `out_ptr` never overtakes unread input).
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const URL_SAFE_ALPHABET: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_16: usize,
) -> (usize, bool) {
    let chunks_32 = chunks_16 / 2;
    let mut done_16 = 0;

    // SAFETY: Caller guarantees `chunks_16 * 16` reads at `in_ptr`, `chunks_16 * 12` writes at
    // `out_ptr`, and non-overtaking pointer advancement for in-place decoding.
    unsafe {
        let (hi_128, lo_128, roll_128, shuf_128, spec_ch_128, spec_adj_128) =
            decode_128_tables::<URL_SAFE_ALPHABET>();
        if chunks_32 > 0 {
            let hi_mask = _mm256_set_m128i(hi_128, hi_128);
            let lo_mask = _mm256_set_m128i(lo_128, lo_128);
            let roll_tbl = _mm256_set_m128i(roll_128, roll_128);
            let pack_shuf = _mm256_set_m128i(shuf_128, shuf_128);
            let lane_perm = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
            let special_ch = _mm256_set_m128i(spec_ch_128, spec_ch_128);
            let special_adj = _mm256_set_m128i(spec_adj_128, spec_adj_128);
            let mask_0f = _mm256_set1_epi8(0x0f);
            let mul_0140 = _mm256_set1_epi32(0x0140_0140);
            let mul_0001_1000 = _mm256_set1_epi32(0x0001_1000);
            let zero = _mm256_setzero_si256();

            for _ in 0..chunks_32 {
                let Some(res) = decode_chunk_32::<URL_SAFE_ALPHABET>(
                    in_ptr,
                    hi_mask,
                    lo_mask,
                    roll_tbl,
                    pack_shuf,
                    lane_perm,
                    special_ch,
                    special_adj,
                    mask_0f,
                    mul_0140,
                    mul_0001_1000,
                    zero,
                ) else {
                    return (done_16, true);
                };
                store_24_avx2(out_ptr, res);
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(24);
                done_16 += 2;
            }
        }

        let mut had_invalid = false;
        if done_16 < chunks_16 {
            let mask_0f = _mm_set1_epi8(0x0f);
            let mul_0140 = _mm_set1_epi32(0x0140_0140);
            let mul_0001_1000 = _mm_set1_epi32(0x0001_1000);
            let zero = _mm_setzero_si128();
            if let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(
                in_ptr,
                hi_128,
                lo_128,
                roll_128,
                shuf_128,
                spec_ch_128,
                spec_adj_128,
                mask_0f,
                mul_0140,
                mul_0001_1000,
                zero,
            ) {
                store_12_ssse3(out_ptr, res);
                done_16 += 1;
            } else {
                had_invalid = true;
            }
        }
        (done_16, had_invalid)
    }
}
