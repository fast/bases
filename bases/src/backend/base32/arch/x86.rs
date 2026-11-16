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

//! x86 / x86_64 SSSE3 and AVX2 Base32 kernels.

#[cfg(target_arch = "x86")]
use core::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86_arch;
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
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm_cvtsi128_si32;
use x86_arch::_mm_extract_epi16;
use x86_arch::_mm_loadu_si128;
use x86_arch::_mm_madd_epi16;
use x86_arch::_mm_maddubs_epi16;
use x86_arch::_mm_movemask_epi8;
use x86_arch::_mm_mulhi_epu16;
use x86_arch::_mm_mullo_epi16;
use x86_arch::_mm_or_si128;
use x86_arch::_mm_set1_epi8;
use x86_arch::_mm_set1_epi16;
use x86_arch::_mm_set1_epi32;
use x86_arch::_mm_setr_epi8;
use x86_arch::_mm_setr_epi16;
use x86_arch::_mm_setzero_si128;
use x86_arch::_mm_shuffle_epi8;
use x86_arch::_mm_slli_epi64;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm_slli_si128;
use x86_arch::_mm_srli_epi16;
use x86_arch::_mm_srli_epi64;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm_srli_si128;
use x86_arch::_mm_storel_epi64;
use x86_arch::_mm_storeu_si128;
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
use x86_arch::_mm256_set_m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi32;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_setzero_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_shuffle_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_slli_epi64;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_srli_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_srli_epi64;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_storeu_si256;

const HI_MASK_MIXED: [i8; 16] = [0, 0, 0, 1, 2, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_UPPER: [i8; 16] = [0, 0, 0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_LOWER: [i8; 16] = [0, 0, 0, 1, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0];

const LO_MASK_STD: [i8; 16] = [4, 6, 7, 7, 7, 7, 7, 7, 6, 6, 6, 2, 2, 2, 2, 2];
const LO_MASK_HEX: [i8; 16] = [5, 7, 7, 7, 7, 7, 7, 3, 3, 3, 2, 2, 2, 2, 2, 2];

const ROLL_STD: [i8; 16] = [0, 0, 0, -24, -65, -65, -97, -97, 0, 0, 0, 0, 0, 0, 0, 0];
const ROLL_HEX: [i8; 16] = [0, 0, 0, -48, -55, -55, -87, -87, 0, 0, 0, 0, 0, 0, 0, 0];

const fn select_hi_mask<const UPPERCASE: bool, const STRICT_CASE: bool>() -> [i8; 16] {
    if !STRICT_CASE {
        HI_MASK_MIXED
    } else if UPPERCASE {
        HI_MASK_UPPER
    } else {
        HI_MASK_LOWER
    }
}

#[inline]
pub fn encode<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    #[cfg(target_feature = "avx2")]
    {
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        unsafe { encode_avx2::<EXTENDED_HEX, UPPERCASE>(input, output) }
    }

    #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime feature detection confirmed AVX2 support.
            unsafe { encode_avx2::<EXTENDED_HEX, UPPERCASE>(input, output) }
        } else if is_x86_feature_detected!("ssse3") {
            // SAFETY: Runtime feature detection confirmed SSSE3 support.
            unsafe { encode_ssse3::<EXTENDED_HEX, UPPERCASE>(input, output) }
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
        unsafe { encode_ssse3::<EXTENDED_HEX, UPPERCASE>(input, output) }
    }
}

#[inline]
pub fn decode<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks_16 = (input.len() / 16).min(output.len() / 10);
    if chunks_16 == 0 {
        return (0, 0);
    }

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let done_16 = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time and `chunks_16` is
            // bounded by `input` and `output`.
            unsafe {
                decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_16)
            }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support and `chunks_16` is
                // bounded by `input` and `output`.
                unsafe {
                    decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_16)
                }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support and `chunks_16` is
                // bounded by `input` and `output`.
                unsafe {
                    decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_16)
                }
            } else {
                let _ = (in_ptr, out_ptr);
                0
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
            unsafe {
                decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_16)
            }
        }
    };

    (done_16 * 16, done_16 * 10)
}

#[inline]
pub fn decode_in_place<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
    complete_len: usize,
) -> (usize, usize) {
    let chunks_16 = complete_len.min(buf.len()) / 16;
    if chunks_16 == 0 {
        return (0, 0);
    }

    let ptr = buf.as_mut_ptr();
    let in_ptr = ptr.cast_const();
    let done_16 = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time, `chunks_16 * 16 <=
            // buf.len()`, and each step loads its full input before writing fewer output bytes.
            unsafe { decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_16) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support, `chunks_16 * 16 <=
                // buf.len()`, and each step loads its full input before writing fewer output bytes.
                unsafe {
                    decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_16)
                }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support, `chunks_16 * 16 <=
                // buf.len()`, and each step loads its full input before writing fewer output bytes.
                unsafe {
                    decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_16)
                }
            } else {
                let _ = (in_ptr, ptr);
                0
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
            unsafe { decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_16) }
        }
    };

    (done_16 * 16, done_16 * 10)
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn encode_128_consts<const EXTENDED_HEX: bool, const UPPERCASE: bool>() -> (
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
    let w_shuf = _mm_setr_epi8(1, 0, 2, 1, 3, 2, 4, 3, 6, 5, 7, 6, 8, 7, 9, 8);
    let even_mask = _mm_setr_epi16(
        0xF800_u16 as i16,
        0x3E00_u16 as i16,
        0x0F80_u16 as i16,
        0x03E0_u16 as i16,
        0xF800_u16 as i16,
        0x3E00_u16 as i16,
        0x0F80_u16 as i16,
        0x03E0_u16 as i16,
    );
    let even_mul = _mm_setr_epi16(
        0x0020, 0x0080, 0x0200, 0x0800, 0x0020, 0x0080, 0x0200, 0x0800,
    );
    let odd_mask = _mm_setr_epi16(
        0x07C0, 0x01F0, 0x007C, 0x001F, 0x07C0, 0x01F0, 0x007C, 0x001F,
    );
    let odd_mul = _mm_setr_epi16(
        0x0004, 0x0010, 0x0040, 0x0100, 0x0004, 0x0010, 0x0040, 0x0100,
    );

    let (thresh, base, diff) = if EXTENDED_HEX {
        (9_i8, b'0' as i8, if UPPERCASE { 7_i8 } else { 39_i8 })
    } else {
        let base = if UPPERCASE { b'A' as i8 } else { b'a' as i8 };
        let diff = 24_i8.wrapping_sub(base);
        (25_i8, base, diff)
    };

    (
        w_shuf,
        even_mask,
        even_mul,
        odd_mask,
        odd_mul,
        _mm_set1_epi8(thresh),
        _mm_set1_epi8(base),
        _mm_set1_epi8(diff),
        _mm_set1_epi8(0),
    )
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 16 reads and `out_ptr` for 16 writes, with SSSE3
/// enabled.
#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
unsafe fn encode_chunk_16(
    in_ptr: *const u8,
    out_ptr: *mut u8,
    w_shuf: __m128i,
    even_mask: __m128i,
    even_mul: __m128i,
    odd_mask: __m128i,
    odd_mul: __m128i,
    thresh: __m128i,
    base: __m128i,
    diff: __m128i,
) {
    // SAFETY: Caller guarantees `in_ptr` is valid for 16 readable bytes and `out_ptr` for 16
    // writes.
    unsafe {
        let raw = _mm_loadu_si128(in_ptr.cast());
        let w = _mm_shuffle_epi8(raw, w_shuf);
        let even = _mm_mulhi_epu16(_mm_and_si128(w, even_mask), even_mul);
        let odd = _mm_mullo_epi16(_mm_and_si128(w, odd_mask), odd_mul);
        let indices = _mm_or_si128(even, odd);
        let is_alt = _mm_cmpgt_epi8(indices, thresh);
        let offset = _mm_add_epi8(base, _mm_and_si128(is_alt, diff));
        let ascii = _mm_add_epi8(indices, offset);
        _mm_storeu_si128(out_ptr.cast(), ascii);
    }
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[cfg(not(target_feature = "avx2"))]
#[target_feature(enable = "ssse3")]
unsafe fn encode_ssse3<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
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

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = output.as_mut_ptr();
    // SAFETY: `chunks > 0` guarantees `input.len() >= (chunks - 1) * 10 + 16` and `output.len() >=
    // chunks * 16`.
    unsafe {
        let (w_shuf, even_mask, even_mul, odd_mask, odd_mul, thresh, base, diff, _) =
            encode_128_consts::<EXTENDED_HEX, UPPERCASE>();
        for _ in 0..chunks {
            encode_chunk_16(
                in_ptr, out_ptr, w_shuf, even_mask, even_mul, odd_mask, odd_mul, thresh, base, diff,
            );
            in_ptr = in_ptr.add(10);
            out_ptr = out_ptr.add(16);
        }
    }
    (chunks * 10, chunks * 16)
}

/// # Safety
///
/// Requires AVX2 support on the current CPU.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks = if input.len() >= 26 {
        ((input.len() - 6) / 20).min(output.len() / 32)
    } else {
        0
    };
    let mut input_index = chunks * 20;
    let mut output_index = chunks * 32;

    // SAFETY: `chunks > 0` implies `input.len() >= (chunks - 1) * 20 + 26` and `output.len() >=
    // chunks * 32`. The trailing 16-byte check ensures `input_index + 16 <= input.len()` and
    // `output_index + 16 <= output.len()`.
    unsafe {
        let (w_shuf_128, em_128, eml_128, om_128, oml_128, th_128, bs_128, df_128, _) =
            encode_128_consts::<EXTENDED_HEX, UPPERCASE>();

        if chunks > 0 {
            let w_shuf = _mm256_set_m128i(w_shuf_128, w_shuf_128);
            let even_mask = _mm256_set_m128i(em_128, em_128);
            let even_mul = _mm256_set_m128i(eml_128, eml_128);
            let odd_mask = _mm256_set_m128i(om_128, om_128);
            let odd_mul = _mm256_set_m128i(oml_128, oml_128);
            let thresh = _mm256_set_m128i(th_128, th_128);
            let base = _mm256_set_m128i(bs_128, bs_128);
            let diff = _mm256_set_m128i(df_128, df_128);

            let mut in_ptr = input.as_ptr();
            let mut out_ptr = output.as_mut_ptr();
            for _ in 0..chunks {
                let raw = _mm256_set_m128i(
                    _mm_loadu_si128(in_ptr.add(10).cast()),
                    _mm_loadu_si128(in_ptr.cast()),
                );
                let w = _mm256_shuffle_epi8(raw, w_shuf);
                let even = _mm256_mulhi_epu16(_mm256_and_si256(w, even_mask), even_mul);
                let odd = _mm256_mullo_epi16(_mm256_and_si256(w, odd_mask), odd_mul);
                let indices = _mm256_or_si256(even, odd);
                let is_alt = _mm256_cmpgt_epi8(indices, thresh);
                let offset = _mm256_add_epi8(base, _mm256_and_si256(is_alt, diff));
                let ascii = _mm256_add_epi8(indices, offset);
                _mm256_storeu_si256(out_ptr.cast(), ascii);
                in_ptr = in_ptr.add(20);
                out_ptr = out_ptr.add(32);
            }
        }

        if input_index + 16 <= input.len() && output_index + 16 <= output.len() {
            encode_chunk_16(
                input.as_ptr().add(input_index),
                output.as_mut_ptr().add(output_index),
                w_shuf_128,
                em_128,
                eml_128,
                om_128,
                oml_128,
                th_128,
                bs_128,
                df_128,
            );
            input_index += 10;
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
unsafe fn decode_128_tables<
    const EXTENDED_HEX: bool,
    const UPPERCASE: bool,
    const STRICT_CASE: bool,
>() -> (__m128i, __m128i, __m128i, __m128i) {
    let hi = select_hi_mask::<UPPERCASE, STRICT_CASE>();
    let lo = if EXTENDED_HEX {
        LO_MASK_HEX
    } else {
        LO_MASK_STD
    };
    let roll = if EXTENDED_HEX { ROLL_HEX } else { ROLL_STD };
    let hi_mask = _mm_setr_epi8(
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7], hi[8], hi[9], hi[10], hi[11],
        hi[12], hi[13], hi[14], hi[15],
    );
    let lo_mask = _mm_setr_epi8(
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], lo[8], lo[9], lo[10], lo[11],
        lo[12], lo[13], lo[14], lo[15],
    );
    let roll_tbl = _mm_setr_epi8(
        roll[0], roll[1], roll[2], roll[3], roll[4], roll[5], roll[6], roll[7], roll[8], roll[9],
        roll[10], roll[11], roll[12], roll[13], roll[14], roll[15],
    );
    let pack_shuf = _mm_setr_epi8(4, 3, 2, 1, 0, 12, 11, 10, 9, 8, -1, -1, -1, -1, -1, -1);
    (hi_mask, lo_mask, roll_tbl, pack_shuf)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 16 reads and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_16(
    in_ptr: *const u8,
    hi_mask: __m128i,
    lo_mask: __m128i,
    roll_tbl: __m128i,
    pack_shuf: __m128i,
    mask_0f: __m128i,
    mul_0120: __m128i,
    mul_0001_0400: __m128i,
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

        let vals = _mm_add_epi8(v, _mm_shuffle_epi8(roll_tbl, hi));
        let merged10 = _mm_maddubs_epi16(vals, mul_0120);
        let packed20 = _mm_madd_epi16(merged10, mul_0001_0400);
        let q = _mm_or_si128(_mm_slli_epi64(packed20, 20), _mm_srli_epi64(packed20, 32));
        Some(_mm_shuffle_epi8(q, pack_shuf))
    }
}

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 10 writes and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn store_10_ssse3(out_ptr: *mut u8, res: __m128i) {
    // SAFETY: Caller guarantees `out_ptr` is valid for 10 writes and `res` holds the 10 decoded
    // bytes in its low bytes.
    unsafe {
        _mm_storel_epi64(out_ptr.cast(), res);
        ptr::write_unaligned(
            out_ptr.add(8).cast::<u16>(),
            _mm_extract_epi16::<4>(res) as u16,
        );
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads, `out_ptr` is valid for
/// `chunks * 10` writes, and SSSE3 is enabled. `out_ptr` must either not overlap `in_ptr` or equal
/// `in_ptr` (each iteration loads 16 input bytes before writing 10 output bytes, so `out_ptr` never
/// overtakes unread input).
#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
) -> usize {
    let mut done = 0;
    // SAFETY: Caller guarantees `chunks * 16` reads at `in_ptr`, `chunks * 10` writes at `out_ptr`,
    // and non-overtaking pointer advancement for in-place decoding.
    unsafe {
        let (hi_mask, lo_mask, roll_tbl, pack_shuf) =
            decode_128_tables::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();
        let mask_0f = _mm_set1_epi8(0x0f);
        let mul_0120 = _mm_set1_epi16(0x0120);
        let mul_0001_0400 = _mm_set1_epi32(0x0001_0400);
        let zero = _mm_setzero_si128();

        while done < chunks {
            let Some(res) = decode_chunk_16(
                in_ptr,
                hi_mask,
                lo_mask,
                roll_tbl,
                pack_shuf,
                mask_0f,
                mul_0120,
                mul_0001_0400,
                zero,
            ) else {
                break;
            };
            store_10_ssse3(out_ptr, res);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(10);
            done += 1;
        }
    }
    done
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 32 reads and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_chunk_32(
    in_ptr: *const u8,
    hi_mask: __m256i,
    lo_mask: __m256i,
    roll_tbl: __m256i,
    pack_shuf: __m256i,
    mask_0f: __m256i,
    mul_0120: __m256i,
    mul_0001_0400: __m256i,
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

        let vals = _mm256_add_epi8(v, _mm256_shuffle_epi8(roll_tbl, hi));
        let merged10 = _mm256_maddubs_epi16(vals, mul_0120);
        let packed20 = _mm256_madd_epi16(merged10, mul_0001_0400);
        let q = _mm256_or_si256(
            _mm256_slli_epi64(packed20, 20),
            _mm256_srli_epi64(packed20, 32),
        );
        Some(_mm256_shuffle_epi8(q, pack_shuf))
    }
}

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 20 writes and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_20_avx2(out_ptr: *mut u8, res: __m256i) {
    // SAFETY: `res` contains 10 decoded bytes in `res_lo[0..10]` (with bytes `10..16` zeroed) and
    // 10 decoded bytes in `res_hi[0..10]`, and caller guarantees `out_ptr` is valid for 20 writes.
    unsafe {
        let res_lo = _mm256_castsi256_si128(res);
        let res_hi = _mm256_extracti128_si256::<1>(res);
        let lo16 = _mm_or_si128(res_lo, _mm_slli_si128::<10>(res_hi));
        _mm_storeu_si128(out_ptr.cast(), lo16);
        let hi4 = _mm_srli_si128::<6>(res_hi);
        ptr::write_unaligned(out_ptr.add(16).cast::<i32>(), _mm_cvtsi128_si32(hi4));
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_16 * 16` reads, `out_ptr` is valid for
/// `chunks_16 * 10` writes, and AVX2 is enabled. `out_ptr` must either not overlap `in_ptr` or
/// equal `in_ptr` (each iteration loads 32 or 16 input bytes before writing 20 or 10 output bytes,
/// so `out_ptr` never overtakes unread input).
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_16: usize,
) -> usize {
    let chunks_32 = chunks_16 / 2;
    let mut done_16 = 0;

    // SAFETY: Caller guarantees `chunks_16 * 16` reads at `in_ptr`, `chunks_16 * 10` writes at
    // `out_ptr`, and non-overtaking pointer advancement for in-place decoding.
    unsafe {
        let (hi_128, lo_128, roll_128, shuf_128) =
            decode_128_tables::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();
        if chunks_32 > 0 {
            let hi_mask = _mm256_set_m128i(hi_128, hi_128);
            let lo_mask = _mm256_set_m128i(lo_128, lo_128);
            let roll_tbl = _mm256_set_m128i(roll_128, roll_128);
            let pack_shuf = _mm256_set_m128i(shuf_128, shuf_128);
            let mask_0f = _mm256_set1_epi8(0x0f);
            let mul_0120 = _mm256_set1_epi16(0x0120);
            let mul_0001_0400 = _mm256_set1_epi32(0x0001_0400);
            let zero = _mm256_setzero_si256();

            for _ in 0..chunks_32 {
                let Some(res) = decode_chunk_32(
                    in_ptr,
                    hi_mask,
                    lo_mask,
                    roll_tbl,
                    pack_shuf,
                    mask_0f,
                    mul_0120,
                    mul_0001_0400,
                    zero,
                ) else {
                    return done_16;
                };
                store_20_avx2(out_ptr, res);
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(20);
                done_16 += 2;
            }
        }

        if done_16 < chunks_16 {
            let mask_0f = _mm_set1_epi8(0x0f);
            let mul_0120 = _mm_set1_epi16(0x0120);
            let mul_0001_0400 = _mm_set1_epi32(0x0001_0400);
            let zero = _mm_setzero_si128();
            if let Some(res) = decode_chunk_16(
                in_ptr,
                hi_128,
                lo_128,
                roll_128,
                shuf_128,
                mask_0f,
                mul_0120,
                mul_0001_0400,
                zero,
            ) {
                store_10_ssse3(out_ptr, res);
                done_16 += 1;
            }
        }
    }

    done_16
}
