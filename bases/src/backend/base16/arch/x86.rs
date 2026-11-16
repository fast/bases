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

//! x86 / x86_64 SSSE3 and AVX2 Base16 kernels.

#[cfg(target_arch = "x86")]
use core::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86_arch;
#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
use std::arch::is_x86_feature_detected;

use x86_arch::__m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::__m256i;
use x86_arch::_mm_add_epi8;
use x86_arch::_mm_and_si128;
use x86_arch::_mm_cmpeq_epi8;
use x86_arch::_mm_loadu_si128;
use x86_arch::_mm_maddubs_epi16;
use x86_arch::_mm_movemask_epi8;
use x86_arch::_mm_or_si128;
use x86_arch::_mm_packus_epi16;
use x86_arch::_mm_set1_epi8;
use x86_arch::_mm_set1_epi16;
use x86_arch::_mm_setr_epi8;
use x86_arch::_mm_setzero_si128;
use x86_arch::_mm_shuffle_epi8;
use x86_arch::_mm_srli_epi16;
use x86_arch::_mm_storeu_si128;
use x86_arch::_mm_unpackhi_epi8;
use x86_arch::_mm_unpacklo_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_add_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_and_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_cmpeq_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_loadu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_maddubs_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_movemask_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_or_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_packus_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_permute2x128_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_permute4x64_epi64;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set_m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_setzero_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_shuffle_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_srli_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_storeu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_unpackhi_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_unpacklo_epi8;

const HI_MASK_MIXED: [i8; 16] = [0, 0, 0, 1, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_UPPER: [i8; 16] = [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HI_MASK_LOWER: [i8; 16] = [0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const fn select_hi_mask<const UPPERCASE: bool, const STRICT_CASE: bool>() -> [i8; 16] {
    if !STRICT_CASE {
        HI_MASK_MIXED
    } else if UPPERCASE {
        HI_MASK_UPPER
    } else {
        HI_MASK_LOWER
    }
}

#[cfg(target_feature = "avx2")]
pub fn encode<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    // SAFETY: `target_feature = "avx2"` is enabled at compile time.
    unsafe { encode_avx2::<UPPERCASE>(input, output) }
}

#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
pub fn encode<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime feature detection confirmed AVX2 support.
        unsafe { encode_avx2::<UPPERCASE>(input, output) }
    } else if is_x86_feature_detected!("ssse3") {
        // SAFETY: Runtime feature detection confirmed SSSE3 support.
        unsafe { encode_ssse3::<UPPERCASE>(input, output) }
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
pub fn encode<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    // SAFETY: `target_feature = "ssse3"` is enabled at compile time.
    unsafe { encode_ssse3::<UPPERCASE>(input, output) }
}

#[cfg(target_feature = "avx2")]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    // SAFETY: `target_feature = "avx2"` is enabled at compile time.
    unsafe { decode_avx2::<UPPERCASE, STRICT_CASE>(input, output) }
}

#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime feature detection confirmed AVX2 support.
        unsafe { decode_avx2::<UPPERCASE, STRICT_CASE>(input, output) }
    } else if is_x86_feature_detected!("ssse3") {
        // SAFETY: Runtime feature detection confirmed SSSE3 support.
        unsafe { decode_ssse3::<UPPERCASE, STRICT_CASE>(input, output) }
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
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    // SAFETY: `target_feature = "ssse3"` is enabled at compile time.
    unsafe { decode_ssse3::<UPPERCASE, STRICT_CASE>(input, output) }
}

#[cfg(target_feature = "avx2")]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    // SAFETY: `target_feature = "avx2"` is enabled at compile time.
    unsafe { decode_in_place_avx2::<UPPERCASE, STRICT_CASE>(buf) }
}

#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime feature detection confirmed AVX2 support.
        unsafe { decode_in_place_avx2::<UPPERCASE, STRICT_CASE>(buf) }
    } else if is_x86_feature_detected!("ssse3") {
        // SAFETY: Runtime feature detection confirmed SSSE3 support.
        unsafe { decode_in_place_ssse3::<UPPERCASE, STRICT_CASE>(buf) }
    } else {
        let _ = buf;
        (0, 0)
    }
}

#[cfg(all(
    not(target_feature = "avx2"),
    not(feature = "std"),
    target_feature = "ssse3"
))]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    // SAFETY: `target_feature = "ssse3"` is enabled at compile time.
    unsafe { decode_in_place_ssse3::<UPPERCASE, STRICT_CASE>(buf) }
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn hex_table_128<const UPPERCASE: bool>() -> __m128i {
    if UPPERCASE {
        _mm_setr_epi8(
            b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8, b'4' as i8, b'5' as i8, b'6' as i8,
            b'7' as i8, b'8' as i8, b'9' as i8, b'A' as i8, b'B' as i8, b'C' as i8, b'D' as i8,
            b'E' as i8, b'F' as i8,
        )
    } else {
        _mm_setr_epi8(
            b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8, b'4' as i8, b'5' as i8, b'6' as i8,
            b'7' as i8, b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8,
            b'e' as i8, b'f' as i8,
        )
    }
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn encode_chunk_16(in_ptr: *const u8, out_ptr: *mut u8, hex_tbl: __m128i, mask_0f: __m128i) {
    // SAFETY: Caller guarantees `in_ptr` is valid for 16 reads and `out_ptr` for 32 writes.
    unsafe {
        let v = _mm_loadu_si128(in_ptr.cast());
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);
        let lo = _mm_and_si128(v, mask_0f);
        let hi_ascii = _mm_shuffle_epi8(hex_tbl, hi);
        let lo_ascii = _mm_shuffle_epi8(hex_tbl, lo);
        let out0 = _mm_unpacklo_epi8(hi_ascii, lo_ascii);
        let out1 = _mm_unpackhi_epi8(hi_ascii, lo_ascii);
        _mm_storeu_si128(out_ptr.cast(), out0);
        _mm_storeu_si128(out_ptr.add(16).cast(), out1);
    }
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[cfg(not(target_feature = "avx2"))]
#[target_feature(enable = "ssse3")]
unsafe fn encode_ssse3<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    let chunks = (input.len() / 16).min(output.len() / 32);
    if chunks == 0 {
        return (0, 0);
    }

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = output.as_mut_ptr();
    // SAFETY: `chunks * 16 <= input.len()` and `chunks * 32 <= output.len()`, and SSSE3 is enabled.
    unsafe {
        let hex_tbl = hex_table_128::<UPPERCASE>();
        let mask_0f = _mm_set1_epi8(0x0f);
        for _ in 0..chunks {
            encode_chunk_16(in_ptr, out_ptr, hex_tbl, mask_0f);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(32);
        }
    }
    (chunks * 16, chunks * 32)
}

/// # Safety
///
/// Requires AVX2 support on the current CPU.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2<const UPPERCASE: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    let chunks = (input.len() / 32).min(output.len() / 64);
    let mut input_index = chunks * 32;
    let mut output_index = chunks * 64;

    // SAFETY: `chunks * 32 <= input.len()`, `chunks * 64 <= output.len()`, and if the trailing
    // 16-byte condition holds, 16 more input bytes and 32 more output bytes are in bounds.
    unsafe {
        let hex_128 = hex_table_128::<UPPERCASE>();
        if chunks > 0 {
            let hex_tbl = _mm256_set_m128i(hex_128, hex_128);
            let mask_0f = _mm256_set1_epi8(0x0f);
            let mut in_ptr = input.as_ptr();
            let mut out_ptr = output.as_mut_ptr();
            for _ in 0..chunks {
                let v = _mm256_loadu_si256(in_ptr.cast());
                let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);
                let lo = _mm256_and_si256(v, mask_0f);
                let hi_ascii = _mm256_shuffle_epi8(hex_tbl, hi);
                let lo_ascii = _mm256_shuffle_epi8(hex_tbl, lo);
                let unp_lo = _mm256_unpacklo_epi8(hi_ascii, lo_ascii);
                let unp_hi = _mm256_unpackhi_epi8(hi_ascii, lo_ascii);
                let out0 = _mm256_permute2x128_si256::<0x20>(unp_lo, unp_hi);
                let out1 = _mm256_permute2x128_si256::<0x31>(unp_lo, unp_hi);
                _mm256_storeu_si256(out_ptr.cast(), out0);
                _mm256_storeu_si256(out_ptr.add(32).cast(), out1);
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(64);
            }
        }
        if input_index + 16 <= input.len() && output_index + 32 <= output.len() {
            let mask_0f = _mm_set1_epi8(0x0f);
            encode_chunk_16(
                input.as_ptr().add(input_index),
                output.as_mut_ptr().add(output_index),
                hex_128,
                mask_0f,
            );
            input_index += 16;
            output_index += 32;
        }
    }
    (input_index, output_index)
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_128_tables<const UPPERCASE: bool, const STRICT_CASE: bool>()
-> (__m128i, __m128i, __m128i) {
    let m = select_hi_mask::<UPPERCASE, STRICT_CASE>();
    let hi_mask = _mm_setr_epi8(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
        m[14], m[15],
    );
    let lo_mask = _mm_setr_epi8(1, 7, 7, 7, 7, 7, 7, 1, 1, 1, 0, 0, 0, 0, 0, 0);
    let hi_off = _mm_setr_epi8(0, 0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    (hi_mask, lo_mask, hi_off)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 32 reads and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_chunk_32(
    in_ptr: *const u8,
    hi_mask: __m128i,
    lo_mask: __m128i,
    hi_off: __m128i,
    mask_0f: __m128i,
    mul_0110: __m128i,
    zero: __m128i,
) -> Option<__m128i> {
    // SAFETY: Caller guarantees `in_ptr` is valid for 32 readable bytes and SSSE3 is enabled.
    unsafe {
        let in0 = _mm_loadu_si128(in_ptr.cast());
        let in1 = _mm_loadu_si128(in_ptr.add(16).cast());

        let hi0 = _mm_and_si128(_mm_srli_epi16(in0, 4), mask_0f);
        let lo0 = _mm_and_si128(in0, mask_0f);
        let chk0 = _mm_and_si128(
            _mm_shuffle_epi8(hi_mask, hi0),
            _mm_shuffle_epi8(lo_mask, lo0),
        );

        let hi1 = _mm_and_si128(_mm_srli_epi16(in1, 4), mask_0f);
        let lo1 = _mm_and_si128(in1, mask_0f);
        let chk1 = _mm_and_si128(
            _mm_shuffle_epi8(hi_mask, hi1),
            _mm_shuffle_epi8(lo_mask, lo1),
        );

        let err = _mm_or_si128(_mm_cmpeq_epi8(chk0, zero), _mm_cmpeq_epi8(chk1, zero));
        if _mm_movemask_epi8(err) != 0 {
            return None;
        }

        let nib0 = _mm_add_epi8(lo0, _mm_shuffle_epi8(hi_off, hi0));
        let nib1 = _mm_add_epi8(lo1, _mm_shuffle_epi8(hi_off, hi1));
        let pair0 = _mm_maddubs_epi16(nib0, mul_0110);
        let pair1 = _mm_maddubs_epi16(nib1, mul_0110);
        Some(_mm_packus_epi16(pair0, pair1))
    }
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[cfg(not(target_feature = "avx2"))]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks = (input.len() / 32).min(output.len() / 16);
    if chunks == 0 {
        return (0, 0);
    }

    let mut done = 0;
    // SAFETY: `chunks * 32 <= input.len()` and `chunks * 16 <= output.len()`, and SSSE3 is enabled.
    unsafe {
        let (hi_mask, lo_mask, hi_off) = decode_128_tables::<UPPERCASE, STRICT_CASE>();
        let mask_0f = _mm_set1_epi8(0x0f);
        let mul_0110 = _mm_set1_epi16(0x0110);
        let zero = _mm_setzero_si128();
        let mut in_ptr = input.as_ptr();
        let mut out_ptr = output.as_mut_ptr();

        while done < chunks {
            let Some(packed) =
                decode_chunk_32(in_ptr, hi_mask, lo_mask, hi_off, mask_0f, mul_0110, zero)
            else {
                break;
            };
            _mm_storeu_si128(out_ptr.cast(), packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
            done += 1;
        }
    }
    (done * 32, done * 16)
}

/// # Safety
///
/// Requires SSSE3 support on the current CPU.
#[cfg(not(target_feature = "avx2"))]
#[target_feature(enable = "ssse3")]
unsafe fn decode_in_place_ssse3<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    let chunks = buf.len() / 32;
    if chunks == 0 {
        return (0, 0);
    }

    let mut done = 0;
    let ptr = buf.as_mut_ptr();
    // SAFETY: `chunks * 32 <= buf.len()`. Each iteration finishes loading `in_ptr[0..32]` before
    // storing `out_ptr[0..16]`, and `out_ptr` lags behind `in_ptr` by `16 * done` bytes.
    unsafe {
        let (hi_mask, lo_mask, hi_off) = decode_128_tables::<UPPERCASE, STRICT_CASE>();
        let mask_0f = _mm_set1_epi8(0x0f);
        let mul_0110 = _mm_set1_epi16(0x0110);
        let zero = _mm_setzero_si128();
        let mut in_ptr: *const u8 = ptr;
        let mut out_ptr: *mut u8 = ptr;

        while done < chunks {
            let Some(packed) =
                decode_chunk_32(in_ptr, hi_mask, lo_mask, hi_off, mask_0f, mul_0110, zero)
            else {
                break;
            };
            _mm_storeu_si128(out_ptr.cast(), packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
            done += 1;
        }
    }
    (done * 32, done * 16)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 64 reads and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_chunk_64(
    in_ptr: *const u8,
    hi_mask: __m256i,
    lo_mask: __m256i,
    hi_off: __m256i,
    mask_0f: __m256i,
    mul_0110: __m256i,
    zero: __m256i,
) -> Option<__m256i> {
    // SAFETY: Caller guarantees `in_ptr` is valid for 64 readable bytes and AVX2 is enabled.
    unsafe {
        let in0 = _mm256_loadu_si256(in_ptr.cast());
        let in1 = _mm256_loadu_si256(in_ptr.add(32).cast());

        let hi0 = _mm256_and_si256(_mm256_srli_epi16(in0, 4), mask_0f);
        let lo0 = _mm256_and_si256(in0, mask_0f);
        let chk0 = _mm256_and_si256(
            _mm256_shuffle_epi8(hi_mask, hi0),
            _mm256_shuffle_epi8(lo_mask, lo0),
        );

        let hi1 = _mm256_and_si256(_mm256_srli_epi16(in1, 4), mask_0f);
        let lo1 = _mm256_and_si256(in1, mask_0f);
        let chk1 = _mm256_and_si256(
            _mm256_shuffle_epi8(hi_mask, hi1),
            _mm256_shuffle_epi8(lo_mask, lo1),
        );

        let err = _mm256_or_si256(_mm256_cmpeq_epi8(chk0, zero), _mm256_cmpeq_epi8(chk1, zero));
        if _mm256_movemask_epi8(err) != 0 {
            return None;
        }

        let nib0 = _mm256_add_epi8(lo0, _mm256_shuffle_epi8(hi_off, hi0));
        let nib1 = _mm256_add_epi8(lo1, _mm256_shuffle_epi8(hi_off, hi1));
        let pair0 = _mm256_maddubs_epi16(nib0, mul_0110);
        let pair1 = _mm256_maddubs_epi16(nib1, mul_0110);
        let packed = _mm256_packus_epi16(pair0, pair1);
        Some(_mm256_permute4x64_epi64::<0xD8>(packed))
    }
}

/// # Safety
///
/// Requires AVX2 support on the current CPU.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize) {
    let chunks = (input.len() / 64).min(output.len() / 32);
    let mut done = 0;

    // SAFETY: `chunks * 64 <= input.len()` and `chunks * 32 <= output.len()`, and the trailing
    // 32-byte step checks bounds before accessing `input` and `output`.
    unsafe {
        let (hi_128, lo_128, off_128) = decode_128_tables::<UPPERCASE, STRICT_CASE>();
        if chunks > 0 {
            let hi_mask = _mm256_set_m128i(hi_128, hi_128);
            let lo_mask = _mm256_set_m128i(lo_128, lo_128);
            let hi_off = _mm256_set_m128i(off_128, off_128);
            let mask_0f = _mm256_set1_epi8(0x0f);
            let mul_0110 = _mm256_set1_epi16(0x0110);
            let zero = _mm256_setzero_si256();
            let mut in_ptr = input.as_ptr();
            let mut out_ptr = output.as_mut_ptr();

            while done < chunks {
                let Some(packed) =
                    decode_chunk_64(in_ptr, hi_mask, lo_mask, hi_off, mask_0f, mul_0110, zero)
                else {
                    return (done * 64, done * 32);
                };
                _mm256_storeu_si256(out_ptr.cast(), packed);
                in_ptr = in_ptr.add(64);
                out_ptr = out_ptr.add(32);
                done += 1;
            }
        }

        let mut input_index = done * 64;
        let mut output_index = done * 32;
        if input_index + 32 <= input.len() && output_index + 16 <= output.len() {
            let mask_0f = _mm_set1_epi8(0x0f);
            let mul_0110 = _mm_set1_epi16(0x0110);
            let zero = _mm_setzero_si128();
            if let Some(packed) = decode_chunk_32(
                input.as_ptr().add(input_index),
                hi_128,
                lo_128,
                off_128,
                mask_0f,
                mul_0110,
                zero,
            ) {
                _mm_storeu_si128(output.as_mut_ptr().add(output_index).cast(), packed);
                input_index += 32;
                output_index += 16;
            }
        }
        (input_index, output_index)
    }
}

/// # Safety
///
/// Requires AVX2 support on the current CPU.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_in_place_avx2<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    let chunks = buf.len() / 64;
    let mut done = 0;
    let ptr = buf.as_mut_ptr();
    let len = buf.len();

    // SAFETY: `chunks * 64 <= len`, and the trailing 32-byte check keeps reads/writes in bounds.
    // Every chunk finishes loading its input vector before writing to `out_ptr`.
    unsafe {
        let (hi_128, lo_128, off_128) = decode_128_tables::<UPPERCASE, STRICT_CASE>();
        if chunks > 0 {
            let hi_mask = _mm256_set_m128i(hi_128, hi_128);
            let lo_mask = _mm256_set_m128i(lo_128, lo_128);
            let hi_off = _mm256_set_m128i(off_128, off_128);
            let mask_0f = _mm256_set1_epi8(0x0f);
            let mul_0110 = _mm256_set1_epi16(0x0110);
            let zero = _mm256_setzero_si256();
            let mut in_ptr: *const u8 = ptr;
            let mut out_ptr: *mut u8 = ptr;

            while done < chunks {
                let Some(packed) =
                    decode_chunk_64(in_ptr, hi_mask, lo_mask, hi_off, mask_0f, mul_0110, zero)
                else {
                    return (done * 64, done * 32);
                };
                _mm256_storeu_si256(out_ptr.cast(), packed);
                in_ptr = in_ptr.add(64);
                out_ptr = out_ptr.add(32);
                done += 1;
            }
        }

        let mut input_index = done * 64;
        let mut output_index = done * 32;
        if input_index + 32 <= len {
            let mask_0f = _mm_set1_epi8(0x0f);
            let mul_0110 = _mm_set1_epi16(0x0110);
            let zero = _mm_setzero_si128();
            if let Some(packed) = decode_chunk_32(
                ptr.add(input_index),
                hi_128,
                lo_128,
                off_128,
                mask_0f,
                mul_0110,
                zero,
            ) {
                _mm_storeu_si128(ptr.add(output_index).cast(), packed);
                input_index += 32;
                output_index += 16;
            }
        }
        (input_index, output_index)
    }
}
