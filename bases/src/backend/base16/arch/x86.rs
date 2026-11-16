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
use core::mem::MaybeUninit;
#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
use std::arch::is_x86_feature_detected;

use x86_arch::__m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::__m256i;
use x86_arch::_mm_add_epi8;
use x86_arch::_mm_adds_epu8;
use x86_arch::_mm_and_si128;
use x86_arch::_mm_loadu_si128;
use x86_arch::_mm_maddubs_epi16;
use x86_arch::_mm_min_epu8;
use x86_arch::_mm_movemask_epi8;
use x86_arch::_mm_or_si128;
use x86_arch::_mm_packus_epi16;
use x86_arch::_mm_set1_epi8;
use x86_arch::_mm_set1_epi16;
use x86_arch::_mm_setr_epi8;
use x86_arch::_mm_shuffle_epi8;
use x86_arch::_mm_srli_epi16;
use x86_arch::_mm_storeu_si128;
use x86_arch::_mm_sub_epi8;
use x86_arch::_mm_subs_epu8;
use x86_arch::_mm_unpackhi_epi8;
use x86_arch::_mm_unpacklo_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_add_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_adds_epu8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_and_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_loadu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_maddubs_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_min_epu8;
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
use x86_arch::_mm256_shuffle_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_srli_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_storeu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_sub_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_subs_epu8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_unpackhi_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_unpacklo_epi8;

#[inline]
pub fn encode<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    if input.len() < 16 || output.len() < 32 {
        return (0, 0);
    }

    #[cfg(target_feature = "avx2")]
    {
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        unsafe { encode_avx2::<UPPERCASE>(input, output) }
    }

    #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
    {
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
    {
        // SAFETY: `target_feature = "ssse3"` is enabled at compile time.
        unsafe { encode_ssse3::<UPPERCASE>(input, output) }
    }
}

#[inline]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    let chunks_32 = (input.len() / 32).min(output.len() / 16);
    if chunks_32 == 0 {
        return (0, 0);
    }

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr().cast::<u8>();
    let done_32 = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time and `chunks_32` is
            // bounded by `input` and `output`.
            unsafe { decode_avx2::<UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_32) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support and `chunks_32` is
                // bounded by `input` and `output`.
                unsafe { decode_avx2::<UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_32) }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support and `chunks_32` is
                // bounded by `input` and `output`.
                unsafe { decode_ssse3::<UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_32) }
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
            // SAFETY: `target_feature = "ssse3"` is enabled at compile time and `chunks_32` is
            // bounded by `input` and `output`.
            unsafe { decode_ssse3::<UPPERCASE, STRICT_CASE>(in_ptr, out_ptr, chunks_32) }
        }
    };

    (done_32 * 32, done_32 * 16)
}

#[inline]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    let chunks_32 = buf.len() / 32;
    if chunks_32 == 0 {
        return (0, 0);
    }

    let ptr = buf.as_mut_ptr();
    let in_ptr = ptr.cast_const();
    let done_32 = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time, `chunks_32 * 32 <=
            // buf.len()`, and each step loads its full input before writing half as many bytes.
            unsafe { decode_avx2::<UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_32) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support, `chunks_32 * 32 <=
                // buf.len()`, and each step loads its full input before writing half as many bytes.
                unsafe { decode_avx2::<UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_32) }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support, `chunks_32 * 32 <=
                // buf.len()`, and each step loads its full input before writing half as many bytes.
                unsafe { decode_ssse3::<UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_32) }
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
            // SAFETY: `target_feature = "ssse3"` is enabled at compile time, `chunks_32 * 32 <=
            // buf.len()`, and each step loads its full input before writing half as many bytes.
            unsafe { decode_ssse3::<UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_32) }
        }
    };

    (done_32 * 32, done_32 * 16)
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
unsafe fn encode_ssse3<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    let chunks = (input.len() / 16).min(output.len() / 32);
    if chunks == 0 {
        return (0, 0);
    }

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = output.as_mut_ptr().cast::<u8>();
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
unsafe fn encode_avx2<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
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
            let mut out_ptr = output.as_mut_ptr().cast::<u8>();
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
                output.as_mut_ptr().cast::<u8>().add(output_index),
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
unsafe fn decode_nibbles_128<const UPPERCASE: bool, const STRICT_CASE: bool>(
    v: __m128i,
) -> __m128i {
    // Saturation maps digits to 0-9 and letters to 10-15; all other bytes stay above 15.
    let digits = _mm_sub_epi8(
        _mm_subs_epu8(
            _mm_add_epi8(v, _mm_set1_epi8(!b'9' as i8)),
            _mm_set1_epi8(6),
        ),
        _mm_set1_epi8(-16),
    );
    let letters = if STRICT_CASE {
        v
    } else {
        _mm_and_si128(v, _mm_set1_epi8(!0x20))
    };
    let base = if STRICT_CASE && !UPPERCASE {
        b'a'
    } else {
        b'A'
    };
    let letters = _mm_adds_epu8(
        _mm_sub_epi8(letters, _mm_set1_epi8(base as i8)),
        _mm_set1_epi8(10),
    );
    _mm_min_epu8(digits, letters)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 32 reads and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_chunk_32<const UPPERCASE: bool, const STRICT_CASE: bool>(
    in_ptr: *const u8,
) -> Option<__m128i> {
    // SAFETY: Caller guarantees `in_ptr` is valid for 32 readable bytes and SSSE3 is enabled.
    unsafe {
        let in0 = _mm_loadu_si128(in_ptr.cast());
        let in1 = _mm_loadu_si128(in_ptr.add(16).cast());

        let nib0 = decode_nibbles_128::<UPPERCASE, STRICT_CASE>(in0);
        let nib1 = decode_nibbles_128::<UPPERCASE, STRICT_CASE>(in1);
        // Biasing by 127 - 15 sets the sign bit exactly when either nibble is invalid.
        let err = _mm_adds_epu8(_mm_or_si128(nib0, nib1), _mm_set1_epi8(112));
        if _mm_movemask_epi8(err) != 0 {
            return None;
        }

        let mul_0110 = _mm_set1_epi16(0x0110);
        let pair0 = _mm_maddubs_epi16(nib0, mul_0110);
        let pair1 = _mm_maddubs_epi16(nib1, mul_0110);
        Some(_mm_packus_epi16(pair0, pair1))
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 32` reads, `out_ptr` is valid for
/// `chunks * 16` writes, and SSSE3 is enabled. `out_ptr` must either not overlap `in_ptr` or equal
/// `in_ptr` (each iteration loads 32 input bytes before writing 16 output bytes, so `out_ptr` never
/// overtakes unread input).
#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
) -> usize {
    let mut done = 0;
    // SAFETY: Caller guarantees `chunks * 32` reads at `in_ptr`, `chunks * 16` writes at `out_ptr`,
    // and non-overtaking pointer advancement for in-place decoding.
    unsafe {
        while done < chunks {
            let Some(packed) = decode_chunk_32::<UPPERCASE, STRICT_CASE>(in_ptr) else {
                break;
            };
            _mm_storeu_si128(out_ptr.cast(), packed);
            in_ptr = in_ptr.add(32);
            out_ptr = out_ptr.add(16);
            done += 1;
        }
    }
    done
}

/// # Safety
///
/// Requires AVX2 support on the current CPU.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_nibbles_256<const UPPERCASE: bool, const STRICT_CASE: bool>(
    v: __m256i,
) -> __m256i {
    let digits = _mm256_sub_epi8(
        _mm256_subs_epu8(
            _mm256_add_epi8(v, _mm256_set1_epi8(!b'9' as i8)),
            _mm256_set1_epi8(6),
        ),
        _mm256_set1_epi8(-16),
    );
    let letters = if STRICT_CASE {
        v
    } else {
        _mm256_and_si256(v, _mm256_set1_epi8(!0x20))
    };
    let base = if STRICT_CASE && !UPPERCASE {
        b'a'
    } else {
        b'A'
    };
    let letters = _mm256_adds_epu8(
        _mm256_sub_epi8(letters, _mm256_set1_epi8(base as i8)),
        _mm256_set1_epi8(10),
    );
    _mm256_min_epu8(digits, letters)
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for 64 reads and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_chunk_64<const UPPERCASE: bool, const STRICT_CASE: bool>(
    in_ptr: *const u8,
) -> Option<__m256i> {
    // SAFETY: Caller guarantees `in_ptr` is valid for 64 readable bytes and AVX2 is enabled.
    unsafe {
        let in0 = _mm256_loadu_si256(in_ptr.cast());
        let in1 = _mm256_loadu_si256(in_ptr.add(32).cast());

        let nib0 = decode_nibbles_256::<UPPERCASE, STRICT_CASE>(in0);
        let nib1 = decode_nibbles_256::<UPPERCASE, STRICT_CASE>(in1);
        let err = _mm256_adds_epu8(_mm256_or_si256(nib0, nib1), _mm256_set1_epi8(112));
        if _mm256_movemask_epi8(err) != 0 {
            return None;
        }

        let mul_0110 = _mm256_set1_epi16(0x0110);
        let pair0 = _mm256_maddubs_epi16(nib0, mul_0110);
        let pair1 = _mm256_maddubs_epi16(nib1, mul_0110);
        let packed = _mm256_packus_epi16(pair0, pair1);
        Some(_mm256_permute4x64_epi64::<0xD8>(packed))
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_32 * 32` reads, `out_ptr` is valid for
/// `chunks_32 * 16` writes, and AVX2 is enabled. `out_ptr` must either not overlap `in_ptr` or
/// equal `in_ptr` (each iteration loads 64 or 32 input bytes before writing 32 or 16 output bytes,
/// so `out_ptr` never overtakes unread input).
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_32: usize,
) -> usize {
    let chunks_64 = chunks_32 / 2;
    let mut done_32 = 0;

    // SAFETY: Caller guarantees `chunks_32 * 32` reads at `in_ptr`, `chunks_32 * 16` writes at
    // `out_ptr`, and non-overtaking pointer advancement for in-place decoding.
    unsafe {
        for _ in 0..chunks_64 {
            let Some(packed) = decode_chunk_64::<UPPERCASE, STRICT_CASE>(in_ptr) else {
                return done_32;
            };
            _mm256_storeu_si256(out_ptr.cast(), packed);
            in_ptr = in_ptr.add(64);
            out_ptr = out_ptr.add(32);
            done_32 += 2;
        }

        if done_32 < chunks_32 {
            if let Some(packed) = decode_chunk_32::<UPPERCASE, STRICT_CASE>(in_ptr) {
                _mm_storeu_si128(out_ptr.cast(), packed);
                done_32 += 1;
            }
        }
    }

    done_32
}
