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

use crate::backend::base32::LO_MASK_HEX;
use crate::backend::base32::LO_MASK_STD;
use crate::backend::base32::select_hi_mask;
use crate::backend::simd::X86Token;
use crate::backend::simd::simd_fn;

const ROLL_STD: [i8; 16] = [0, 0, 0, -24, -65, -65, -97, -97, 0, 0, 0, 0, 0, 0, 0, 0];
const ROLL_HEX: [i8; 16] = [0, 0, 0, -48, -55, -55, -87, -87, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline]
pub fn encode<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> Option<(usize, usize)> {
    if input.len() < 16 || output.len() < 16 {
        return Some((0, 0));
    }

    match X86Token::detect() {
        #[cfg(any(feature = "std", target_feature = "avx2"))]
        Some(X86Token::Avx2(token)) => {
            Some(encode_avx2::<EXTENDED_HEX, UPPERCASE>(token, input, output))
        }
        #[cfg(not(target_feature = "avx2"))]
        Some(X86Token::Ssse3(token)) => Some(encode_ssse3::<EXTENDED_HEX, UPPERCASE>(
            token, input, output,
        )),
        None => None,
    }
}

#[inline]
pub fn decode<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> Option<(usize, usize)> {
    let chunks_16 = (input.len() / 16).min(output.len() / 10);
    if chunks_16 == 0 {
        return Some((0, 0));
    }

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr().cast::<u8>();
    let out_len = output.len();
    let done_16 = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time and `chunks_16` is
            // bounded by `input` and `output`.
            unsafe {
                decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                    in_ptr, out_ptr, chunks_16, out_len,
                )
            }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support and `chunks_16` is
                // bounded by `input` and `output`.
                unsafe {
                    decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                        in_ptr, out_ptr, chunks_16, out_len,
                    )
                }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support and `chunks_16` is
                // bounded by `input` and `output`.
                unsafe {
                    decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                        in_ptr, out_ptr, chunks_16, out_len,
                    )
                }
            } else {
                let _ = (in_ptr, out_ptr, out_len);
                return None;
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
                decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                    in_ptr, out_ptr, chunks_16, out_len,
                )
            }
        }
    };

    Some((done_16 * 16, done_16 * 10))
}

#[inline]
pub fn decode_in_place<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
    complete_len: usize,
    required: usize,
) -> (usize, usize) {
    let chunks_16 = complete_len.min(buf.len()) / 16;
    if chunks_16 == 0 {
        return (0, 0);
    }

    let out_len = required.min(buf.len());
    let ptr = buf.as_mut_ptr();
    let in_ptr = ptr.cast_const();
    let done_16 = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time, `chunks_16 * 16 <=
            // buf.len()`, and each step loads its full input before writing fewer output bytes.
            unsafe {
                decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(in_ptr, ptr, chunks_16, out_len)
            }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support, `chunks_16 * 16 <=
                // buf.len()`, and each step loads its full input before writing fewer output bytes.
                unsafe {
                    decode_avx2::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                        in_ptr, ptr, chunks_16, out_len,
                    )
                }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support, `chunks_16 * 16 <=
                // buf.len()`, and each step loads its full input before writing fewer output bytes.
                unsafe {
                    decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                        in_ptr, ptr, chunks_16, out_len,
                    )
                }
            } else {
                let _ = (in_ptr, ptr, out_len);
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
            unsafe {
                decode_ssse3::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>(
                    in_ptr, ptr, chunks_16, out_len,
                )
            }
        }
    };

    (done_16 * 16, done_16 * 10)
}

#[inline]
#[target_feature(enable = "ssse3")]
fn load_16(bytes: &[u8; 16]) -> __m128i {
    // SAFETY: `bytes` contains 16 initialized readable bytes.
    unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn load_16_i8(bytes: &[i8; 16]) -> __m128i {
    // SAFETY: `bytes` contains 16 initialized readable bytes.
    unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn encode_128_consts<const EXTENDED_HEX: bool, const UPPERCASE: bool>() -> (
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
    )
}

#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
fn encode_chunk_16(
    raw: __m128i,
    w_shuf: __m128i,
    even_mask: __m128i,
    even_mul: __m128i,
    odd_mask: __m128i,
    odd_mul: __m128i,
    thresh: __m128i,
    base: __m128i,
    diff: __m128i,
) -> __m128i {
    let w = _mm_shuffle_epi8(raw, w_shuf);
    let even = _mm_mulhi_epu16(_mm_and_si128(w, even_mask), even_mul);
    let odd = _mm_mullo_epi16(_mm_and_si128(w, odd_mask), odd_mul);
    let indices = _mm_or_si128(even, odd);
    let is_alt = _mm_cmpgt_epi8(indices, thresh);
    let offset = _mm_add_epi8(base, _mm_and_si128(is_alt, diff));
    _mm_add_epi8(indices, offset)
}

simd_fn! {
    #[cfg(not(target_feature = "avx2"))]
    fn encode_ssse3<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
        _token: Ssse3Token,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> (usize, usize) {
        if input.len() < 16 {
            return (0, 0);
        }
        let chunks = ((input.len() - 6) / 10).min(output.len() / 16);
        if chunks == 0 {
            return (0, 0);
        }

        let mut in_ptr = input.as_ptr();
        let mut out_ptr = output.as_mut_ptr().cast::<u8>();
        let (w_shuf, even_mask, even_mul, odd_mask, odd_mul, thresh, base, diff) =
            encode_128_consts::<EXTENDED_HEX, UPPERCASE>();
        for _ in 0..chunks {
            // SAFETY: `chunks > 0` guarantees `input.len() >= (chunks - 1) * 10 + 16`.
            let raw = unsafe { _mm_loadu_si128(in_ptr.cast()) };
            let ascii = encode_chunk_16(
                raw, w_shuf, even_mask, even_mul, odd_mask, odd_mul, thresh, base, diff,
            );
            // SAFETY: `output.len() >= chunks * 16`, so `out_ptr` is valid for 16 writes and
            // advancing both pointers stays in bounds.
            unsafe {
                _mm_storeu_si128(out_ptr.cast(), ascii);
                in_ptr = in_ptr.add(10);
                out_ptr = out_ptr.add(16);
            }
        }
        (chunks * 10, chunks * 16)
    }
}

simd_fn! {
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    fn encode_avx2<const EXTENDED_HEX: bool, const UPPERCASE: bool>(
        _token: Avx2Token,
        input: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> (usize, usize) {
        let chunks = if input.len() >= 26 {
            ((input.len() - 6) / 20).min(output.len() / 32)
        } else {
            0
        };
        let mut input_index = chunks * 20;
        let mut output_index = chunks * 32;

        let (w_shuf_128, em_128, eml_128, om_128, oml_128, th_128, bs_128, df_128) =
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
            let mut out_ptr = output.as_mut_ptr().cast::<u8>();
            for _ in 0..chunks {
                // SAFETY: `chunks > 0` implies `input.len() >= (chunks - 1) * 20 + 26`, so both
                // 16-byte loads at `in_ptr` and `in_ptr.add(10)` are in bounds.
                let (hi_raw, lo_raw) = unsafe {
                    (
                        _mm_loadu_si128(in_ptr.add(10).cast()),
                        _mm_loadu_si128(in_ptr.cast()),
                    )
                };
                let raw = _mm256_set_m128i(hi_raw, lo_raw);
                let w = _mm256_shuffle_epi8(raw, w_shuf);
                let even = _mm256_mulhi_epu16(_mm256_and_si256(w, even_mask), even_mul);
                let odd = _mm256_mullo_epi16(_mm256_and_si256(w, odd_mask), odd_mul);
                let indices = _mm256_or_si256(even, odd);
                let is_alt = _mm256_cmpgt_epi8(indices, thresh);
                let offset = _mm256_add_epi8(base, _mm256_and_si256(is_alt, diff));
                let ascii = _mm256_add_epi8(indices, offset);
                // SAFETY: `output.len() >= chunks * 32`, so `out_ptr` is valid for 32 writes and
                // advancing both pointers stays in bounds.
                unsafe {
                    _mm256_storeu_si256(out_ptr.cast(), ascii);
                    in_ptr = in_ptr.add(20);
                    out_ptr = out_ptr.add(32);
                }
            }
        }

        if input_index + 16 <= input.len() && output_index + 16 <= output.len() {
            // SAFETY: The guard above ensures `input_index + 16 <= input.len()`.
            let raw = unsafe { _mm_loadu_si128(input.as_ptr().add(input_index).cast()) };
            let ascii = encode_chunk_16(
                raw, w_shuf_128, em_128, eml_128, om_128, oml_128, th_128, bs_128, df_128,
            );
            // SAFETY: The guard above ensures `output_index + 16 <= output.len()`.
            unsafe {
                _mm_storeu_si128(
                    output.as_mut_ptr().cast::<u8>().add(output_index).cast(),
                    ascii,
                )
            };
            input_index += 10;
            output_index += 16;
        }

        (input_index, output_index)
    }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn decode_128_tables<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>()
-> (__m128i, __m128i, __m128i, __m128i) {
    let hi_mask = load_16(select_hi_mask::<UPPERCASE, STRICT_CASE>());
    let lo_mask = load_16(if EXTENDED_HEX {
        &LO_MASK_HEX
    } else {
        &LO_MASK_STD
    });
    let roll_tbl = load_16_i8(if EXTENDED_HEX { &ROLL_HEX } else { &ROLL_STD });
    let pack_shuf = _mm_setr_epi8(4, 3, 2, 1, 0, 12, 11, 10, 9, 8, -1, -1, -1, -1, -1, -1);
    (hi_mask, lo_mask, roll_tbl, pack_shuf)
}

#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_16(
    v: __m128i,
    hi_mask: __m128i,
    lo_mask: __m128i,
    roll_tbl: __m128i,
    pack_shuf: __m128i,
    mask_0f: __m128i,
    mul_0120: __m128i,
    mul_0001_0400: __m128i,
    zero: __m128i,
) -> Option<__m128i> {
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

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 10 writes and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn store_10_ssse3(out_ptr: *mut u8, res: __m128i) {
    let hi16 = _mm_extract_epi16::<4>(res) as u16;
    // SAFETY: Caller guarantees `out_ptr` is valid for 10 writes.
    unsafe {
        _mm_storel_epi64(out_ptr.cast(), res);
        ptr::write_unaligned(out_ptr.add(8).cast::<u16>(), hi16);
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads, `out_ptr` is valid for
/// `out_len >= chunks * 10` writes, and SSSE3 is enabled. `out_ptr` must either not overlap
/// `in_ptr` or equal `in_ptr` (each iteration loads 16 input bytes before writing, wide 16-byte
/// stores require `10 * k + 16 <= out_len`, and `out_ptr` never overtakes unread input).
#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
    out_len: usize,
) -> usize {
    let wide_chunks = (out_len.saturating_sub(6) / 10).min(chunks);
    let mut done = 0;
    let (hi_mask, lo_mask, roll_tbl, pack_shuf) =
        decode_128_tables::<EXTENDED_HEX, UPPERCASE, STRICT_CASE>();
    let mask_0f = _mm_set1_epi8(0x0f);
    let mul_0120 = _mm_set1_epi16(0x0120);
    let mul_0001_0400 = _mm_set1_epi32(0x0001_0400);
    let zero = _mm_setzero_si128();

    while done < wide_chunks {
        // SAFETY: `done < wide_chunks <= chunks` guarantees 16 readable bytes at `in_ptr`.
        let v = unsafe { _mm_loadu_si128(in_ptr.cast()) };
        let Some(res) = decode_chunk_16(
            v,
            hi_mask,
            lo_mask,
            roll_tbl,
            pack_shuf,
            mask_0f,
            mul_0120,
            mul_0001_0400,
            zero,
        ) else {
            return done;
        };
        // SAFETY: `done < wide_chunks <= chunks` guarantees `done * 10 + 16 <= out_len`, in-place
        // writes lag behind `in_ptr`, and advancing `in_ptr`/`out_ptr` by 16 and 10 bytes stays
        // within both buffers.
        unsafe {
            _mm_storeu_si128(out_ptr.cast(), res);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(10);
        }
        done += 1;
    }

    while done < chunks {
        // SAFETY: `done < chunks` guarantees 16 readable bytes at `in_ptr`.
        let v = unsafe { _mm_loadu_si128(in_ptr.cast()) };
        let Some(res) = decode_chunk_16(
            v,
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
        // SAFETY: `done < chunks` guarantees 10 writable bytes at `out_ptr`, in-place writes lag
        // behind `in_ptr`, and advancing `in_ptr`/`out_ptr` by 16 and 10 bytes stays within both
        // buffers.
        unsafe {
            store_10_ssse3(out_ptr, res);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(10);
        }
        done += 1;
    }
    done
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_32(
    v: __m256i,
    hi_mask: __m256i,
    lo_mask: __m256i,
    roll_tbl: __m256i,
    pack_shuf: __m256i,
    mask_0f: __m256i,
    mul_0120: __m256i,
    mul_0001_0400: __m256i,
    zero: __m256i,
) -> Option<__m256i> {
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

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 20 writes and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_20_avx2(out_ptr: *mut u8, res: __m256i) {
    let res_lo = _mm256_castsi256_si128(res);
    let res_hi = _mm256_extracti128_si256::<1>(res);
    let lo16 = _mm_or_si128(res_lo, _mm_slli_si128::<10>(res_hi));
    let hi4 = _mm_cvtsi128_si32(_mm_srli_si128::<6>(res_hi));
    // SAFETY: Caller guarantees `out_ptr` is valid for 20 writes.
    unsafe {
        _mm_storeu_si128(out_ptr.cast(), lo16);
        ptr::write_unaligned(out_ptr.add(16).cast::<i32>(), hi4);
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_16 * 16` reads, `out_ptr` is valid for
/// `out_len >= chunks_16 * 10` writes, and AVX2 is enabled. `out_ptr` must either not overlap
/// `in_ptr` or equal `in_ptr` (each step loads 32 or 16 input bytes before writing, wide stores
/// stay within `out_len`, and `out_ptr` never overtakes unread input).
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const EXTENDED_HEX: bool, const UPPERCASE: bool, const STRICT_CASE: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_16: usize,
    out_len: usize,
) -> usize {
    let chunks_32 = chunks_16 / 2;
    let wide_32 = (out_len.saturating_sub(6) / 20).min(chunks_32);
    let mut done_16 = 0;

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

        for idx_32 in 0..chunks_32 {
            // SAFETY: `idx_32 < chunks_32` guarantees 32 readable bytes at `in_ptr`.
            let v = unsafe { _mm256_loadu_si256(in_ptr.cast()) };
            let Some(res) = decode_chunk_32(
                v,
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
            let lo = _mm256_castsi256_si128(res);
            let hi = _mm256_extracti128_si256::<1>(res);
            // SAFETY: `idx_32 < chunks_32` guarantees 20 writable bytes at `out_ptr` (and 26 when
            // `idx_32 < wide_32`), this chunk's input was already loaded, and advancing by 32 and
            // 20 bytes stays within both buffers.
            unsafe {
                if idx_32 < wide_32 {
                    _mm_storeu_si128(out_ptr.cast(), lo);
                    _mm_storeu_si128(out_ptr.add(10).cast(), hi);
                } else {
                    store_20_avx2(out_ptr, res);
                }
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(20);
            }
            done_16 += 2;
        }
    }

    if done_16 < chunks_16 {
        let mask_0f = _mm_set1_epi8(0x0f);
        let mul_0120 = _mm_set1_epi16(0x0120);
        let mul_0001_0400 = _mm_set1_epi32(0x0001_0400);
        let zero = _mm_setzero_si128();
        // SAFETY: `done_16 < chunks_16` guarantees 16 readable bytes at `in_ptr`.
        let v = unsafe { _mm_loadu_si128(in_ptr.cast()) };
        if let Some(res) = decode_chunk_16(
            v,
            hi_128,
            lo_128,
            roll_128,
            shuf_128,
            mask_0f,
            mul_0120,
            mul_0001_0400,
            zero,
        ) {
            // SAFETY: `done_16 < chunks_16` guarantees 10 writable bytes at `out_ptr` (and 16 when
            // `done_16 * 10 + 16 <= out_len`), this chunk's input was already loaded, and AVX2
            // implies SSSE3.
            unsafe {
                if done_16 * 10 + 16 <= out_len {
                    _mm_storeu_si128(out_ptr.cast(), res);
                } else {
                    store_10_ssse3(out_ptr, res);
                }
            }
            done_16 += 1;
        }
    }

    done_16
}
