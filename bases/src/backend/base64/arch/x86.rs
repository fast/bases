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

use crate::backend::base64::HI_MASK;
use crate::backend::base64::LO_MASK_STD;
use crate::backend::base64::LO_MASK_URL;
use crate::backend::simd::X86Token;
use crate::backend::simd::simd_fn;

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

    match X86Token::detect() {
        #[cfg(any(feature = "std", target_feature = "avx2"))]
        Some(X86Token::Avx2(token)) => encode_avx2::<URL_SAFE_ALPHABET>(token, input, output),
        #[cfg(not(target_feature = "avx2"))]
        Some(X86Token::Ssse3(token)) => encode_ssse3::<URL_SAFE_ALPHABET>(token, input, output),
        None => (0, 0),
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
    let out_len = output.len();
    let out_ptr = output.as_mut_ptr().cast::<u8>();
    let (done_16, had_invalid) = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time and `chunks_16` and
            // `out_len` are bounded by `input` and `output`.
            unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16, out_len) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support and `chunks_16` and
                // `out_len` are bounded by `input` and `output`.
                unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16, out_len) }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support and `chunks_16` and
                // `out_len` are bounded by `input` and `output`.
                unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16, out_len) }
            } else {
                let _ = (in_ptr, out_ptr, out_len);
                (0, false)
            }
        }

        #[cfg(all(
            not(target_feature = "avx2"),
            not(feature = "std"),
            target_feature = "ssse3"
        ))]
        {
            // SAFETY: `target_feature = "ssse3"` is enabled at compile time and `chunks_16` and
            // `out_len` are bounded by `input` and `output`.
            unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, out_ptr, chunks_16, out_len) }
        }
    };

    (done_16 * 16, done_16 * 12, had_invalid)
}

#[inline]
pub fn decode_in_place<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    complete_len: usize,
    required: usize,
) -> (usize, usize, bool) {
    let chunks_16 = complete_len.min(buf.len()) / 16;
    if chunks_16 == 0 {
        return (0, 0, false);
    }

    let out_len = required.min(buf.len());
    let ptr = buf.as_mut_ptr();
    let in_ptr = ptr.cast_const();
    let (done_16, had_invalid) = {
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: `target_feature = "avx2"` is enabled at compile time, `chunks_16 * 16 <=
            // buf.len()`, bounded wide stores stay within `out_len`, and each step loads its full
            // input before writing fewer output bytes.
            unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16, out_len) }
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: Runtime feature detection confirmed AVX2 support, `chunks_16 * 16 <=
                // buf.len()`, bounded wide stores stay within `out_len`, and each step loads its
                // full input before writing fewer output bytes.
                unsafe { decode_avx2::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16, out_len) }
            } else if is_x86_feature_detected!("ssse3") {
                // SAFETY: Runtime feature detection confirmed SSSE3 support, `chunks_16 * 16 <=
                // buf.len()`, bounded wide stores stay within `out_len`, and each step loads its
                // full input before writing fewer output bytes.
                unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16, out_len) }
            } else {
                let _ = (in_ptr, ptr, out_len);
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
            // buf.len()`, bounded wide stores stay within `out_len`, and each step loads its full
            // input before writing fewer output bytes.
            unsafe { decode_ssse3::<URL_SAFE_ALPHABET>(in_ptr, ptr, chunks_16, out_len) }
        }
    };

    (done_16 * 16, done_16 * 12, had_invalid)
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
fn encode_128_consts<const URL_SAFE_ALPHABET: bool>() -> (
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
    let c25 = _mm_set1_epi8(25);
    let offset_lut = if URL_SAFE_ALPHABET {
        _mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -17, 32, 0, 0,
        )
    } else {
        _mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,
        )
    };
    (shuf, mask0, mul0, mask1, mul1, c51, c25, offset_lut)
}

#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
fn encode_chunk_12(
    raw: __m128i,
    shuf: __m128i,
    mask0: __m128i,
    mul0: __m128i,
    mask1: __m128i,
    mul1: __m128i,
    c51: __m128i,
    c25: __m128i,
    offset_lut: __m128i,
) -> __m128i {
    let w = _mm_shuffle_epi8(raw, shuf);
    let t0 = _mm_mulhi_epu16(_mm_and_si128(w, mask0), mul0);
    let t1 = _mm_mullo_epi16(_mm_and_si128(w, mask1), mul1);
    let indices = _mm_or_si128(t0, t1);
    let result = _mm_subs_epu8(indices, c51);
    // Map A-Z to slot 0, a-z to slot 1, and the remaining symbols to slots 2-13.
    let key = _mm_sub_epi8(result, _mm_cmpgt_epi8(indices, c25));
    _mm_add_epi8(indices, _mm_shuffle_epi8(offset_lut, key))
}

simd_fn! {
    #[cfg(not(target_feature = "avx2"))]
    fn encode_ssse3<const URL_SAFE_ALPHABET: bool>(
        _token: Ssse3Token,
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
        let (shuf, mask0, mul0, mask1, mul1, c51, c25, offset_lut) =
            encode_128_consts::<URL_SAFE_ALPHABET>();
        for _ in 0..chunks {
            // SAFETY: `chunks > 0` implies `input.len() >= (chunks - 1) * 12 + 16`.
            let raw = unsafe { _mm_loadu_si128(in_ptr.cast()) };
            let ascii = encode_chunk_12(raw, shuf, mask0, mul0, mask1, mul1, c51, c25, offset_lut);
            // SAFETY: `output.len() >= chunks * 16`, so `out_ptr` is valid for 16 writes and
            // advancing both pointers stays in bounds.
            unsafe {
                _mm_storeu_si128(out_ptr.cast(), ascii);
                in_ptr = in_ptr.add(12);
                out_ptr = out_ptr.add(16);
            }
        }
        (chunks * 12, chunks * 16)
    }
}

simd_fn! {
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    fn encode_avx2<const URL_SAFE_ALPHABET: bool>(
        _token: Avx2Token,
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

        let (sh_128, m0_128, ml0_128, m1_128, ml1_128, c51_128, c25_128, lut_128) =
            encode_128_consts::<URL_SAFE_ALPHABET>();

        if chunks > 0 {
            let shuf = _mm256_set_m128i(sh_128, sh_128);
            let mask0 = _mm256_set_m128i(m0_128, m0_128);
            let mul0 = _mm256_set_m128i(ml0_128, ml0_128);
            let mask1 = _mm256_set_m128i(m1_128, m1_128);
            let mul1 = _mm256_set_m128i(ml1_128, ml1_128);
            let c51 = _mm256_set_m128i(c51_128, c51_128);
            let c25 = _mm256_set_m128i(c25_128, c25_128);
            let offset_lut = _mm256_set_m128i(lut_128, lut_128);

            let mut in_ptr = input.as_ptr();
            let mut out_ptr = output.as_mut_ptr().cast::<u8>();
            for _ in 0..chunks {
                // SAFETY: `chunks > 0` guarantees `input.len() >= (chunks - 1) * 24 + 28`, so both
                // 16-byte loads at `in_ptr` and `in_ptr.add(12)` are in bounds.
                let (hi_raw, lo_raw) = unsafe {
                    (
                        _mm_loadu_si128(in_ptr.add(12).cast()),
                        _mm_loadu_si128(in_ptr.cast()),
                    )
                };
                let raw = _mm256_set_m128i(hi_raw, lo_raw);
                let w = _mm256_shuffle_epi8(raw, shuf);
                let t0 = _mm256_mulhi_epu16(_mm256_and_si256(w, mask0), mul0);
                let t1 = _mm256_mullo_epi16(_mm256_and_si256(w, mask1), mul1);
                let indices = _mm256_or_si256(t0, t1);
                let result = _mm256_subs_epu8(indices, c51);
                let key = _mm256_sub_epi8(result, _mm256_cmpgt_epi8(indices, c25));
                let ascii = _mm256_add_epi8(indices, _mm256_shuffle_epi8(offset_lut, key));
                // SAFETY: `output.len() >= chunks * 32`, so `out_ptr` is valid for 32 writes and
                // advancing both pointers stays in bounds.
                unsafe {
                    _mm256_storeu_si256(out_ptr.cast(), ascii);
                    in_ptr = in_ptr.add(24);
                    out_ptr = out_ptr.add(32);
                }
            }
        }

        if input_index + 16 <= input.len() && output_index + 16 <= output.len() {
            // SAFETY: The guard above ensures `input_index + 16 <= input.len()`.
            let raw = unsafe { _mm_loadu_si128(input.as_ptr().add(input_index).cast()) };
            let ascii = encode_chunk_12(
                raw, sh_128, m0_128, ml0_128, m1_128, ml1_128, c51_128, c25_128, lut_128,
            );
            // SAFETY: The guard above ensures `output_index + 16 <= output.len()`.
            unsafe {
                _mm_storeu_si128(
                    output.as_mut_ptr().cast::<u8>().add(output_index).cast(),
                    ascii,
                )
            };
            input_index += 12;
            output_index += 16;
        }

        (input_index, output_index)
    }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn decode_128_tables<const URL_SAFE_ALPHABET: bool>()
-> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let hi_mask = load_16(&HI_MASK);
    let lo_mask = load_16(if URL_SAFE_ALPHABET {
        &LO_MASK_URL
    } else {
        &LO_MASK_STD
    });
    let roll_tbl = load_16_i8(if URL_SAFE_ALPHABET {
        &ROLL_URL
    } else {
        &ROLL_STD
    });
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

#[inline]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_16<const URL_SAFE_ALPHABET: bool>(
    v: __m128i,
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

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 12 writes and SSSE3 is enabled.
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn store_12_ssse3(out_ptr: *mut u8, res: __m128i) {
    let hi32 = _mm_cvtsi128_si32(_mm_srli_si128::<8>(res));
    // SAFETY: Caller guarantees `out_ptr` is valid for 12 writes.
    unsafe {
        _mm_storel_epi64(out_ptr.cast(), res);
        ptr::write_unaligned(out_ptr.add(8).cast::<i32>(), hi32);
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks * 16` reads, `out_ptr` is valid for
/// `out_len >= chunks * 12` writes, and SSSE3 is enabled. `out_ptr` must either not overlap
/// `in_ptr` or equal `in_ptr` (each iteration loads 16 input bytes before writing, and bounded
/// 16-byte stores satisfy `12k + 16 <= 16(k + 1)` so `out_ptr` never overtakes unread input).
#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3<const URL_SAFE_ALPHABET: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks: usize,
    out_len: usize,
) -> (usize, bool) {
    let wide_chunks = (out_len.saturating_sub(4) / 12).min(chunks);
    let mut done = 0;
    let (hi_mask, lo_mask, roll_tbl, pack_shuf, special_ch, special_adj) =
        decode_128_tables::<URL_SAFE_ALPHABET>();
    let mask_0f = _mm_set1_epi8(0x0f);
    let mul_0140 = _mm_set1_epi32(0x0140_0140);
    let mul_0001_1000 = _mm_set1_epi32(0x0001_1000);
    let zero = _mm_setzero_si128();

    while done < wide_chunks {
        // SAFETY: `done < wide_chunks <= chunks` guarantees 16 readable bytes at `in_ptr`.
        let v = unsafe { _mm_loadu_si128(in_ptr.cast()) };
        let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(
            v,
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
            return (done, true);
        };
        // SAFETY: `done < wide_chunks <= chunks` guarantees `done * 12 + 16 <= out_len`, in-place
        // writes lag behind `in_ptr`, and advancing `in_ptr`/`out_ptr` by 16 and 12 bytes stays
        // within both buffers.
        unsafe {
            _mm_storeu_si128(out_ptr.cast(), res);
            in_ptr = in_ptr.add(16);
            out_ptr = out_ptr.add(12);
        }
        done += 1;
    }

    if done < chunks {
        // SAFETY: `done < chunks` guarantees 16 readable bytes at `in_ptr`.
        let v = unsafe { _mm_loadu_si128(in_ptr.cast()) };
        let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(
            v,
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
            return (done, true);
        };
        // SAFETY: `done < chunks` guarantees 12 writable bytes at `out_ptr`, and this chunk's input
        // was already loaded so in-place writes do not overwrite unread input.
        unsafe { store_12_ssse3(out_ptr, res) };
        done += 1;
    }
    (done, false)
}

/// Decodes 32 Base64 bytes into un-permuted lanes holding output bytes `0..12` in lane 0 and
/// `12..24` in lane 1.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn decode_chunk_32<const URL_SAFE_ALPHABET: bool>(
    v: __m256i,
    hi_mask: __m256i,
    lo_mask: __m256i,
    roll_tbl: __m256i,
    pack_shuf: __m256i,
    special_ch: __m256i,
    special_adj: __m256i,
    mask_0f: __m256i,
    mul_0140: __m256i,
    mul_0001_1000: __m256i,
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

    let base = _mm256_add_epi8(v, _mm256_shuffle_epi8(roll_tbl, hi));
    let adj = _mm256_and_si256(_mm256_cmpeq_epi8(v, special_ch), special_adj);
    let vals = if URL_SAFE_ALPHABET {
        _mm256_add_epi8(base, adj)
    } else {
        _mm256_sub_epi8(base, adj)
    };
    let merged = _mm256_maddubs_epi16(vals, mul_0140);
    let packed = _mm256_madd_epi16(merged, mul_0001_1000);
    Some(_mm256_shuffle_epi8(packed, pack_shuf))
}

/// # Safety
///
/// Caller must ensure `out_ptr` is valid for 24 writes and AVX2 is enabled.
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_24_avx2(out_ptr: *mut u8, res: __m256i) {
    let lo = _mm256_castsi256_si128(res);
    let hi = _mm256_extracti128_si256::<1>(res);
    // SAFETY: Caller guarantees `out_ptr` is valid for 24 writes.
    unsafe {
        _mm_storeu_si128(out_ptr.cast(), lo);
        _mm_storel_epi64(out_ptr.add(16).cast(), hi);
    }
}

/// # Safety
///
/// Caller must ensure `in_ptr` is valid for `chunks_16 * 16` reads, `out_ptr` is valid for
/// `out_len >= chunks_16 * 12` writes, and AVX2 is enabled. `out_ptr` must either not overlap
/// `in_ptr` or equal `in_ptr` (each iteration loads 32 or 16 input bytes before writing, and
/// bounded wide stores satisfy `24k + 28 <= 32(k + 1)` and `12k + 16 <= 16(k + 1)` so `out_ptr`
/// never overtakes unread input).
#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2<const URL_SAFE_ALPHABET: bool>(
    mut in_ptr: *const u8,
    mut out_ptr: *mut u8,
    chunks_16: usize,
    out_len: usize,
) -> (usize, bool) {
    let chunks_32 = chunks_16 / 2;
    let wide_32 = (out_len.saturating_sub(4) / 24).min(chunks_32);
    let mut done_16 = 0;

    let (hi_128, lo_128, roll_128, shuf_128, spec_ch_128, spec_adj_128) =
        decode_128_tables::<URL_SAFE_ALPHABET>();
    if chunks_32 > 0 {
        let hi_mask = _mm256_set_m128i(hi_128, hi_128);
        let lo_mask = _mm256_set_m128i(lo_128, lo_128);
        let roll_tbl = _mm256_set_m128i(roll_128, roll_128);
        let pack_shuf = _mm256_set_m128i(shuf_128, shuf_128);
        let special_ch = _mm256_set_m128i(spec_ch_128, spec_ch_128);
        let special_adj = _mm256_set_m128i(spec_adj_128, spec_adj_128);
        let mask_0f = _mm256_set1_epi8(0x0f);
        let mul_0140 = _mm256_set1_epi32(0x0140_0140);
        let mul_0001_1000 = _mm256_set1_epi32(0x0001_1000);
        let zero = _mm256_setzero_si256();

        for _ in 0..wide_32 {
            // SAFETY: `done_16 + 2 <= chunks_16` guarantees 32 readable bytes at `in_ptr`.
            let v = unsafe { _mm256_loadu_si256(in_ptr.cast()) };
            let Some(shuf) = decode_chunk_32::<URL_SAFE_ALPHABET>(
                v,
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
                return (done_16, true);
            };
            let lo = _mm256_castsi256_si128(shuf);
            let hi = _mm256_extracti128_si256::<1>(shuf);
            // SAFETY: `done_16 / 2 < wide_32 <= chunks_32` guarantees `done_16 + 2 <= chunks_16`
            // and `done_16 * 12 + 28 <= out_len`, in-place writes lag behind `in_ptr`, and
            // advancing `in_ptr`/`out_ptr` by 32 and 24 bytes stays within both buffers.
            unsafe {
                _mm_storeu_si128(out_ptr.cast(), lo);
                _mm_storeu_si128(out_ptr.add(12).cast(), hi);
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(24);
            }
            done_16 += 2;
        }

        if wide_32 < chunks_32 {
            // SAFETY: `done_16 + 2 <= chunks_16` guarantees 32 readable bytes at `in_ptr`.
            let v = unsafe { _mm256_loadu_si256(in_ptr.cast()) };
            let Some(shuf) = decode_chunk_32::<URL_SAFE_ALPHABET>(
                v,
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
                return (done_16, true);
            };
            let lane_perm = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
            let res = _mm256_permutevar8x32_epi32(shuf, lane_perm);
            // SAFETY: `done_16 + 2 <= chunks_16` guarantees 24 writable bytes at `out_ptr`,
            // in-place writes lag behind `in_ptr`, and advancing `in_ptr`/`out_ptr` by 32 and 24
            // bytes stays within both buffers.
            unsafe {
                store_24_avx2(out_ptr, res);
                in_ptr = in_ptr.add(32);
                out_ptr = out_ptr.add(24);
            }
            done_16 += 2;
        }
    }

    let mut had_invalid = false;
    if done_16 < chunks_16 {
        let mask_0f = _mm_set1_epi8(0x0f);
        let mul_0140 = _mm_set1_epi32(0x0140_0140);
        let mul_0001_1000 = _mm_set1_epi32(0x0001_1000);
        let zero = _mm_setzero_si128();
        // SAFETY: `done_16 < chunks_16` guarantees 16 readable bytes at `in_ptr`.
        let v = unsafe { _mm_loadu_si128(in_ptr.cast()) };
        if let Some(res) = decode_chunk_16::<URL_SAFE_ALPHABET>(
            v,
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
            // SAFETY: `done_16 < chunks_16` guarantees 12 writable bytes at `out_ptr` (and 16 when
            // `out_len - done_16 * 12 >= 16`), this chunk's input was already loaded, and AVX2
            // implies SSSE3.
            unsafe {
                if out_len.saturating_sub(done_16 * 12) >= 16 {
                    _mm_storeu_si128(out_ptr.cast(), res);
                } else {
                    store_12_ssse3(out_ptr, res);
                }
            }
            done_16 += 1;
        } else {
            had_invalid = true;
        }
    }
    (done_16, had_invalid)
}
