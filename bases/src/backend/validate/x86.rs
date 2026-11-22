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

//! x86 / x86_64 SSSE3 and AVX2 nibble-lut validation kernels.

#[cfg(target_arch = "x86")]
use core::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86_arch;
#[cfg(all(not(target_feature = "avx2"), feature = "std"))]
use std::arch::is_x86_feature_detected;

use x86_arch::__m128i;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::__m256i;
use x86_arch::_mm_and_si128;
use x86_arch::_mm_cmpeq_epi8;
use x86_arch::_mm_loadu_si128;
use x86_arch::_mm_min_epu8;
use x86_arch::_mm_movemask_epi8;
use x86_arch::_mm_set1_epi8;
use x86_arch::_mm_setzero_si128;
use x86_arch::_mm_shuffle_epi8;
use x86_arch::_mm_srli_epi16;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_and_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_broadcastsi128_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_cmpeq_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_loadu_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_min_epu8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_movemask_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_set1_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_setzero_si256;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_shuffle_epi8;
#[cfg(any(feature = "std", target_feature = "avx2"))]
use x86_arch::_mm256_srli_epi16;

#[cfg(not(target_feature = "avx2"))]
const SSSE3_BATCH_LEN: usize = 64;
#[cfg(any(feature = "std", target_feature = "avx2"))]
const AVX2_BATCH_LEN: usize = 256;

#[inline(always)]
pub fn validate(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    if input.len() < 16 {
        return Ok(0);
    }

    #[cfg(target_feature = "avx2")]
    {
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        unsafe { validate_avx2(input, hi_mask, lo_mask) }
    }

    #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime detection confirmed AVX2 support.
            unsafe { validate_avx2(input, hi_mask, lo_mask) }
        } else if is_x86_feature_detected!("ssse3") {
            // SAFETY: Runtime detection confirmed SSSE3 support.
            unsafe { validate_ssse3(input, hi_mask, lo_mask) }
        } else {
            Ok(0)
        }
    }

    #[cfg(all(
        not(target_feature = "avx2"),
        not(feature = "std"),
        target_feature = "ssse3"
    ))]
    {
        // SAFETY: `target_feature = "ssse3"` is enabled at compile time.
        unsafe { validate_ssse3(input, hi_mask, lo_mask) }
    }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn load_16(chunk: &[u8; 16]) -> __m128i {
    // SAFETY: `chunk` contains 16 initialized readable bytes.
    unsafe { _mm_loadu_si128(chunk.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn classify_16_bytes(
    chunk: &[u8; 16],
    hi_mask: __m128i,
    lo_mask: __m128i,
    mask_0f: __m128i,
) -> __m128i {
    let v = load_16(chunk);
    let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);
    // PSHUFB indexes bits [3:0] for ASCII and zeroes any lane with bit 7 set.
    _mm_and_si128(_mm_shuffle_epi8(hi_mask, hi), _mm_shuffle_epi8(lo_mask, v))
}

#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
fn classify_32_ssse3(chunk: &[u8; 32], hi: __m128i, lo: __m128i, mask_0f: __m128i) -> __m128i {
    let blocks = chunk.as_chunks::<16>().0;
    _mm_min_epu8(
        classify_16_bytes(&blocks[0], hi, lo, mask_0f),
        classify_16_bytes(&blocks[1], hi, lo, mask_0f),
    )
}

#[cfg(not(target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "ssse3")]
fn classify_64_ssse3(chunk: &[u8; 64], hi: __m128i, lo: __m128i, mask_0f: __m128i) -> __m128i {
    let halves = chunk.as_chunks::<32>().0;
    _mm_min_epu8(
        classify_32_ssse3(&halves[0], hi, lo, mask_0f),
        classify_32_ssse3(&halves[1], hi, lo, mask_0f),
    )
}

#[cfg(not(target_feature = "avx2"))]
#[target_feature(enable = "ssse3")]
fn validate_ssse3(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    let len = input.len();
    let hi = load_16(hi_mask);
    let lo = load_16(lo_mask);
    let mask_0f = _mm_set1_epi8(0x0f);
    let zero = _mm_setzero_si128();
    let mut checked = 0;

    for batch in input.as_chunks::<SSSE3_BATCH_LEN>().0 {
        let agg = classify_64_ssse3(batch, hi, lo, mask_0f);
        if _mm_movemask_epi8(_mm_cmpeq_epi8(agg, zero)) != 0 {
            return find_invalid_ssse3_from(input, checked, hi_mask, lo_mask);
        }
        checked += SSSE3_BATCH_LEN;
    }

    let rem = len - checked;
    if rem != 0 {
        let agg = if len < 32 {
            let first = input
                .first_chunk::<16>()
                .expect("input has at least 16 bytes");
            let c0 = classify_16_bytes(first, hi, lo, mask_0f);
            if len == 16 {
                c0
            } else {
                let last = input[len - 16..]
                    .first_chunk::<16>()
                    .expect("input has at least 16 bytes");
                _mm_min_epu8(c0, classify_16_bytes(last, hi, lo, mask_0f))
            }
        } else if rem <= 16 {
            let last = input[len - 16..]
                .first_chunk::<16>()
                .expect("input has at least 16 bytes");
            classify_16_bytes(last, hi, lo, mask_0f)
        } else if rem <= 32 {
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            classify_32_ssse3(last, hi, lo, mask_0f)
        } else {
            let head = input[checked..]
                .first_chunk::<32>()
                .expect("remaining tail has at least 32 bytes");
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            _mm_min_epu8(
                classify_32_ssse3(head, hi, lo, mask_0f),
                classify_32_ssse3(last, hi, lo, mask_0f),
            )
        };
        if _mm_movemask_epi8(_mm_cmpeq_epi8(agg, zero)) != 0 {
            return find_invalid_ssse3_from(input, checked, hi_mask, lo_mask);
        }
    }

    Ok(len)
}

#[cfg(not(target_feature = "avx2"))]
#[cold]
#[inline(never)]
#[target_feature(enable = "ssse3")]
fn find_invalid_ssse3_from(
    input: &[u8],
    mut offset: usize,
    hi_mask: &[u8; 16],
    lo_mask: &[u8; 16],
) -> Result<usize, usize> {
    let hi = load_16(hi_mask);
    let lo = load_16(lo_mask);
    let mask_0f = _mm_set1_epi8(0x0f);
    let zero = _mm_setzero_si128();
    for block in input[offset..].as_chunks::<16>().0 {
        let invalid = _mm_movemask_epi8(_mm_cmpeq_epi8(
            classify_16_bytes(block, hi, lo, mask_0f),
            zero,
        )) as u32;
        if invalid != 0 {
            return Err(offset + invalid.trailing_zeros() as usize);
        }
        offset += 16;
    }
    let start = input.len() - 16;
    let last = input[start..]
        .first_chunk::<16>()
        .expect("input has at least 16 bytes");
    let invalid = _mm_movemask_epi8(_mm_cmpeq_epi8(
        classify_16_bytes(last, hi, lo, mask_0f),
        zero,
    )) as u32;
    debug_assert_ne!(invalid, 0);
    Err(start + invalid.trailing_zeros() as usize)
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn load_32(chunk: &[u8; 32]) -> __m256i {
    // SAFETY: `chunk` contains 32 initialized readable bytes.
    unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) }
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn classify_32_avx2(
    chunk: &[u8; 32],
    hi_mask: __m256i,
    lo_mask: __m256i,
    mask_0f: __m256i,
) -> __m256i {
    let v = load_32(chunk);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);
    // VPSHUFB indexes bits [3:0] for ASCII and zeroes any lane with bit 7 set.
    _mm256_and_si256(
        _mm256_shuffle_epi8(hi_mask, hi),
        _mm256_shuffle_epi8(lo_mask, v),
    )
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn classify_64_avx2(chunk: &[u8; 64], hi: __m256i, lo: __m256i, mask_0f: __m256i) -> __m256i {
    let halves = chunk.as_chunks::<32>().0;
    _mm256_min_epu8(
        classify_32_avx2(&halves[0], hi, lo, mask_0f),
        classify_32_avx2(&halves[1], hi, lo, mask_0f),
    )
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn classify_128_avx2(chunk: &[u8; 128], hi: __m256i, lo: __m256i, mask_0f: __m256i) -> __m256i {
    let halves = chunk.as_chunks::<64>().0;
    _mm256_min_epu8(
        classify_64_avx2(&halves[0], hi, lo, mask_0f),
        classify_64_avx2(&halves[1], hi, lo, mask_0f),
    )
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn validate_avx2(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    let len = input.len();
    if len < 32 {
        let hi = load_16(hi_mask);
        let lo = load_16(lo_mask);
        let mask_0f = _mm_set1_epi8(0x0f);
        let first = input
            .first_chunk::<16>()
            .expect("input has at least 16 bytes");
        let c0 = classify_16_bytes(first, hi, lo, mask_0f);
        let agg = if len == 16 {
            c0
        } else {
            let last = input[len - 16..]
                .first_chunk::<16>()
                .expect("input has at least 16 bytes");
            _mm_min_epu8(c0, classify_16_bytes(last, hi, lo, mask_0f))
        };
        if _mm_movemask_epi8(_mm_cmpeq_epi8(agg, _mm_setzero_si128())) != 0 {
            return find_invalid_avx2_from(input, 0, hi_mask, lo_mask);
        }
        return Ok(len);
    }
    if len >= AVX2_BATCH_LEN {
        return validate_avx2_bulk(input, hi_mask, lo_mask);
    }

    let hi = _mm256_broadcastsi128_si256(load_16(hi_mask));
    let lo = _mm256_broadcastsi128_si256(load_16(lo_mask));
    let mask_0f = _mm256_set1_epi8(0x0f);
    let agg = if len <= 64 {
        let first = input
            .first_chunk::<32>()
            .expect("input has at least 32 bytes");
        let c0 = classify_32_avx2(first, hi, lo, mask_0f);
        if len == 32 {
            c0
        } else {
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            _mm256_min_epu8(c0, classify_32_avx2(last, hi, lo, mask_0f))
        }
    } else if len <= 128 {
        let first = input
            .first_chunk::<64>()
            .expect("input has at least 64 bytes");
        let last = input[len - 64..]
            .first_chunk::<64>()
            .expect("input has at least 64 bytes");
        _mm256_min_epu8(
            classify_64_avx2(first, hi, lo, mask_0f),
            classify_64_avx2(last, hi, lo, mask_0f),
        )
    } else {
        let first = input
            .first_chunk::<128>()
            .expect("input has at least 128 bytes");
        let last = input[len - 128..]
            .first_chunk::<128>()
            .expect("input has at least 128 bytes");
        _mm256_min_epu8(
            classify_128_avx2(first, hi, lo, mask_0f),
            classify_128_avx2(last, hi, lo, mask_0f),
        )
    };

    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(agg, _mm256_setzero_si256())) != 0 {
        return find_invalid_avx2_from(input, 0, hi_mask, lo_mask);
    }
    Ok(len)
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(never)]
fn validate_avx2_bulk(
    input: &[u8],
    hi_mask: &[u8; 16],
    lo_mask: &[u8; 16],
) -> Result<usize, usize> {
    let len = input.len();
    let hi = _mm256_broadcastsi128_si256(load_16(hi_mask));
    let lo = _mm256_broadcastsi128_si256(load_16(lo_mask));
    let mask_0f = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();

    // Check the first 64 bytes before advancing to a 32-byte-aligned batch cursor.
    let head = input
        .first_chunk::<64>()
        .expect("bulk input has at least 256 bytes");
    if _mm256_movemask_epi8(_mm256_cmpeq_epi8(
        classify_64_avx2(head, hi, lo, mask_0f),
        zero,
    )) != 0
    {
        return find_invalid_avx2_from(input, 0, hi_mask, lo_mask);
    }

    let align = input.as_ptr().align_offset(32);
    let mut checked = if (1..32).contains(&align) {
        32 + align
    } else {
        64
    };

    for batch in input[checked..].as_chunks::<AVX2_BATCH_LEN>().0 {
        let halves = batch.as_chunks::<128>().0;
        let agg = _mm256_min_epu8(
            classify_128_avx2(&halves[0], hi, lo, mask_0f),
            classify_128_avx2(&halves[1], hi, lo, mask_0f),
        );
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(agg, zero)) != 0 {
            return find_invalid_avx2_from(input, checked, hi_mask, lo_mask);
        }
        checked += AVX2_BATCH_LEN;
    }

    let rem = len - checked;
    if rem != 0 {
        let agg = if rem <= 32 {
            let last = input[len - 32..]
                .first_chunk::<32>()
                .expect("input has at least 32 bytes");
            classify_32_avx2(last, hi, lo, mask_0f)
        } else if rem <= 64 {
            let last = input[len - 64..]
                .first_chunk::<64>()
                .expect("input has at least 64 bytes");
            classify_64_avx2(last, hi, lo, mask_0f)
        } else if rem <= 128 {
            let last = input[len - 128..]
                .first_chunk::<128>()
                .expect("input has at least 128 bytes");
            classify_128_avx2(last, hi, lo, mask_0f)
        } else {
            let tail_head = input[checked..]
                .first_chunk::<128>()
                .expect("remaining tail has at least 128 bytes");
            let last = input[len - 128..]
                .first_chunk::<128>()
                .expect("input has at least 128 bytes");
            _mm256_min_epu8(
                classify_128_avx2(tail_head, hi, lo, mask_0f),
                classify_128_avx2(last, hi, lo, mask_0f),
            )
        };
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(agg, zero)) != 0 {
            return find_invalid_avx2_from(input, checked, hi_mask, lo_mask);
        }
    }

    Ok(len)
}

#[cfg(any(feature = "std", target_feature = "avx2"))]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2")]
fn find_invalid_avx2_from(
    input: &[u8],
    mut offset: usize,
    hi_mask: &[u8; 16],
    lo_mask: &[u8; 16],
) -> Result<usize, usize> {
    let hi = load_16(hi_mask);
    let lo = load_16(lo_mask);
    let mask_0f = _mm_set1_epi8(0x0f);
    let zero = _mm_setzero_si128();
    for block in input[offset..].as_chunks::<16>().0 {
        let invalid = _mm_movemask_epi8(_mm_cmpeq_epi8(
            classify_16_bytes(block, hi, lo, mask_0f),
            zero,
        )) as u32;
        if invalid != 0 {
            return Err(offset + invalid.trailing_zeros() as usize);
        }
        offset += 16;
    }
    let start = input.len() - 16;
    let last = input[start..]
        .first_chunk::<16>()
        .expect("input has at least 16 bytes");
    let invalid = _mm_movemask_epi8(_mm_cmpeq_epi8(
        classify_16_bytes(last, hi, lo, mask_0f),
        zero,
    )) as u32;
    debug_assert_ne!(invalid, 0);
    Err(start + invalid.trailing_zeros() as usize)
}

#[cfg(all(test, not(target_feature = "avx2"), feature = "std", not(miri)))]
mod tests {
    use std::arch::is_x86_feature_detected;

    use super::*;
    use crate::backend::base64::HI_MASK;
    use crate::backend::base64::LO_MASK_STD;

    #[test]
    fn ssse3_reports_earliest_invalid_offset_across_batches_and_overlapping_tails() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        // Runtime dispatch selects AVX2 on modern x86 hosts; exercise SSSE3 directly.
        let check = |slice: &[u8]| {
            // SAFETY: Runtime feature detection above confirmed SSSE3 support.
            unsafe { validate_ssse3(slice, &HI_MASK, &LO_MASK_STD) }
        };
        let mut full = [b'A'; 95];
        for len in [16, 23, 32, 41, 64, 95] {
            let buf = &mut full[..len];
            buf.fill(b'A');
            assert_eq!(check(buf), Ok(len));

            let mid = len / 2;
            buf[mid] = b'!';
            buf[len - 1] = 0xff;
            assert_eq!(check(buf), Err(mid));

            buf[mid] = b'A';
            assert_eq!(check(buf), Err(len - 1));
        }
    }
}
