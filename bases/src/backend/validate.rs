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

//! Shared SIMD and scalar alphabet validation.

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
#[allow(unsafe_code)]
mod aarch64;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod wasm32;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
))]
#[allow(unsafe_code)]
mod x86;

/// Returns the byte offset of the first symbol rejected by `lut`, or `None` if all bytes are valid.
///
/// `lut` must have bit 7 (`0x80`) set if and only if an entry is invalid, and `(hi_mask, lo_mask)`
/// must classify the exact same ASCII subset via `(hi_mask[b >> 4] & lo_mask[b & 0x0f]) != 0` with
/// `hi_mask[8..16]` all zero.
#[inline(always)]
pub fn validate_symbols(
    input: &[u8],
    lut: &[u8; 256],
    hi_mask: &[u8; 16],
    lo_mask: &[u8; 16],
) -> Option<usize> {
    if input.len() < 16 {
        return find_invalid_short(input, lut);
    }
    match arch_validate(input, hi_mask, lo_mask) {
        Ok(checked) => {
            if checked == input.len() {
                None
            } else {
                find_invalid_scalar(&input[checked..], lut).map(|offset| checked + offset)
            }
        }
        Err(index) => Some(index),
    }
}

#[inline(always)]
fn arch_validate(input: &[u8], hi_mask: &[u8; 16], lo_mask: &[u8; 16]) -> Result<usize, usize> {
    if cfg!(miri) {
        return Ok(0);
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    {
        return aarch64::validate(input, hi_mask, lo_mask);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return x86::validate(input, hi_mask, lo_mask);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm32::validate(input, hi_mask, lo_mask);
    }

    #[allow(unreachable_code)]
    let _ = (input, hi_mask, lo_mask);
    #[allow(unreachable_code)]
    Ok(0)
}

#[inline]
fn find_invalid_short(input: &[u8], lut: &[u8; 256]) -> Option<usize> {
    input
        .iter()
        .position(|&byte| lut[usize::from(byte)] & 0x80 != 0)
}

#[inline]
fn find_invalid_scalar(input: &[u8], lut: &[u8; 256]) -> Option<usize> {
    if input
        .first()
        .is_some_and(|&byte| lut[usize::from(byte)] & 0x80 != 0)
    {
        return Some(0);
    }
    find_invalid_blocks(input, lut)
}

#[inline(never)]
fn find_invalid_blocks(input: &[u8], lut: &[u8; 256]) -> Option<usize> {
    let (chunks, tail) = input.as_chunks::<8>();
    for (chunk_index, s) in chunks.iter().enumerate() {
        let v0 = lut[usize::from(s[0])] | lut[usize::from(s[1])];
        let v1 = lut[usize::from(s[2])] | lut[usize::from(s[3])];
        let v2 = lut[usize::from(s[4])] | lut[usize::from(s[5])];
        let v3 = lut[usize::from(s[6])] | lut[usize::from(s[7])];
        if ((v0 | v1) | (v2 | v3)) & 0x80 != 0 {
            return find_invalid_short(s, lut).map(|offset| chunk_index * 8 + offset);
        }
    }
    find_invalid_short(tail, lut).map(|offset| chunks.len() * 8 + offset)
}
