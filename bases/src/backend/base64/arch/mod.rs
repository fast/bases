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

//! Base64 backend dispatch.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
))]
mod x86;

#[inline]
pub fn encode<const URL_SAFE_ALPHABET: bool>(input: &[u8], output: &mut [u8]) -> (usize, usize) {
    if cfg!(miri) {
        return (0, 0);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return self::aarch64::encode::<URL_SAFE_ALPHABET>(input, output);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return self::x86::encode::<URL_SAFE_ALPHABET>(input, output);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return self::wasm32::encode::<URL_SAFE_ALPHABET>(input, output);
    }

    #[allow(unreachable_code)]
    let _ = (input, output);
    #[allow(unreachable_code)]
    (0, 0)
}

#[inline]
pub fn decode<const URL_SAFE_ALPHABET: bool>(
    input: &[u8],
    output: &mut [u8],
) -> (usize, usize, bool) {
    if cfg!(miri) {
        return (0, 0, false);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return self::aarch64::decode::<URL_SAFE_ALPHABET>(input, output);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return self::x86::decode::<URL_SAFE_ALPHABET>(input, output);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return self::wasm32::decode::<URL_SAFE_ALPHABET>(input, output);
    }

    #[allow(unreachable_code)]
    let _ = (input, output);
    #[allow(unreachable_code)]
    (0, 0, false)
}

#[inline]
pub fn decode_in_place<const URL_SAFE_ALPHABET: bool>(
    buf: &mut [u8],
    complete_len: usize,
) -> (usize, usize, bool) {
    if cfg!(miri) {
        return (0, 0, false);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return self::aarch64::decode_in_place::<URL_SAFE_ALPHABET>(buf, complete_len);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return self::x86::decode_in_place::<URL_SAFE_ALPHABET>(buf, complete_len);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return self::wasm32::decode_in_place::<URL_SAFE_ALPHABET>(buf, complete_len);
    }

    #[allow(unreachable_code)]
    let _ = (buf, complete_len);
    #[allow(unreachable_code)]
    (0, 0, false)
}
