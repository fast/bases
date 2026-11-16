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

//! Base16 (hexadecimal) backend dispatch.

use core::mem::MaybeUninit;

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
pub fn encode<const UPPERCASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    if cfg!(miri) {
        return (0, 0);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return aarch64::encode::<UPPERCASE>(input, output);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return x86::encode::<UPPERCASE>(input, output);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm32::encode::<UPPERCASE>(input, output);
    }

    #[allow(unreachable_code)]
    let _ = (input, output);
    #[allow(unreachable_code)]
    (0, 0)
}

#[inline]
pub fn decode<const UPPERCASE: bool, const STRICT_CASE: bool>(
    input: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> (usize, usize) {
    if cfg!(miri) {
        return (0, 0);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return aarch64::decode::<UPPERCASE, STRICT_CASE>(input, output);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return x86::decode::<UPPERCASE, STRICT_CASE>(input, output);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm32::decode::<UPPERCASE, STRICT_CASE>(input, output);
    }

    #[allow(unreachable_code)]
    let _ = (input, output);
    #[allow(unreachable_code)]
    (0, 0)
}

#[inline]
pub fn decode_in_place<const UPPERCASE: bool, const STRICT_CASE: bool>(
    buf: &mut [u8],
) -> (usize, usize) {
    if cfg!(miri) {
        return (0, 0);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return aarch64::decode_in_place::<UPPERCASE, STRICT_CASE>(buf);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    {
        return x86::decode_in_place::<UPPERCASE, STRICT_CASE>(buf);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm32::decode_in_place::<UPPERCASE, STRICT_CASE>(buf);
    }

    #[allow(unreachable_code)]
    let _ = buf;
    #[allow(unreachable_code)]
    (0, 0)
}
