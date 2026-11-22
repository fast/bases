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

//! Private execution backends.
//!
//! Callers must provide sufficient output capacity and validate input length
//! and padding shape before decoding. Backends validate alphabet symbols and
//! unused trailing bits. CPU-specific implementations must preserve these contracts.

pub mod base16;
pub mod base32;
pub mod base64;

mod validate;

use core::mem::MaybeUninit;

pub use self::validate::validate_symbols;
use crate::DecodeError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodePlan {
    pub required: usize,
    pub data_len: usize,
    pub complete_len: usize,
    pub padding: usize,
}

#[inline(always)]
pub fn write_bytes<const N: usize>(dst: &mut [MaybeUninit<u8>], src: [u8; N]) {
    dst[..N].copy_from_slice(&src.map(MaybeUninit::new));
}

#[inline]
pub fn invalid_byte_error(byte: u8, index: usize) -> DecodeError {
    if byte == b'=' {
        DecodeError::InvalidPadding { index }
    } else {
        DecodeError::InvalidByte { index, byte }
    }
}

#[inline]
pub fn trailing_padding(input: &[u8]) -> usize {
    input.iter().rev().take_while(|byte| **byte == b'=').count()
}
