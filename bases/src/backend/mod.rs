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

use crate::DecodeError;

#[inline]
pub(crate) fn invalid_byte_error(byte: u8, index: usize) -> DecodeError {
    if byte == b'=' {
        DecodeError::InvalidPadding { index }
    } else {
        DecodeError::InvalidByte { index, byte }
    }
}

#[inline]
pub(crate) fn trailing_padding(input: &[u8]) -> usize {
    input.iter().rev().take_while(|byte| **byte == b'=').count()
}
