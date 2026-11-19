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

use core::mem::MaybeUninit;

use crate::DecodeError;

const CHECK_CHUNK_LEN: usize = 8;

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

/// Finds the first invalid symbol using a table with the high bit set only for invalid entries.
#[inline]
pub fn find_invalid(input: &[u8], lut: &[u8; 256]) -> Option<usize> {
    if input
        .first()
        .is_some_and(|&byte| lut[usize::from(byte)] & 0x80 != 0)
    {
        return Some(0);
    }
    find_invalid_blocks(input, lut)
}

// Keep block scanning out of line so early errors avoid its register pressure.
#[inline(never)]
fn find_invalid_blocks(input: &[u8], lut: &[u8; 256]) -> Option<usize> {
    let (chunks, tail) = input.as_chunks::<CHECK_CHUNK_LEN>();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let values = chunk
            .iter()
            .fold(0, |values, &byte| values | lut[usize::from(byte)]);
        if values & 0x80 != 0 {
            return chunk
                .iter()
                .position(|&byte| lut[usize::from(byte)] & 0x80 != 0)
                .map(|offset| chunk_index * CHECK_CHUNK_LEN + offset);
        }
    }
    tail.iter()
        .position(|&byte| lut[usize::from(byte)] & 0x80 != 0)
        .map(|offset| chunks.len() * CHECK_CHUNK_LEN + offset)
}

#[inline]
pub fn trailing_padding(input: &[u8]) -> usize {
    input.iter().rev().take_while(|byte| **byte == b'=').count()
}
