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

use core::fmt;
use core::hash::Hash;
use core::hash::Hasher;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::Base16Lower;
use crate::Base16Upper;
use crate::Codec;
use crate::codec;
use crate::codec::sealed::Profile;

const OUTPUT_CHUNK_LEN: usize = 1024;

/// A borrowed binary value encoded lazily with profile `C`.
///
/// Create this adapter with [`Codec::display`]. Formatting does not allocate.
/// Formatting flags are ignored; the adapter always emits the complete encoding.
///
/// # Examples
///
/// ```
/// use core::fmt::Write;
///
/// use bases::Base64UrlUnpadded;
///
/// # struct StackBuf {
/// #     buf: [u8; 16],
/// #     len: usize,
/// # }
/// # impl Write for StackBuf {
/// #     fn write_str(&mut self, s: &str) -> core::fmt::Result {
/// #         self.buf[self.len..self.len + s.len()].copy_from_slice(s.as_bytes());
/// #         self.len += s.len();
/// #         Ok(())
/// #     }
/// # }
/// let mut out = StackBuf {
///     buf: [0; 16],
///     len: 0,
/// };
/// write!(&mut out, "{}", Base64UrlUnpadded::display(b"\xfb\xff")).unwrap();
/// assert_eq!(&out.buf[..out.len], b"-_8");
/// ```
pub struct EncodedDisplay<'input, C: Codec> {
    input: &'input [u8],
    profile: PhantomData<C>,
}

impl<C: Codec> Clone for EncodedDisplay<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: Codec> Copy for EncodedDisplay<'_, C> {}

impl<C: Codec> fmt::Debug for EncodedDisplay<'_, C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncodedDisplay")
            .field("profile", &C::name())
            .field("input", &self.input)
            .finish()
    }
}

impl<C: Codec> PartialEq for EncodedDisplay<'_, C> {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
    }
}

impl<C: Codec> Eq for EncodedDisplay<'_, C> {}

impl<C: Codec> Hash for EncodedDisplay<'_, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
    }
}

impl<'input, C: Codec> EncodedDisplay<'input, C> {
    pub(crate) const fn new(input: &'input [u8]) -> Self {
        Self {
            input,
            profile: PhantomData,
        }
    }
}

impl<C: Codec> fmt::Display for EncodedDisplay<'_, C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_encoded::<C>(self.input, formatter)
    }
}

/// A borrowed byte slice formatted as Base16 with explicit case selection.
///
/// Only [`fmt::LowerHex`] (`{:x}`) and [`fmt::UpperHex`] (`{:X}`) are
/// implemented; [`fmt::Display`] (`{}`) is intentionally omitted so callers
/// always choose the output case explicitly. Formatting does not allocate.
/// Other formatting flags are ignored; the complete encoding is always emitted.
///
/// ```compile_fail,E0277
/// fn requires_display<T: core::fmt::Display>(_: T) {}
/// requires_display(bases::HexDisplay(b"hi"));
/// ```
///
/// # Examples
///
/// ```
/// use core::fmt::Write;
///
/// use bases::HexDisplay;
///
/// # struct StackBuf {
/// #     buf: [u8; 16],
/// #     len: usize,
/// # }
/// # impl Write for StackBuf {
/// #     fn write_str(&mut self, s: &str) -> core::fmt::Result {
/// #         self.buf[self.len..self.len + s.len()].copy_from_slice(s.as_bytes());
/// #         self.len += s.len();
/// #         Ok(())
/// #     }
/// # }
/// let value = HexDisplay(b"\xde\xad\xbe\xef");
/// let mut out = StackBuf {
///     buf: [0; 16],
///     len: 0,
/// };
/// write!(&mut out, "{value:x}").unwrap();
/// assert_eq!(&out.buf[..out.len], b"deadbeef");
///
/// out.len = 0;
/// write!(&mut out, "{value:X}").unwrap();
/// assert_eq!(&out.buf[..out.len], b"DEADBEEF");
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HexDisplay<'input>(
    /// The bytes to format.
    pub &'input [u8],
);

impl fmt::UpperHex for HexDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_encoded::<Base16Upper>(self.0, formatter)
    }
}

impl fmt::LowerHex for HexDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_encoded::<Base16Lower>(self.0, formatter)
    }
}

fn fmt_encoded<C: Codec>(input: &[u8], formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    if input.is_empty() {
        return formatter.write_str("");
    }

    let small_cutoff = (64 / C::Profile::ENCODED_QUANTUM) * C::Profile::INPUT_QUANTUM;
    if input.len() <= small_cutoff {
        let mut output = [MaybeUninit::<u8>::uninit(); 64];
        let written = C::Profile::encode_impl(input, &mut output);
        #[allow(unsafe_code)]
        // SAFETY: Binary-to-text backends write `written` initialized ASCII bytes into `output`.
        let encoded = unsafe { codec::uninit_ascii_to_str(&output[..written]) };
        return formatter.write_str(encoded);
    }

    let input_chunk_len =
        (OUTPUT_CHUNK_LEN / C::Profile::ENCODED_QUANTUM) * C::Profile::INPUT_QUANTUM;
    let mut output = [MaybeUninit::<u8>::uninit(); OUTPUT_CHUNK_LEN];
    for chunk in input.chunks(input_chunk_len) {
        let written = C::Profile::encode_impl(chunk, &mut output);
        #[allow(unsafe_code)]
        // SAFETY: Binary-to-text backends write `written` initialized ASCII bytes into `output`.
        let encoded = unsafe { codec::uninit_ascii_to_str(&output[..written]) };
        formatter.write_str(encoded)?;
    }

    Ok(())
}
