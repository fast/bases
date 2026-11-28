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

use core::error::Error;
use core::fmt;
#[cfg(feature = "std")]
use std::io;

/// An encoding error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The encoded length cannot be represented by `usize`.
    LengthOverflow,
    /// The output buffer is too small.
    OutputTooSmall {
        /// The number of bytes required.
        required: usize,
        /// The number of bytes available.
        available: usize,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthOverflow => formatter.write_str("encoded length overflow"),
            Self::OutputTooSmall {
                required,
                available,
            } => write!(
                formatter,
                "output buffer is too small: required {required} bytes, available {available}"
            ),
        }
    }
}

impl Error for EncodeError {}

/// An input validation error encountered while decoding.
///
/// Byte offsets are zero-based. For [`Decoder`](crate::Decoder), offsets and
/// invalid input lengths are measured from the start of the whole stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The encoded input has an impossible length for the selected profile.
    InvalidLength {
        /// The encoded input length.
        length: usize,
    },
    /// A byte does not belong to the selected alphabet.
    InvalidByte {
        /// The byte offset in the encoded input.
        index: usize,
        /// The invalid byte.
        byte: u8,
    },
    /// Padding is invalid, forbidden, or followed by more input.
    InvalidPadding {
        /// The padding byte offset. When more input follows a padded stream,
        /// this points to the start of that stream's padding.
        index: usize,
    },
    /// Unused bits in the final encoded symbol are not zero.
    NonCanonical {
        /// The byte offset of the non-canonical symbol.
        index: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { length } => {
                write!(formatter, "invalid encoded length: {length}")
            }
            Self::InvalidByte { index, byte } => {
                write!(formatter, "invalid byte 0x{byte:02x} at offset {index}")
            }
            Self::InvalidPadding { index } => {
                write!(formatter, "invalid padding at offset {index}")
            }
            Self::NonCanonical { index } => {
                write!(formatter, "non-canonical trailing bits at offset {index}")
            }
        }
    }
}

impl Error for DecodeError {}

/// A decoding error encountered when writing into a caller-supplied buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodeSliceError {
    /// The encoded input is invalid.
    Decode(DecodeError),
    /// The output buffer is too small.
    OutputTooSmall {
        /// The number of bytes required.
        required: usize,
        /// The number of bytes available.
        available: usize,
    },
    /// The cumulative encoded length cannot be represented by `usize`.
    LengthOverflow,
}

impl From<DecodeError> for DecodeSliceError {
    fn from(error: DecodeError) -> Self {
        Self::Decode(error)
    }
}

impl fmt::Display for DecodeSliceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => error.fmt(formatter),
            Self::OutputTooSmall {
                required,
                available,
            } => write!(
                formatter,
                "output buffer is too small: required {required} bytes, available {available}"
            ),
            Self::LengthOverflow => formatter.write_str("encoded length overflow"),
        }
    }
}

impl Error for DecodeSliceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::OutputTooSmall { .. } | Self::LengthOverflow => None,
        }
    }
}

/// A finalization error together with the stream that failed to finish.
///
/// Inspect [`error`](Self::error) for the cause. For buffer-size errors,
/// recover the stream with [`into_inner`](Self::into_inner) and retry with
/// a larger output buffer.
///
/// For [`Encoder`](crate::Encoder) and [`Decoder`](crate::Decoder), the
/// recovered stream is unchanged. With the `std` feature,
/// `io::EncoderWriter` preserves any progress made
/// before an I/O error so that retrying finalization resumes from the first
/// unwritten byte. It no longer accepts input, and dropping its error does
/// not retry finalization. See the finalization method for the recovered
/// stream's state and permitted operations.
///
/// Converting this error into [`EncodeError`], [`DecodeError`], or
/// [`DecodeSliceError`] discards the stream. This lets callers use `?` when
/// recovery is not needed. With `std`, the same applies to `std::io::Error`.
///
/// # Examples
///
/// ```
/// use bases::Base64;
/// use bases::EncodeError;
///
/// let mut encoder = Base64::encoder();
/// encoder.update(b"f", &mut [])?;
///
/// let mut output = [0_u8; 4];
/// let failure = encoder.finish(&mut output[..3]).unwrap_err();
/// assert_eq!(
///     failure.error(),
///     &EncodeError::OutputTooSmall {
///         required: 4,
///         available: 3
///     },
/// );
///
/// let encoder = failure.into_inner();
/// assert_eq!(encoder.finish(&mut output)?, "Zg==");
/// # Ok::<(), EncodeError>(())
/// ```
pub struct FinishError<S, E> {
    stream: S,
    error: E,
}

impl<S, E> FinishError<S, E> {
    pub(crate) fn new(stream: S, error: E) -> Self {
        Self { stream, error }
    }

    /// Returns the error that caused finalization to fail.
    #[must_use]
    pub fn error(&self) -> &E {
        &self.error
    }

    /// Recovers the stream, discarding the error.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.stream
    }

    /// Returns the underlying error, discarding the stream.
    #[must_use]
    pub fn into_error(self) -> E {
        self.error
    }

    /// Returns `(error, stream)`, preserving the stream's state at failure.
    #[must_use]
    pub fn into_parts(self) -> (E, S) {
        (self.error, self.stream)
    }
}

impl<S, E: fmt::Debug> fmt::Debug for FinishError<S, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deriving Debug would impose S: Debug on generic codec callers.
        formatter
            .debug_struct("FinishError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<S, E: fmt::Display> fmt::Display for FinishError<S, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl<S, E: Error + 'static> Error for FinishError<S, E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}

impl<S> From<FinishError<S, EncodeError>> for EncodeError {
    fn from(error: FinishError<S, EncodeError>) -> Self {
        error.into_error()
    }
}

impl<S> From<FinishError<S, DecodeError>> for DecodeError {
    fn from(error: FinishError<S, DecodeError>) -> Self {
        error.into_error()
    }
}

impl<S> From<FinishError<S, DecodeSliceError>> for DecodeSliceError {
    fn from(error: FinishError<S, DecodeSliceError>) -> Self {
        error.into_error()
    }
}

#[cfg(feature = "std")]
impl<S> From<FinishError<S, io::Error>> for io::Error {
    fn from(error: FinishError<S, io::Error>) -> Self {
        error.into_error()
    }
}

#[cfg(test)]
mod tests {
    use core::error::Error;

    use crate::Base64;
    use crate::DecodeError;
    use crate::DecodeSliceError;

    #[test]
    fn finish_exposes_the_codec_error_as_its_source() {
        let mut decoder = Base64::decoder();
        decoder.update("Zg", &mut []).unwrap();
        let failure = decoder.finish(&mut []).unwrap_err();
        let source = failure.source().unwrap();
        assert_eq!(
            source.downcast_ref::<DecodeSliceError>(),
            Some(&DecodeSliceError::Decode(DecodeError::InvalidLength {
                length: 2,
            }))
        );
        assert_eq!(
            source
                .source()
                .and_then(|err| err.downcast_ref::<DecodeError>()),
            Some(&DecodeError::InvalidLength { length: 2 })
        );
    }
}
