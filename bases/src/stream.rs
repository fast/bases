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
use core::marker::PhantomData;

use crate::Codec;
use crate::DecodeError;
use crate::DecodeSliceError;
use crate::EncodeError;
use crate::FinishError;
use crate::backend::DecodePlan;
use crate::codec;
use crate::codec::sealed::Profile;

const MAX_INPUT_QUANTUM: usize = 5;
const MAX_ENCODED_QUANTUM: usize = 8;

/// An incremental encoder for profile `C`.
///
/// Input can be split at any byte boundary. Call [`finish`](Self::finish)
/// after the last [`update`](Self::update) to emit the remaining symbols and
/// padding. The encoder uses bounded storage and does not allocate.
///
/// # Examples
///
/// ```
/// use bases::Base64;
///
/// let mut encoder = Base64::encoder();
/// let mut first = [0_u8; 4];
/// let mut last = [0_u8; 4];
/// assert_eq!(encoder.update(b"fo", &mut first).unwrap(), "");
/// assert_eq!(encoder.update(b"o", &mut first).unwrap(), "Zm9v");
/// assert_eq!(encoder.finish(&mut last).unwrap(), "");
/// ```
pub struct Encoder<C: Codec> {
    pending: [u8; MAX_INPUT_QUANTUM],
    pending_len: usize,
    profile: PhantomData<C>,
}

impl<C: Codec> Clone for Encoder<C> {
    fn clone(&self) -> Self {
        Self {
            pending: self.pending,
            pending_len: self.pending_len,
            profile: PhantomData,
        }
    }
}

impl<C: Codec> fmt::Debug for Encoder<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Encoder")
            .field("profile", &C::name())
            .field("pending_len", &self.pending_len)
            .finish_non_exhaustive()
    }
}

impl<C: Codec> Encoder<C> {
    /// Creates an encoder with no buffered input.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: [0; MAX_INPUT_QUANTUM],
            pending_len: 0,
            profile: PhantomData,
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending_len
    }

    /// Computes the exact output length of the next [`update`](Self::update).
    ///
    /// Includes input buffered by earlier updates.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::LengthOverflow`] if the result cannot be
    /// represented by `usize`.
    pub fn update_output_len(&self, input_len: usize) -> Result<usize, EncodeError> {
        let input_quantum = C::Profile::INPUT_QUANTUM;
        let complete = self
            .pending_len
            .checked_add(input_len)
            .ok_or(EncodeError::LengthOverflow)?
            / input_quantum;
        complete
            .checked_mul(C::Profile::ENCODED_QUANTUM)
            .ok_or(EncodeError::LengthOverflow)
    }

    /// Computes the exact output length of [`finish`](Self::finish).
    #[must_use]
    pub fn finish_output_len(&self) -> usize {
        C::Profile::encoded_len_impl(self.pending_len)
            .expect("a buffered fragment fits in one encoded quantum")
    }

    /// Encodes another input chunk, returning the written prefix of `output`.
    ///
    /// On success, consumes all of `input` and retains any incomplete quantum.
    /// The returned string may be empty until a complete quantum is available.
    /// Use [`update_output_len`](Self::update_output_len) to size `output`.
    ///
    /// On error, the encoder is unchanged and output contents are unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::LengthOverflow`] if the output length overflows
    /// `usize`, or [`EncodeError::OutputTooSmall`] if `output` is shorter than
    /// [`update_output_len`](Self::update_output_len).
    pub fn update<'output>(
        &mut self,
        input: &[u8],
        output: &'output mut [u8],
    ) -> Result<&'output mut str, EncodeError> {
        let required = self.update_output_len(input.len())?;
        if output.len() < required {
            return Err(EncodeError::OutputTooSmall {
                required,
                available: output.len(),
            });
        }

        let input_quantum = C::Profile::INPUT_QUANTUM;
        let encoded_quantum = C::Profile::ENCODED_QUANTUM;
        let out = &mut output[..required];
        let mut input_index = 0;
        let mut output_index = 0;

        if self.pending_len != 0 {
            let needed = input_quantum - self.pending_len;
            let taken = needed.min(input.len());
            self.pending[self.pending_len..self.pending_len + taken]
                .copy_from_slice(&input[..taken]);
            self.pending_len += taken;
            input_index += taken;

            if self.pending_len == input_quantum {
                output_index += C::Profile::encode_quantum_impl(
                    &self.pending[..input_quantum],
                    codec::as_uninit_slice(&mut out[output_index..output_index + encoded_quantum]),
                );
                self.pending_len = 0;
            }
        }

        if self.pending_len == 0 {
            let remaining = &input[input_index..];
            let complete_len = remaining.len() / input_quantum * input_quantum;
            if complete_len != 0 {
                output_index += C::Profile::encode_impl(
                    &remaining[..complete_len],
                    codec::as_uninit_slice(&mut out[output_index..]),
                );
            }

            let trailing = &remaining[complete_len..];
            self.pending[..trailing.len()].copy_from_slice(trailing);
            self.pending_len = trailing.len();
        }

        debug_assert_eq!(output_index, required);
        #[allow(unsafe_code)]
        // SAFETY: Binary-to-text backends always write ASCII bytes into `out`.
        Ok(unsafe { codec::ascii_to_str_mut(out) })
    }

    /// Finishes the stream, returning the final encoded prefix of `output`.
    ///
    /// On error, [`FinishError::into_inner`] recovers the unchanged encoder
    /// for retrying with a larger buffer. Output contents are unspecified on
    /// error.
    ///
    /// # Errors
    ///
    /// Returns [`FinishError`] carrying [`EncodeError::OutputTooSmall`] if
    /// `output` is shorter than [`finish_output_len`](Self::finish_output_len).
    pub fn finish(self, output: &mut [u8]) -> Result<&mut str, FinishError<Self, EncodeError>> {
        C::encode_into(&self.pending[..self.pending_len], output)
            .map_err(|error| FinishError::new(self, error))
    }
}

impl<C: Codec> Default for Encoder<C> {
    fn default() -> Self {
        Self::new()
    }
}

/// An incremental decoder for profile `C`.
///
/// Input can be split at any byte boundary. Call [`finish`](Self::finish)
/// after the last [`update`](Self::update), even if no output remains, to
/// validate the final fragment. The decoder uses bounded storage and does
/// not allocate. Error offsets are measured from the start of the stream.
///
/// # Examples
///
/// ```
/// use bases::Base64;
///
/// let mut decoder = Base64::decoder();
/// let mut first = [0_u8; 3];
/// let mut last = [0_u8; 3];
/// assert_eq!(decoder.update(b"Zm", &mut first).unwrap(), b"");
/// assert_eq!(decoder.update(b"9v", &mut first).unwrap(), b"foo");
/// assert_eq!(decoder.finish(&mut last).unwrap(), b"");
/// ```
pub struct Decoder<C: Codec> {
    pending: [u8; MAX_ENCODED_QUANTUM],
    pending_len: usize,
    total_input_len: usize,
    padding_index: Option<usize>,
    profile: PhantomData<C>,
}

impl<C: Codec> Clone for Decoder<C> {
    fn clone(&self) -> Self {
        Self {
            pending: self.pending,
            pending_len: self.pending_len,
            total_input_len: self.total_input_len,
            padding_index: self.padding_index,
            profile: PhantomData,
        }
    }
}

impl<C: Codec> fmt::Debug for Decoder<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Decoder")
            .field("profile", &C::name())
            .field("pending_len", &self.pending_len)
            .field("total_input_len", &self.total_input_len)
            .field("padding_index", &self.padding_index)
            .finish_non_exhaustive()
    }
}

impl<C: Codec> Decoder<C> {
    /// Creates a decoder with no buffered input.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: [0; MAX_ENCODED_QUANTUM],
            pending_len: 0,
            total_input_len: 0,
            padding_index: None,
            profile: PhantomData,
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending_len
    }

    /// Computes an output-length upper bound for the next [`update`](Self::update).
    ///
    /// Includes input buffered by earlier updates. Actual output can be
    /// shorter when the chunk completes a padded final quantum. This accepts
    /// every `usize` without overflow, but does not validate input or check
    /// cumulative stream length.
    #[must_use]
    pub fn max_update_output_len(&self, input_len: usize) -> usize {
        let encoded_quantum = C::Profile::ENCODED_QUANTUM;
        // Add only the remainder to pending input, even for usize::MAX queries.
        let complete = input_len / encoded_quantum
            + (input_len % encoded_quantum + self.pending_len) / encoded_quantum;
        complete * C::Profile::INPUT_QUANTUM
    }

    /// Computes the exact output length of [`finish`](Self::finish).
    ///
    /// This validates the buffered fragment's length and padding shape, but
    /// alphabet and canonical-bit validation still happens during `finish`.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the buffered trailing fragment has an invalid
    /// length or padding shape.
    pub fn finish_output_len(&self) -> Result<usize, DecodeError> {
        if self.padding_index.is_some() || self.pending_len == 0 {
            return Ok(0);
        }

        C::Profile::decode_plan_impl(&self.pending[..self.pending_len])
            .map(|plan| plan.required)
            .map_err(|error| {
                adjust_decode_error(error, self.pending_offset(), self.total_input_len)
            })
    }

    /// Decodes another input chunk, returning the written prefix of `output`.
    ///
    /// On success, consumes all of `input` and retains any incomplete quantum.
    /// The returned slice may be empty until a complete quantum is available.
    /// After padding, non-empty input returns [`DecodeError::InvalidPadding`]
    /// at the original padding offset.
    ///
    /// On error, the decoder is unchanged and output contents are unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeSliceError::LengthOverflow`] if the cumulative stream
    /// length overflows `usize`, [`DecodeSliceError::OutputTooSmall`] if
    /// `output` cannot fit the complete quanta, or [`DecodeSliceError::Decode`]
    /// if the chunk contains invalid encoded data.
    pub fn update<'output>(
        &mut self,
        input: impl AsRef<[u8]>,
        output: &'output mut [u8],
    ) -> Result<&'output mut [u8], DecodeSliceError> {
        let input = input.as_ref();
        if input.is_empty() {
            return Ok(&mut output[..0]);
        }
        if let Some(index) = self.padding_index {
            return Err(DecodeError::InvalidPadding { index }.into());
        }

        let total_input_len = self
            .total_input_len
            .checked_add(input.len())
            .ok_or(DecodeSliceError::LengthOverflow)?;
        // Pending input is already part of the checked cumulative length.
        let combined_len = self.pending_len + input.len();
        let encoded_quantum = C::Profile::ENCODED_QUANTUM;
        let process_len = combined_len / encoded_quantum * encoded_quantum;
        let pending_offset = self.pending_offset();

        let (padding, padding_reduction, padding_index) = if C::Profile::PADDED {
            analyze_padding::<C>(
                &self.pending[..self.pending_len],
                input,
                process_len,
                pending_offset,
            )?
        } else {
            (0, 0, None)
        };

        let input_quantum = C::Profile::INPUT_QUANTUM;
        let complete_quanta = process_len / encoded_quantum;
        let required = complete_quanta * input_quantum - padding_reduction;
        if output.len() < required {
            return Err(DecodeSliceError::OutputTooSmall {
                required,
                available: output.len(),
            });
        }

        let out = &mut output[..required];
        // Commit state only after the whole chunk validates, so a late error
        // leaves buffered input and global offsets available for retry.
        let mut pending = self.pending;
        let mut pending_len = self.pending_len;
        let mut input_index = 0;
        let mut output_index = 0;

        if pending_len != 0 {
            let needed = encoded_quantum - pending_len;
            let taken = needed.min(input.len());
            pending[pending_len..pending_len + taken].copy_from_slice(&input[..taken]);
            pending_len += taken;
            input_index += taken;

            if pending_len == encoded_quantum {
                let remaining_complete_len =
                    (input.len() - input_index) / encoded_quantum * encoded_quantum;
                let written = if padding != 0 && remaining_complete_len == 0 {
                    let plan = DecodePlan {
                        required: input_quantum - padding_reduction,
                        data_len: encoded_quantum - padding,
                        complete_len: 0,
                        padding,
                    };
                    C::Profile::decode_with_plan_impl(
                        &pending[..encoded_quantum],
                        plan,
                        codec::as_uninit_slice(
                            &mut out[output_index..output_index + plan.required],
                        ),
                    )
                } else {
                    C::Profile::decode_quantum_impl(
                        &pending[..encoded_quantum],
                        codec::as_uninit_slice(
                            &mut out[output_index..output_index + input_quantum],
                        ),
                    )
                }
                .map_err(|error| adjust_decode_error(error, pending_offset, total_input_len))?;
                output_index += written;
                pending_len = 0;
            }
        }

        if pending_len == 0 {
            let remaining = &input[input_index..];
            let complete_len = remaining.len() / encoded_quantum * encoded_quantum;
            if complete_len != 0 {
                let offset = self.total_input_len + input_index;
                let chunk_required =
                    (complete_len / encoded_quantum) * input_quantum - padding_reduction;
                let chunk_data_len = complete_len - padding;
                let plan = DecodePlan {
                    required: chunk_required,
                    data_len: chunk_data_len,
                    complete_len: (chunk_data_len / encoded_quantum) * encoded_quantum,
                    padding,
                };
                output_index += C::Profile::decode_with_plan_impl(
                    &remaining[..complete_len],
                    plan,
                    codec::as_uninit_slice(&mut out[output_index..]),
                )
                .map_err(|error| adjust_decode_error(error, offset, total_input_len))?;
            }

            let trailing = &remaining[complete_len..];
            pending[..trailing.len()].copy_from_slice(trailing);
            pending_len = trailing.len();
        }

        self.pending = pending;
        self.pending_len = pending_len;
        self.total_input_len = total_input_len;
        self.padding_index = padding_index;

        debug_assert_eq!(output_index, required);
        Ok(out)
    }

    /// Validates the final fragment, returning the decoded prefix of `output`.
    ///
    /// On error, [`FinishError::into_inner`] recovers the unchanged decoder.
    /// Output contents are unspecified on error.
    ///
    /// Padded profiles require a complete final quantum; unpadded profiles
    /// accept the partial lengths defined by their encoding.
    ///
    /// # Errors
    ///
    /// Returns [`FinishError`] carrying [`DecodeSliceError::OutputTooSmall`] if
    /// `output` is too short, or [`DecodeSliceError::Decode`] if the buffered
    /// final fragment is invalid.
    pub fn finish(
        self,
        output: &mut [u8],
    ) -> Result<&mut [u8], FinishError<Self, DecodeSliceError>> {
        if self.padding_index.is_some() || self.pending_len == 0 {
            return Ok(&mut output[..0]);
        }

        match C::decode_into(&self.pending[..self.pending_len], output) {
            Ok(decoded) => Ok(decoded),
            Err(error) => {
                let error = adjust_slice_error(error, self.pending_offset(), self.total_input_len);
                Err(FinishError::new(self, error))
            }
        }
    }

    fn pending_offset(&self) -> usize {
        self.total_input_len - self.pending_len
    }
}

impl<C: Codec> Default for Decoder<C> {
    fn default() -> Self {
        Self::new()
    }
}

fn analyze_padding<C: Codec>(
    pending: &[u8],
    input: &[u8],
    process_len: usize,
    stream_offset: usize,
) -> Result<(usize, usize, Option<usize>), DecodeError> {
    let combined_len = pending.len() + input.len();

    if process_len == 0 {
        return Ok((0, 0, None));
    }

    let reject_padding = |start: usize| {
        let mut quantum = start..start + C::Profile::ENCODED_QUANTUM;
        if let Some(index) = quantum.find(|index| combined_byte(pending, input, *index) == b'=') {
            return Err(DecodeError::InvalidPadding {
                index: stream_offset + index,
            });
        }
        Ok(())
    };

    // The backend decodes the buffered quantum separately from the contiguous
    // input. Padding at the end of either slice is legal only at stream end.
    if !pending.is_empty() && process_len > C::Profile::ENCODED_QUANTUM {
        reject_padding(0)?;
    }
    if process_len != combined_len {
        reject_padding(process_len - C::Profile::ENCODED_QUANTUM)?;
        return Ok((0, 0, None));
    }

    let mut padding = 0;
    while padding < combined_len
        && combined_byte(pending, input, combined_len - padding - 1) == b'='
    {
        padding += 1;
    }
    if padding == 0 {
        return Ok((0, 0, None));
    }

    let first_padding = combined_len - padding;
    let reduction =
        C::Profile::decoded_padding_reduction(padding).ok_or(DecodeError::InvalidPadding {
            index: stream_offset + first_padding,
        })?;
    Ok((padding, reduction, Some(stream_offset + first_padding)))
}

fn combined_byte(pending: &[u8], input: &[u8], index: usize) -> u8 {
    if index < pending.len() {
        pending[index]
    } else {
        input[index - pending.len()]
    }
}

fn adjust_decode_error(error: DecodeError, offset: usize, total_input_len: usize) -> DecodeError {
    match error {
        DecodeError::InvalidLength { .. } => DecodeError::InvalidLength {
            length: total_input_len,
        },
        DecodeError::InvalidByte { index, byte } => DecodeError::InvalidByte {
            index: offset + index,
            byte,
        },
        DecodeError::InvalidPadding { index } => DecodeError::InvalidPadding {
            index: offset + index,
        },
        DecodeError::NonCanonical { index } => DecodeError::NonCanonical {
            index: offset + index,
        },
    }
}

fn adjust_slice_error(
    error: DecodeSliceError,
    offset: usize,
    total_input_len: usize,
) -> DecodeSliceError {
    match error {
        DecodeSliceError::Decode(error) => {
            DecodeSliceError::Decode(adjust_decode_error(error, offset, total_input_len))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use crate::Base64Unpadded;
    use crate::DecodeSliceError;
    use crate::Decoder;

    #[test]
    fn cumulative_length_overflow_leaves_decoder_usable() {
        // Represent a stream ending on a quantum boundary without processing
        // usize::MAX bytes merely to reach the counter limit.
        let mut decoder = Decoder::<Base64Unpadded> {
            total_input_len: usize::MAX - 3,
            ..Decoder::new()
        };
        let mut output = [0_u8; 3];
        assert_eq!(
            decoder.update("Zm9v", &mut output),
            Err(DecodeSliceError::LengthOverflow)
        );
        assert_eq!(decoder.update("Zg", &mut []).unwrap(), b"");
        assert_eq!(decoder.finish(&mut output).unwrap(), b"f");
    }
}
