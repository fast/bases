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

//! Standard `std::io::{Read, Write}` streaming adapters.

use std::fmt;
use std::io;
use std::io::Read;
use std::io::Write;

use crate::Codec;
use crate::DecodeSliceError;
use crate::Decoder;
use crate::Encoder;
use crate::codec::sealed::Profile;

const BUF_LEN: usize = 1024;
const MAX_DECODED_QUANTUM: usize = 5;

/// A [`Write`] adapter that encodes written bytes with profile `C`.
///
/// Call [`finish`](Self::finish) to emit any buffered incomplete quantum and
/// padding while surfacing I/O errors. Dropping the writer without calling
/// `finish` performs a best-effort finalization and ignores errors.
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// use bases::Base64;
/// use bases::io::EncoderWriter;
///
/// let mut encoder = EncoderWriter::<Base64, _>::new(Vec::new());
/// encoder.write_all(b"hello").unwrap();
/// let encoded = encoder.finish().unwrap();
/// assert_eq!(encoded, b"aGVsbG8=");
/// ```
pub struct EncoderWriter<C: Codec, W: Write> {
    encoder: Encoder<C>,
    writer: Option<W>,
    output: [u8; BUF_LEN],
    output_pos: usize,
    output_len: usize,
    finished: bool,
    panicked: bool,
}

impl<C: Codec, W: Write + fmt::Debug> fmt::Debug for EncoderWriter<C, W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncoderWriter")
            .field("writer", &self.writer)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<C: Codec, W: Write> EncoderWriter<C, W> {
    /// Creates a new encoding writer wrapping `writer`.
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self {
            encoder: Encoder::new(),
            writer: Some(writer),
            output: [0; BUF_LEN],
            output_pos: 0,
            output_len: 0,
            finished: false,
            panicked: false,
        }
    }

    /// Borrows the underlying writer.
    #[must_use]
    pub fn get_ref(&self) -> &W {
        self.writer
            .as_ref()
            .expect("encoder writer is present before consumption")
    }

    /// Mutably borrows the underlying writer.
    ///
    /// Writing directly to the inner writer may corrupt the encoded stream.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut W {
        self.writer
            .as_mut()
            .expect("encoder writer is present before consumption")
    }

    /// Unwraps the writer without finishing pending input or flushing buffered
    /// encoded output.
    ///
    /// Prefer [`finish`](Self::finish) to emit any remaining symbols and
    /// padding.
    #[must_use]
    pub fn into_inner(mut self) -> W {
        self.finished = true;
        self.writer
            .take()
            .expect("encoder writer is present before consumption")
    }

    /// Encodes any remaining buffered input, writes all buffered output, and
    /// returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns any I/O error emitted by the underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        let result = self.finish_inner();
        self.finished = true;
        result?;
        Ok(self
            .writer
            .take()
            .expect("encoder writer is present before consumption"))
    }

    fn finish_inner(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }

        self.flush_encoded()?;
        if self.encoder.finish_output_len() > 0 {
            let encoded_len = self
                .encoder
                .clone()
                .finish(&mut self.output)
                .expect("encoder scratch buffer must fit one final quantum")
                .len();
            self.output_pos = 0;
            self.output_len = encoded_len;
            self.encoder = Encoder::new();
            self.flush_encoded()?;
        }

        self.finished = true;
        Ok(())
    }

    fn flush_encoded(&mut self) -> io::Result<()> {
        while self.output_pos < self.output_len {
            let result = {
                let writer = self
                    .writer
                    .as_mut()
                    .expect("encoder writer is present before consumption");
                self.panicked = true;
                let res = writer.write(&self.output[self.output_pos..self.output_len]);
                self.panicked = false;
                res
            };

            match result {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole encoded buffer",
                    ));
                }
                Ok(written) => {
                    self.output_pos += written;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }

        self.output_pos = 0;
        self.output_len = 0;
        Ok(())
    }
}

impl<C: Codec, W: Write> Write for EncoderWriter<C, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        self.flush_encoded()?;

        let saved_encoder = self.encoder.clone();
        let max_quanta = BUF_LEN / C::Profile::ENCODED_QUANTUM;
        let max_input = max_quanta * C::Profile::INPUT_QUANTUM - self.encoder.pending_len();
        let chunk_len = buf.len().min(max_input);
        let encoded_len = self
            .encoder
            .update(&buf[..chunk_len], &mut self.output)
            .expect("encoder scratch buffer must fit one input chunk")
            .len();

        if encoded_len == 0 {
            return Ok(chunk_len);
        }

        loop {
            let result = {
                let writer = self
                    .writer
                    .as_mut()
                    .expect("encoder writer is present before consumption");
                self.panicked = true;
                let res = writer.write(&self.output[..encoded_len]);
                self.panicked = false;
                res
            };

            match result {
                Ok(0) => {
                    self.encoder = saved_encoder;
                    return Ok(0);
                }
                Ok(written) => {
                    if written < encoded_len {
                        self.output_pos = written;
                        self.output_len = encoded_len;
                    }
                    return Ok(chunk_len);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    self.encoder = saved_encoder;
                    return Err(error);
                }
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_encoded()?;
        let writer = self
            .writer
            .as_mut()
            .expect("encoder writer is present before consumption");
        self.panicked = true;
        let res = writer.flush();
        self.panicked = false;
        res
    }
}

impl<C: Codec, W: Write> Drop for EncoderWriter<C, W> {
    fn drop(&mut self) {
        if !self.finished && !self.panicked {
            let _ = self.finish_inner();
        }
    }
}

/// A [`Read`] adapter that decodes input bytes read from an underlying reader
/// using profile `C`.
///
/// # Examples
///
/// ```
/// use std::io::Read;
///
/// use bases::Base64;
/// use bases::io::DecoderReader;
///
/// let mut decoder = DecoderReader::<Base64, _>::new(&b"aGVsbG8="[..]);
/// let mut decoded = Vec::new();
/// decoder.read_to_end(&mut decoded).unwrap();
/// assert_eq!(decoded, b"hello");
/// ```
pub struct DecoderReader<C: Codec, R: Read> {
    reader: R,
    decoder: Option<Decoder<C>>,
    encoded: [u8; BUF_LEN],
    encoded_pos: usize,
    encoded_len: usize,
    decoded: [u8; MAX_DECODED_QUANTUM],
    decoded_pos: usize,
    decoded_len: usize,
    eof: bool,
}

impl<C: Codec, R: Read + fmt::Debug> fmt::Debug for DecoderReader<C, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecoderReader")
            .field("reader", &self.reader)
            .field("eof", &self.eof)
            .finish_non_exhaustive()
    }
}

impl<C: Codec, R: Read> DecoderReader<C, R> {
    /// Creates a new decoding reader wrapping `reader`.
    #[must_use]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            decoder: Some(Decoder::new()),
            encoded: [0; BUF_LEN],
            encoded_pos: 0,
            encoded_len: 0,
            decoded: [0; MAX_DECODED_QUANTUM],
            decoded_pos: 0,
            decoded_len: 0,
            eof: false,
        }
    }

    /// Borrows the underlying reader.
    #[must_use]
    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Mutably borrows the underlying reader.
    ///
    /// Reading directly from the inner reader may corrupt the decoded stream.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Unwraps the reader, discarding any buffered encoded or decoded bytes.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<C: Codec, R: Read> Read for DecoderReader<C, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        loop {
            if self.decoded_pos < self.decoded_len {
                let available = self.decoded_len - self.decoded_pos;
                let to_copy = available.min(buf.len());
                buf[..to_copy]
                    .copy_from_slice(&self.decoded[self.decoded_pos..self.decoded_pos + to_copy]);
                self.decoded_pos += to_copy;
                return Ok(to_copy);
            }

            if self.decoder.is_none() {
                return Ok(0);
            }

            if self.encoded_pos == self.encoded_len {
                if self.eof {
                    let decoder = self.decoder.take().expect("checked above");
                    match decoder.finish(&mut self.decoded) {
                        Ok(final_bytes) => {
                            self.decoded_pos = 0;
                            self.decoded_len = final_bytes.len();
                            if self.decoded_len == 0 {
                                return Ok(0);
                            }
                            continue;
                        }
                        Err(finish_error) => {
                            let (error, decoder) = finish_error.into_parts();
                            self.decoder = Some(decoder);
                            return Err(decode_error_to_io(error));
                        }
                    }
                }

                match self.reader.read(&mut self.encoded) {
                    Ok(0) => {
                        self.eof = true;
                        continue;
                    }
                    Ok(read) => {
                        self.encoded_pos = 0;
                        self.encoded_len = read;
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                }
            }

            let decoder = self.decoder.as_mut().expect("checked above");
            let input_quantum = C::Profile::INPUT_QUANTUM;
            let encoded_quantum = C::Profile::ENCODED_QUANTUM;
            let pending_len = decoder.pending_len();
            let available_encoded = self.encoded_len - self.encoded_pos;

            if buf.len() < input_quantum {
                let needed = encoded_quantum - pending_len;
                let take = available_encoded.min(needed);
                let written = decoder
                    .update(
                        &self.encoded[self.encoded_pos..self.encoded_pos + take],
                        &mut self.decoded,
                    )
                    .map_err(decode_error_to_io)?
                    .len();
                self.encoded_pos += take;
                if written > 0 {
                    self.decoded_pos = 0;
                    self.decoded_len = written;
                }
            } else {
                let max_quanta = buf.len() / input_quantum;
                let max_encoded = max_quanta
                    .saturating_mul(encoded_quantum)
                    .saturating_sub(pending_len);
                let take = available_encoded.min(max_encoded);
                let written = decoder
                    .update(
                        &self.encoded[self.encoded_pos..self.encoded_pos + take],
                        buf,
                    )
                    .map_err(decode_error_to_io)?
                    .len();
                self.encoded_pos += take;
                if written > 0 {
                    return Ok(written);
                }
            }
        }
    }
}

fn decode_error_to_io(error: DecodeSliceError) -> io::Error {
    match error {
        DecodeSliceError::Decode(error) => io::Error::new(io::ErrorKind::InvalidData, error),
        other => io::Error::new(io::ErrorKind::InvalidData, other),
    }
}
