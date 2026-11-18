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

use std::io;
use std::io::Read;
use std::io::Write;

use bases::Base32;
use bases::Base64;
use bases::DecodeError;
use bases::io::EncoderWriter;

#[test]
fn encoder_writer_handles_drop_and_into_inner() {
    // Inherent constructors work without Codec in scope.
    let mut dropped_out = Vec::new();
    {
        let mut writer = Base64::encoder_writer(&mut dropped_out);
        writer.write_all(b"hi").unwrap();
    }
    assert_eq!(dropped_out, b"aGk=");

    let mut dropped_empty = Vec::new();
    {
        let _writer = EncoderWriter::<Base32, _>::new(&mut dropped_empty);
    }
    assert!(dropped_empty.is_empty());

    let mut writer = Base64::encoder_writer(Vec::new());
    writer.write_all(b"f").unwrap();
    assert!(writer.get_ref().is_empty());
    let _ = writer.get_mut();
    let raw = writer.into_inner();
    assert!(raw.is_empty());
}

#[test]
fn decoder_reader_repeats_the_finalization_error_at_eof() {
    let mut reader = Base64::decoder_reader(&b"Zg"[..]);
    let mut buf = [0; 3];
    for _ in 0..2 {
        let err = reader.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            err.get_ref().and_then(|e| e.downcast_ref::<DecodeError>()),
            Some(&DecodeError::InvalidLength { length: 2 })
        );
    }
}

struct ScriptedWriter {
    writes: std::vec::IntoIter<io::Result<usize>>,
    bytes: Vec<u8>,
}

impl Write for ScriptedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let count = self.writes.next().unwrap_or(Ok(buf.len()))?.min(buf.len());
        self.bytes.extend_from_slice(&buf[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn writer_retries_a_failed_quantum_without_consuming_pending_input() {
    for failure in [Ok(0), Err(io::Error::from(io::ErrorKind::WouldBlock))] {
        let inner = ScriptedWriter {
            writes: vec![failure].into_iter(),
            bytes: Vec::new(),
        };
        let mut writer = Base64::encoder_writer(inner);
        assert_eq!(writer.write(b"f").unwrap(), 1);
        match writer.write(b"oo") {
            Ok(count) => assert_eq!(count, 0),
            Err(error) => assert_eq!(error.kind(), io::ErrorKind::WouldBlock),
        }
        assert_eq!(writer.write(b"oo").unwrap(), 2);
        assert_eq!(writer.finish().unwrap().bytes, b"Zm9v");
    }
}

#[test]
fn writer_retries_buffered_output_without_duplicating_accepted_input() {
    let inner = ScriptedWriter {
        writes: vec![Ok(2), Err(io::Error::from(io::ErrorKind::WouldBlock))].into_iter(),
        bytes: Vec::new(),
    };
    let mut writer = Base64::encoder_writer(inner);
    assert_eq!(writer.write(b"foo").unwrap(), 3);
    assert_eq!(
        writer.write(b"bar").unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    assert_eq!(writer.write(b"bar").unwrap(), 3);
    assert_eq!(writer.finish().unwrap().bytes, b"Zm9vYmFy");
}
