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

use std::cell::Cell;
use std::io;
use std::io::Read;
use std::io::Write;
use std::panic;
use std::rc::Rc;

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

#[derive(Debug)]
struct ScriptedWriter {
    writes: std::vec::IntoIter<io::Result<usize>>,
    bytes: Vec<u8>,
    flushes: usize,
}

impl ScriptedWriter {
    fn new(writes: Vec<io::Result<usize>>) -> Self {
        Self {
            writes: writes.into_iter(),
            bytes: Vec::new(),
            flushes: 0,
        }
    }
}

impl Write for ScriptedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let count = self.writes.next().unwrap_or(Ok(buf.len()))?.min(buf.len());
        self.bytes.extend_from_slice(&buf[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flushes += 1;
        Ok(())
    }
}

#[test]
fn writer_retries_a_failed_quantum_without_consuming_pending_input() {
    for failure in [Ok(0), Err(io::Error::from(io::ErrorKind::WouldBlock))] {
        let inner = ScriptedWriter::new(vec![failure]);
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
    let inner = ScriptedWriter::new(vec![Ok(2), Err(io::Error::from(io::ErrorKind::WouldBlock))]);
    let mut writer = Base64::encoder_writer(inner);
    assert_eq!(writer.write(b"foo").unwrap(), 3);
    assert_eq!(
        writer.write(b"bar").unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    assert_eq!(writer.write(b"bar").unwrap(), 3);
    assert_eq!(writer.finish().unwrap().bytes, b"Zm9vYmFy");
}

#[test]
fn finish_retries_buffered_output_before_the_final_quantum() {
    let inner = ScriptedWriter::new(vec![Ok(1), Err(io::ErrorKind::WouldBlock.into())]);
    let mut writer = Base64::encoder_writer(inner);
    writer.write_all(b"foob").unwrap();

    let (error, writer) = writer.finish().unwrap_err().into_parts();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(writer.get_ref().bytes, b"Z");
    assert_eq!(writer.finish().unwrap().bytes, b"Zm9vYg==");
}

#[test]
fn finish_retries_a_partially_written_final_quantum() {
    let inner = ScriptedWriter::new(vec![
        Ok(4),
        Ok(1),
        Err(io::ErrorKind::WouldBlock.into()),
        Ok(0),
    ]);
    let mut writer = Base64::encoder_writer(inner);
    writer.write_all(b"foob").unwrap();

    let (error, mut writer) = writer.finish().unwrap_err().into_parts();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert_eq!(writer.get_ref().bytes, b"Zm9vY");
    assert_eq!(writer.write(&[]).unwrap(), 0);
    assert_eq!(
        writer.write(b"extra").unwrap_err().kind(),
        io::ErrorKind::BrokenPipe,
    );

    let (error, writer) = writer.finish().unwrap_err().into_parts();
    assert_eq!(error.kind(), io::ErrorKind::WriteZero);
    assert_eq!(writer.get_ref().bytes, b"Zm9vY");
    assert_eq!(writer.finish().unwrap().bytes, b"Zm9vYg==");
}

#[test]
fn dropping_finish_error_preserves_the_partially_written_output() {
    let mut inner =
        ScriptedWriter::new(vec![Ok(1), Err(io::Error::from(io::ErrorKind::WouldBlock))]);
    let mut writer = Base64::encoder_writer(&mut inner);
    writer.write_all(b"f").unwrap();

    drop(writer.finish().unwrap_err());

    assert_eq!(inner.bytes, b"Z");
    assert_eq!(inner.flushes, 0);
}

#[test]
fn finish_errors_propagate_as_io_errors_with_borrowed_writers() {
    fn finish(writer: EncoderWriter<Base64, &mut ScriptedWriter>) -> io::Result<()> {
        writer.finish()?;
        Ok(())
    }

    let mut inner = ScriptedWriter::new(vec![
        Ok(1),
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "sink refused output",
        )),
    ]);
    let mut writer = Base64::encoder_writer(&mut inner);
    writer.write_all(b"f").unwrap();
    let error = finish(writer).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert_eq!(error.to_string(), "sink refused output");
    assert_eq!(inner.bytes, b"Z");
}

#[test]
fn recovery_supports_non_send_writers_and_flush_does_not_reopen_input() {
    struct LocalWriter<'a> {
        inner: &'a mut ScriptedWriter,
        flushes: Rc<Cell<usize>>,
    }

    impl Write for LocalWriter<'_> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.inner.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes.set(self.flushes.get() + 1);
            self.inner.flush()
        }
    }

    let flushes = Rc::new(Cell::new(0));
    let mut inner =
        ScriptedWriter::new(vec![Ok(1), Err(io::Error::from(io::ErrorKind::WouldBlock))]);
    let mut writer = Base64::encoder_writer(LocalWriter {
        inner: &mut inner,
        flushes: Rc::clone(&flushes),
    });
    writer.write_all(b"f").unwrap();
    let Err(failure) = writer.finish() else {
        panic!("scripted finalization must fail");
    };
    let mut writer = failure.into_inner();
    writer.flush().unwrap();
    assert_eq!(flushes.get(), 1);
    assert_eq!(
        writer.write(b"oo").unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
    let inner = writer.finish().unwrap();
    assert_eq!(inner.inner.bytes, b"Zg==");
    assert_eq!(flushes.get(), 1);
}

#[test]
fn finish_does_not_retry_after_the_underlying_writer_panics() {
    struct PanickingWriter<'a>(&'a Cell<usize>);

    impl Write for PanickingWriter<'_> {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            self.0.set(self.0.get() + 1);
            panic!("writer panicked");
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let calls = Cell::new(0);
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let mut writer = Base64::encoder_writer(PanickingWriter(&calls));
        writer.write_all(b"f").unwrap();
        let _ = writer.finish();
    }));
    assert!(result.is_err());
    assert_eq!(calls.get(), 1);
}
