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

//! Convert a file, socket, or other byte stream using bounded buffers.
//!
//! With no arguments, run an in-memory round trip across several read boundaries. With `encode`
//! or `decode`, convert standard input to standard output using padded Base64. The helpers are
//! generic over the profile and also support unpadded streams.

use std::env;
use std::error::Error;
use std::io;
use std::io::Read;
use std::io::Write;

use bases::Base64;
use bases::Base64UrlUnpadded;
use bases::Codec;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mode = args.next();
    if args.next().is_some() {
        return Err("usage: stream_io [encode|decode]".into());
    }

    match mode.as_deref() {
        Some("encode") => encode::<Base64>(io::stdin().lock(), io::stdout().lock()),
        Some("decode") => decode::<Base64>(io::stdin().lock(), io::stdout().lock()),
        Some(_) => Err("usage: stream_io [encode|decode]".into()),
        None => {
            // Cross several read boundaries and leave a partial encoding quantum.
            let input: Vec<u8> = (0..8_201).map(|index| (index % 256) as u8).collect();
            roundtrip::<Base64>(&input)?;
            roundtrip::<Base64UrlUnpadded>(&input)
        }
    }
}

fn encode<C: Codec>(
    mut source: impl Read,
    mut destination: impl Write,
) -> Result<(), Box<dyn Error>> {
    let mut encoder = C::encoder();
    let mut input = [0_u8; 4096];
    let mut output = Vec::new();

    loop {
        let read = match source.read(&mut input) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };

        // Stream sizing accounts for bytes retained from the preceding read.
        let required = encoder.update_output_len(read)?;
        if output.len() < required {
            output.resize(required, 0);
        }
        let encoded = encoder.update(&input[..read], &mut output)?;
        destination.write_all(encoded.as_bytes())?;
    }

    let required = encoder.finish_output_len();
    if output.len() < required {
        output.resize(required, 0);
    }
    destination.write_all(encoder.finish(&mut output)?.as_bytes())?;
    destination.flush()?;
    Ok(())
}

fn decode<C: Codec>(
    mut source: impl Read,
    mut destination: impl Write,
) -> Result<(), Box<dyn Error>> {
    let mut decoder = C::decoder();
    let mut input = [0_u8; 4096];
    let mut output = Vec::new();

    loop {
        let read = match source.read(&mut input) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };

        let available = decoder.max_update_output_len(read);
        if output.len() < available {
            output.resize(available, 0);
        }
        let decoded = decoder.update(&input[..read], &mut output)?;
        // A later read or finish can still reject the message after a prefix has been written.
        destination.write_all(decoded)?;
    }

    // Even with no buffered output, finish is required to validate the final input fragment.
    let required = decoder.finish_output_len()?;
    if output.len() < required {
        output.resize(required, 0);
    }
    destination.write_all(decoder.finish(&mut output)?)?;
    destination.flush()?;
    Ok(())
}

fn roundtrip<C: Codec>(input: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut encoded = Vec::new();
    encode::<C>(input, &mut encoded)?;
    assert_eq!(encoded, C::encode(input).as_bytes());

    let mut decoded = Vec::new();
    decode::<C>(encoded.as_slice(), &mut decoded)?;
    assert_eq!(decoded, input);
    println!(
        "{}: {} bytes -> {} characters -> original bytes",
        C::name(),
        input.len(),
        encoded.len()
    );

    Ok(())
}
