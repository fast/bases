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

use std::fmt;
use std::fmt::Write as _;
use std::hint::black_box;
use std::io;
use std::io::Read;
use std::io::Write as _;
use std::str;

use base32ct::Encoding as _;
use base64::Engine as _;
use base64::display::Base64Display;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use base64::engine::general_purpose::NO_PAD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use base64::engine::general_purpose::PAD;
use base64::engine::general_purpose::STANDARD;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use base64::engine::simd::Simd;
use base64::read::DecoderReader;
use base64::write::EncoderWriter;
use base64_simd::AsOut;
use bases::Base16Lower;
use bases::Base32;
use bases::Base64;
use bases::Base64UrlUnpadded;
use bases::Codec;
use bases::HexDisplay;
use data_encoding::Encoding;

use crate::buffers::buffer;
use crate::harness::Direction;
use crate::harness::Fixture;
use crate::harness::Peer;
use crate::harness::Runner;

pub fn register(runner: &mut Runner) {
    let standard = data_encoding::BASE64;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let simd = Simd::standard(PAD);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let simd_url = Simd::url_safe(NO_PAD);
    let lowercase = data_encoding::HEXLOWER_PERMISSIVE;
    owned_cases::<Base16Lower>(
        runner,
        "base16",
        &lowercase,
        (32, 20_300_000, 20_500_000),
        &[
            (
                "hex",
                &owned(
                    |bytes| hex::encode(bytes),
                    |text| hex::decode(text).unwrap(),
                ),
            ),
            (
                "const-hex",
                &owned(
                    |bytes| const_hex::encode(bytes),
                    |text| const_hex::decode(text).unwrap(),
                ),
            ),
            (
                "base16ct",
                &owned(base16ct::lower::encode_string, |text| {
                    base16ct::mixed::decode_vec(text).unwrap()
                }),
            ),
        ],
    );
    owned_cases::<Base32>(
        runner,
        "base32",
        &data_encoding::BASE32,
        (1024, 3_150_000, 2_100_000),
        &[(
            "base32ct",
            &owned(base32ct::Base32Upper::encode_string, |text| {
                base32ct::Base32Upper::decode_vec(text).unwrap()
            }),
        )],
    );
    owned_cases::<Base64>(
        runner,
        "base64",
        &standard,
        (1024, 3_300_000, 3_000_000),
        &[
            (
                "base64",
                &owned(
                    |bytes| STANDARD.encode(bytes),
                    |text| STANDARD.decode(text).unwrap(),
                ),
            ),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            (
                "base64::Simd",
                &owned(
                    |bytes| simd.encode(bytes),
                    |text| simd.decode(text).unwrap(),
                ),
            ),
            (
                "base64-simd",
                &owned(
                    |bytes| base64_simd::STANDARD.encode_to_string(bytes),
                    |text| base64_simd::STANDARD.decode_to_vec(text).unwrap(),
                ),
            ),
        ],
    );
    owned_cases::<Base64UrlUnpadded>(
        runner,
        Base64UrlUnpadded::name(),
        &data_encoding::BASE64URL_NOPAD,
        (8, 17_500_000, 22_000_000),
        &[
            (
                "base64",
                &owned(
                    |bytes| URL_SAFE_NO_PAD.encode(bytes),
                    |text| URL_SAFE_NO_PAD.decode(text).unwrap(),
                ),
            ),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            (
                "base64::Simd",
                &owned(
                    |bytes| simd_url.encode(bytes),
                    |text| simd_url.decode(text).unwrap(),
                ),
            ),
            (
                "base64-simd",
                &owned(
                    |bytes| base64_simd::URL_SAFE_NO_PAD.encode_to_string(bytes),
                    |text| base64_simd::URL_SAFE_NO_PAD.decode_to_vec(text).unwrap(),
                ),
            ),
        ],
    );

    runner.case(
        "base16/display/32",
        19_700_000,
        32,
        &lowercase,
        Direction::Encode,
        &[
            (
                "bases",
                &formatted(|out, bytes| write!(out, "{:x}", HexDisplay(bytes))),
            ),
            (
                "data-encoding",
                &formatted(|out, bytes| write!(out, "{}", lowercase.encode_display(bytes))),
            ),
            (
                "base16ct",
                &formatted(|out, bytes| write!(out, "{:x}", base16ct::HexDisplay(bytes))),
            ),
        ],
    );
    runner.case(
        "base64/display/1024",
        2_600_000,
        1024,
        &standard,
        Direction::Encode,
        &[
            (
                "bases",
                &formatted(|out, bytes| write!(out, "{}", Base64::display(bytes))),
            ),
            (
                "data-encoding",
                &formatted(|out, bytes| write!(out, "{}", standard.encode_display(bytes))),
            ),
            (
                "base64",
                &formatted(|out, bytes| write!(out, "{}", Base64Display::new(bytes, &STANDARD))),
            ),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            (
                "base64::Simd",
                &formatted(|out, bytes| write!(out, "{}", Base64Display::new(bytes, &simd))),
            ),
        ],
    );

    let ours = buffer(stream_encode, stream_decode);
    let peer = buffer(
        |input, output| writer_encode(input, output, &STANDARD),
        |input, output| reader_decode(input, output, &STANDARD),
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let simd_peer = buffer(
        |input, output| writer_encode(input, output, &simd),
        |input, output| reader_decode(input, output, &simd),
    );
    let peers: &[Peer<'_>] = &[
        ("bases", &ours),
        ("base64", &peer),
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        ("base64::Simd", &simd_peer),
    ];
    runner.case(
        "base64/stream/encode/16384",
        180_000,
        16_384,
        &standard,
        Direction::Encode,
        peers,
    );
    runner.case(
        "base64/stream/decode/16384",
        145_000,
        16_384,
        &standard,
        Direction::Decode,
        peers,
    );

    for (at_end, side, rounds) in [(false, "first", 30_000_000), (true, "last", 3_600_000)] {
        runner.case(
            &format!("base64/reject/{side}/1024"),
            rounds,
            1024,
            &standard,
            Direction::Decode,
            &[
                (
                    "bases",
                    &rejected(at_end, |input, output| {
                        Base64::decode_into(input, output).is_err()
                    }),
                ),
                (
                    "data-encoding",
                    &rejected(at_end, |input, output| {
                        standard.decode_mut(input, output).is_err()
                    }),
                ),
                (
                    "base64",
                    &rejected(at_end, |input, output| {
                        STANDARD.decode_slice(input, output).is_err()
                    }),
                ),
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                (
                    "base64::Simd",
                    &rejected(at_end, |input, output| {
                        simd.decode_slice(input, output).is_err()
                    }),
                ),
                (
                    "base64-simd",
                    &rejected(at_end, |input, output| {
                        base64_simd::STANDARD
                            .decode(input, output.as_out())
                            .is_err()
                    }),
                ),
            ],
        );
    }

    for (size, rounds) in [(32, 90_000_000), (16_384, 380_000)] {
        runner.case(
            &format!("base16/validate/{size}"),
            rounds,
            size,
            &lowercase,
            Direction::Decode,
            &[
                (
                    "bases",
                    &validated(|input| Base16Lower::validate(input).is_ok()),
                ),
                ("faster-hex", &validated(faster_hex::hex_check)),
                (
                    "const-hex",
                    &validated(|input| const_hex::check(input).is_ok()),
                ),
            ],
        );
    }
    for (size, rounds) in [(32, 60_000_000), (16_384, 400_000)] {
        runner.case(
            &format!("base32/validate/{size}"),
            rounds,
            size,
            &data_encoding::BASE32,
            Direction::Decode,
            &[("bases", &validated(|input| Base32::validate(input).is_ok()))],
        );
    }
    for (size, rounds) in [(32, 75_000_000), (16_384, 420_000)] {
        runner.case(
            &format!("base64/validate/{size}"),
            rounds,
            size,
            &standard,
            Direction::Decode,
            &[
                ("bases", &validated(|input| Base64::validate(input).is_ok())),
                (
                    "base64-simd",
                    &validated(|input| base64_simd::STANDARD.check(input).is_ok()),
                ),
            ],
        );
    }
}

fn owned_cases<C: Codec>(
    runner: &mut Runner,
    group: &str,
    reference: &Encoding,
    (size, encode_rounds, decode_rounds): (usize, usize, usize),
    extra: &[Peer<'_>],
) {
    let ours = owned(C::encode, |text| C::decode(text).unwrap());
    let data = owned(
        |bytes| reference.encode(bytes),
        |text| reference.decode(text.as_bytes()).unwrap(),
    );
    let mut peers: Vec<Peer<'_>> = vec![("bases", &ours), ("data-encoding", &data)];
    peers.extend_from_slice(extra);
    runner.case(
        &format!("{group}/owned/encode/{size}"),
        encode_rounds,
        size,
        reference,
        Direction::Encode,
        &peers,
    );
    runner.case(
        &format!("{group}/owned/decode/{size}"),
        decode_rounds,
        size,
        reference,
        Direction::Decode,
        &peers,
    );
}

fn owned(
    encode: impl Fn(&[u8]) -> String,
    decode: impl Fn(&str) -> Vec<u8>,
) -> impl Fn(&Fixture, usize) {
    move |fixture, rounds| match fixture.direction {
        Direction::Encode => allocations(&fixture.input[..], &fixture.expected, rounds, &encode),
        Direction::Decode => {
            let text = str::from_utf8(&fixture.input).unwrap();
            allocations(text, &fixture.expected, rounds, &decode);
        }
    }
}

fn allocations<I: ?Sized, T: AsRef<[u8]>>(
    input: &I,
    expected: &[u8],
    rounds: usize,
    convert: impl Fn(&I) -> T,
) {
    for _ in 1..rounds {
        let output = convert(black_box(input));
        black_box(output.as_ref());
    }
    let output = convert(black_box(input));
    assert_eq!(black_box(output.as_ref()), expected);
}

fn formatted(write: impl Fn(&mut String, &[u8]) -> fmt::Result) -> impl Fn(&Fixture, usize) {
    move |fixture, rounds| {
        let mut output = String::with_capacity(fixture.capacity);
        for _ in 0..rounds {
            output.clear();
            write(&mut output, black_box(&fixture.input)).unwrap();
            black_box(output.as_bytes());
        }
        assert_eq!(output.as_bytes(), fixture.expected);
    }
}

// An odd source chunk crosses both raw and encoded Base64 quantum boundaries.
const CHUNK: usize = 257;

fn stream_encode(input: &[u8], output: &mut [u8]) -> usize {
    let mut encoder = Base64::encoder();
    let mut written = 0;
    for chunk in input.chunks(CHUNK) {
        written += encoder.update(chunk, &mut output[written..]).unwrap().len();
    }
    written + encoder.finish(&mut output[written..]).unwrap().len()
}

fn writer_encode(input: &[u8], output: &mut [u8], engine: &impl base64::Engine) -> usize {
    let capacity = output.len();
    let mut writer = EncoderWriter::new(output, engine);
    for chunk in input.chunks(CHUNK) {
        writer.write_all(chunk).unwrap();
    }
    capacity - writer.finish().unwrap().len()
}

fn stream_decode(input: &[u8], output: &mut [u8]) -> usize {
    let mut decoder = Base64::decoder();
    let mut written = 0;
    for chunk in input.chunks(CHUNK) {
        written += decoder.update(chunk, &mut output[written..]).unwrap().len();
    }
    written + decoder.finish(&mut output[written..]).unwrap().len()
}

struct Chunked<'a>(&'a [u8]);

impl Read for Chunked<'_> {
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        let count = dst.len().min(CHUNK);
        self.0.read(&mut dst[..count])
    }
}

fn reader_decode(input: &[u8], mut output: &mut [u8], engine: &impl base64::Engine) -> usize {
    let mut reader = DecoderReader::new(Chunked(input), engine);
    io::copy(&mut reader, &mut output).unwrap() as usize
}

fn rejected(at_end: bool, reject: impl Fn(&[u8], &mut [u8]) -> bool) -> impl Fn(&Fixture, usize) {
    move |fixture, rounds| {
        let mut input = fixture.input.clone();
        let mut output = vec![0; fixture.capacity];
        assert!(!reject(&input, &mut output));
        let index = if at_end {
            input.iter().rposition(|&byte| byte != b'=').unwrap()
        } else {
            0
        };
        input[index] = b'!';
        let mut failed = false;
        for _ in 0..rounds {
            failed = black_box(reject(black_box(&input), black_box(&mut output)));
        }
        assert!(failed);
    }
}

fn validated(validate: impl Fn(&[u8]) -> bool) -> impl Fn(&Fixture, usize) {
    move |fixture, rounds| {
        let mut ok = false;
        for _ in 0..rounds {
            ok = black_box(validate(black_box(&fixture.input)));
        }
        assert!(ok);
    }
}
