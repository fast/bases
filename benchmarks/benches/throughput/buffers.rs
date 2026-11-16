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

use std::hint::black_box;

use base32ct::Encoding as _;
use base64::Engine as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use base64::engine::general_purpose::PAD;
use base64::engine::general_purpose::STANDARD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use base64::engine::simd::Simd;
use base64_simd::AsOut;
use base64ct::Encoding as _;
use bases::Base16Lower;
use bases::Base32;
use bases::Base64;
use bases::Codec;
use data_encoding::Encoding;

use crate::harness::Direction;
use crate::harness::Fixture;
use crate::harness::Peer;
use crate::harness::Runner;

pub fn register(runner: &mut Runner) {
    let hex_peer = buffer(
        |input, output| {
            hex::encode_to_slice(input, output).unwrap();
            output.len()
        },
        |input, output| {
            hex::decode_to_slice(input, output).unwrap();
            output.len()
        },
    );
    let faster_hex_peer = buffer(
        |input, output| faster_hex::hex_encode(input, output).unwrap().len(),
        |input, output| {
            faster_hex::hex_decode(input, output).unwrap();
            output.len()
        },
    );
    let const_hex_peer = buffer(
        |input, output| const_hex::encode_to_str(input, output).unwrap().len(),
        |input, output| {
            const_hex::decode_to_slice(input, output).unwrap();
            output.len()
        },
    );
    let base16 = buffer(
        |input, output| base16ct::lower::encode_str(input, output).unwrap().len(),
        |input, output| base16ct::mixed::decode(input, output).unwrap().len(),
    );
    profile::<Base16Lower>(
        runner,
        "base16",
        &data_encoding::HEXLOWER_PERMISSIVE,
        &[
            (8, 30_000_000, 30_000_000),
            (32, 20_000_000, 20_000_000),
            (1024, 5_100_000, 4_000_000),
            (16_384, 300_000, 250_000),
        ],
        &[
            ("hex", &hex_peer),
            ("faster-hex", &faster_hex_peer),
            ("const-hex", &const_hex_peer),
            ("base16ct", &base16),
        ],
    );

    let base32 = buffer(
        |input, output| base32ct::Base32Upper::encode(input, output).unwrap().len(),
        |input, output| base32ct::Base32Upper::decode(input, output).unwrap().len(),
    );
    profile::<Base32>(
        runner,
        "base32",
        &data_encoding::BASE32,
        &[
            (8, 20_000_000, 20_000_000),
            (32, 15_000_000, 10_000_000),
            (1024, 4_300_000, 2_450_000),
            (16_384, 250_000, 150_000),
        ],
        &[("base32ct", &base32)],
    );

    let standard = buffer(
        |input, output| STANDARD.encode_slice(input, output).unwrap(),
        |input, output| STANDARD.decode_slice(input, output).unwrap(),
    );
    let constant_time = buffer(
        |input, output| base64ct::Base64::encode(input, output).unwrap().len(),
        |input, output| base64ct::Base64::decode(input, output).unwrap().len(),
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let simd_engine = Simd::standard(PAD);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let base64_simd_engine = buffer(
        |input, output| simd_engine.encode_slice(input, output).unwrap(),
        |input, output| simd_engine.decode_slice(input, output).unwrap(),
    );
    let simd = buffer(
        |input, output| {
            base64_simd::STANDARD
                .encode_as_str(input, output.as_out())
                .len()
        },
        |input, output| {
            base64_simd::STANDARD
                .decode(input, output.as_out())
                .unwrap()
                .len()
        },
    );
    profile::<Base64>(
        runner,
        "base64",
        &data_encoding::BASE64,
        &[
            (8, 20_000_000, 25_000_000),
            (32, 20_000_000, 25_000_000),
            (64, 26_000_000, 30_000_000),
            (1024, 4_750_000, 3_650_000),
            (16_384, 295_000, 225_000),
        ],
        &[
            ("base64", &standard),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            ("base64::Simd", &base64_simd_engine),
            ("base64ct", &constant_time),
            ("base64-simd", &simd),
        ],
    );
}

fn profile<C: Codec>(
    runner: &mut Runner,
    group: &str,
    reference: &Encoding,
    sizes: &[(usize, usize, usize)],
    extra: &[Peer<'_>],
) {
    let ours = buffer(
        |input, output| C::encode_into(input, output).unwrap().len(),
        |input, output| C::decode_into(input, output).unwrap().len(),
    );
    let other = buffer(
        |input, output| {
            reference.encode_mut(input, output);
            output.len()
        },
        |input, output| reference.decode_mut(input, output).unwrap(),
    );
    let mut peers: Vec<Peer<'_>> = vec![("bases", &ours), ("data-encoding", &other)];
    peers.extend_from_slice(extra);
    for &(size, encode_rounds, decode_rounds) in sizes {
        runner.case(
            &format!("{group}/encode/{size}"),
            encode_rounds,
            size,
            reference,
            Direction::Encode,
            &peers,
        );
        runner.case(
            &format!("{group}/decode/{size}"),
            decode_rounds,
            size,
            reference,
            Direction::Decode,
            &peers,
        );
    }
}

pub fn buffer(
    encode: impl Fn(&[u8], &mut [u8]) -> usize,
    decode: impl Fn(&[u8], &mut [u8]) -> usize,
) -> impl Fn(&Fixture, usize) {
    move |fixture, rounds| match fixture.direction {
        Direction::Encode => convert(fixture, rounds, &encode),
        Direction::Decode => convert(fixture, rounds, &decode),
    }
}

fn convert(fixture: &Fixture, rounds: usize, operation: impl Fn(&[u8], &mut [u8]) -> usize) {
    let mut output = vec![0; fixture.capacity];
    let mut written = 0;
    for _ in 0..rounds {
        written = operation(black_box(&fixture.input), black_box(&mut output));
        black_box(&output[..written]);
    }
    assert_eq!(&output[..written], fixture.expected);
}
