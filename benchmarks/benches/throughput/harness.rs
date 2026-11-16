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

use std::num::NonZeroUsize;
use std::process;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use data_encoding::Encoding;

#[derive(Clone, Copy)]
pub enum Direction {
    Encode,
    Decode,
}

pub struct Fixture {
    pub input: Vec<u8>,
    pub expected: Vec<u8>,
    pub capacity: usize,
    pub direction: Direction,
}

impl Fixture {
    fn new(size: usize, reference: &Encoding, direction: Direction) -> Self {
        let mut state = 0x243f_6a88_u32;
        let raw: Vec<_> = (0..size)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        let encoded = reference.encode(&raw).into_bytes();
        let (input, expected, capacity) = match direction {
            Direction::Encode => {
                let capacity = encoded.len();
                (raw, encoded, capacity)
            }
            Direction::Decode => {
                let capacity = reference.decode_len(encoded.len()).unwrap();
                (encoded, raw, capacity)
            }
        };
        Self {
            input,
            expected,
            capacity,
            direction,
        }
    }
}

pub type Peer<'a> = (&'a str, &'a dyn Fn(&Fixture, usize));

#[derive(Parser)]
#[command(about = "Compare codec implementations on fixed workloads.")]
struct Options {
    /// Select scenarios whose names contain all the given substrings.
    filters: Vec<String>,
    /// List matching scenarios, their fixed rounds and peers.
    #[arg(long, conflicts_with = "test")]
    list: bool,
    /// Run each selected implementation once and check its output without timing.
    #[arg(long)]
    test: bool,
    /// Override the round count equally for all implementations.
    #[arg(long)]
    rounds: Option<NonZeroUsize>,
    /// Number of timing samples per implementation.
    #[arg(long, default_value = "5")]
    repeat: NonZeroUsize,
    /// Run only bases, for instance when sizing a new workload.
    #[arg(long)]
    baseline_only: bool,
    /// Print every sample in measurement order.
    #[arg(long)]
    raw: bool,
    // Cargo passes this to harness-free benchmarks.
    #[arg(long = "bench", hide = true)]
    _bench: bool,
}

pub struct Runner {
    options: Options,
    matched: usize,
}

impl Runner {
    pub fn from_args() -> Self {
        Self {
            options: Options::parse(),
            matched: 0,
        }
    }

    pub fn finish(self) {
        if self.matched == 0 {
            eprintln!("No matching benchmark. Use --list to see the scenarios.");
            process::exit(1);
        }
    }

    pub fn case(
        &mut self,
        name: &str,
        rounds: usize,
        size: usize,
        reference: &Encoding,
        direction: Direction,
        peers: &[Peer<'_>],
    ) {
        let options = &self.options;
        if !options.filters.iter().all(|filter| name.contains(filter)) {
            return;
        }
        self.matched += 1;
        let peers = if options.baseline_only {
            &peers[..1]
        } else {
            peers
        };
        let rounds = options.rounds.map_or(rounds, NonZeroUsize::get);
        if options.list {
            let labels: Vec<_> = peers.iter().map(|p| p.0).collect();
            println!("{name}  {rounds} rounds  [{}]", labels.join(", "));
            return;
        }
        let run = |work: &dyn Fn(&Fixture, usize), n| {
            let fixture = Fixture::new(size, reference, direction);
            work(&fixture, n);
        };
        if options.test {
            for (label, work) in peers {
                run(work, 1);
                println!("ok  {name} / {label}");
            }
            return;
        }
        let mut samples = vec![Vec::with_capacity(options.repeat.get()); peers.len()];
        for repetition in 0..options.repeat.get() {
            for offset in 0..peers.len() {
                let index = (repetition + offset) % peers.len();
                let start = Instant::now();
                run(peers[index].1, rounds);
                samples[index].push(start.elapsed());
            }
        }
        println!("{name}  ({rounds} rounds)");
        let baseline = median(&samples[0]).as_secs_f64();
        for ((label, _), times) in peers.iter().zip(samples) {
            let middle = median(&times);
            let min = times.iter().min().unwrap().as_millis();
            let max = times.iter().max().unwrap().as_millis();
            print!(
                "  {label:15} {:4} ms  [{min}..{max} ms]",
                middle.as_millis()
            );
            if baseline > 0.0 {
                print!("  {:.2}x", middle.as_secs_f64() / baseline);
            }
            if options.raw {
                print!(
                    "  {:?}",
                    times.iter().map(Duration::as_millis).collect::<Vec<_>>()
                );
            }
            println!();
        }
    }
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let lower = sorted[(sorted.len() - 1) / 2];
    lower + (sorted[sorted.len() / 2] - lower) / 2
}
