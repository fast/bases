# Bases

[![Crates.io][crates-badge]][crates-url]
[![Documentation][docs-badge]][docs-url]
[![MSRV 1.88][msrv-badge]](https://www.whatrustisit.com)
[![Apache 2.0 licensed][license-badge]][license-url]
[![Build Status][actions-badge]][actions-url]

[crates-badge]: https://img.shields.io/crates/v/bases.svg
[crates-url]: https://crates.io/crates/bases
[docs-badge]: https://img.shields.io/docsrs/bases
[docs-url]: https://docs.rs/bases
[msrv-badge]: https://img.shields.io/badge/MSRV-1.88-green?logo=rust
[license-badge]: https://img.shields.io/badge/license-Apache--2.0-blue
[license-url]: https://www.apache.org/licenses/LICENSE-2.0
[actions-badge]: https://github.com/fast/bases/actions/workflows/ci.yml/badge.svg
[actions-url]: https://github.com/fast/bases/actions/workflows/ci.yml

Fast, strict RFC 4648 Base16, Base32, and Base64 codecs for Rust.

Bases supports allocation-free caller buffers, owned results, incremental streaming, formatting, and Serde integration. It works in `no_std` without dependencies by default.

```rust
use bases::{Base64, Base64UrlUnpadded};

let mut buffer = [0_u8; 16];
let encoded = Base64::encode_into(b"hello", &mut buffer).unwrap();
assert_eq!(encoded, "aGVsbG8=");

let encoded = Base64UrlUnpadded::encode_into(b"hello", &mut buffer).unwrap();
assert_eq!(encoded, "aGVsbG8");
```

## Documentation

Add `bases = "0.1"` to your dependencies. The [usage guide](bases/README.md) covers profiles, feature flags, streaming, and error handling.

| Task                          | Read                                      |
| ----------------------------- | ----------------------------------------- |
| Choose a profile or API       | [Usage guide](bases/README.md)            |
| Look up public contracts      | [API reference](https://docs.rs/bases)    |
| Run application examples      | [Examples](examples/README.md)            |
| Understand the implementation | [Architecture](bases/ARCHITECTURE.md)     |
| Measure a workload            | [Benchmarks](benchmarks/README.md)        |
| Review release changes        | [Changelog](CHANGELOG.md)                 |

## Development

The workspace separates the library, examples, integration tests, benchmarks, and repository tooling. Run checks from the repository root:

```shell
cargo x check
cargo x test
cargo x lint
```

Use `cargo x bench` for performance comparisons and see the [benchmark guide](benchmarks/README.md) for workloads and measurement limits. See [Releasing Bases](RELEASE.md) for release validation and publication.

## Supported Rust Versions (MSRV 1.88.0)

Bases is built against the latest stable release. The minimum supported version is 1.88.0. The current Bases version is not guaranteed to build on Rust versions earlier than the minimum supported version.

## License

This project is licensed under [Apache License, Version 2.0](LICENSE).
