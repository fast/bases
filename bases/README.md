# Bases

Fast, strict RFC 4648 Base16, Base32, and Base64 codecs for Rust.

## Quick start

Add `bases` to your application:

```toml
[dependencies]
bases = "0.1"
```

The default configuration works in `no_std` without an allocator or dependencies. Encode into a caller-provided buffer, or decode into a fixed-size array when the decoded length is known:

```rust
use bases::{Base64, Base64UrlUnpadded};

let mut buffer = [0_u8; 8];
let encoded = Base64::encode_into(b"hello", &mut buffer).unwrap();
assert_eq!(encoded, "aGVsbG8=");

let decoded: [u8; 5] = Base64::decode_array(encoded).unwrap();
assert_eq!(&decoded, b"hello");

let cursor = Base64UrlUnpadded::encode_into(b"hello", &mut buffer).unwrap();
assert_eq!(cursor, "aGVsbG8");
```

## Choose a profile

Ten crate-root aliases cover standard encoding profiles:

- `Base16Lower` and `Base16Upper` emit lowercase and uppercase hex respectively, and both accept either case when decoding.
- `Base32` and `Base32Unpadded` use the uppercase RFC 4648 alphabet, with or without padding.
- `Base32Hex` and `Base32HexUnpadded` use the uppercase extended-hex alphabet, with or without padding.
- `Base64` and `Base64Unpadded` use the standard Base64 alphabet, with or without padding.
- `Base64Url` and `Base64UrlUnpadded` use the URL-safe alphabet, with or without padding.

Every profile exposes inherent methods without requiring trait imports, and implements the sealed `Codec` trait for generic callers. Configurable zero-sized family types (`base16::Base16`, `base32::Base32`, and `base64::Base64`) accept `policy` markers for other combinations, such as strict single-case Base16 or RFC 5155 Base32Hex with lowercase output and case-insensitive decoding. See the [API reference](https://docs.rs/bases) for types, methods, and their contracts.

For gRPC binary metadata, use `base64::Base64<base64::Standard, policy::OptionalPadding>`. It encodes `b"f"` as `Zg` and decodes both `Zg` and `Zg==`. Partial padding such as `Zg=` is invalid. The same marker works with Base64URL and both Base32 alphabets.

## API layers and features

- **Caller buffers (`no_std`):** `encoded_len`, `max_decoded_len`, `encode_into`, `decode_into`, `decode_in_place`, and `decode_array` operate without allocation. Concrete profiles provide `const fn` constructors and sizing methods.
- **Owned output (`alloc`):** `encode` and `decode` return `String` and `Vec<u8>`; `encode_append` and transactional `decode_append` extend existing buffers.
- **Streaming (`no_std` / `std`):** `Encoder<C>` and `Decoder<C>` process arbitrarily chunked input while retaining at most one partial quantum. The `std` feature adds `io::EncoderWriter` and `io::DecoderReader` for `std::io::Write` and `std::io::Read`.
- **Formatting (`no_std`):** `display` (`EncodedDisplay`) and `HexDisplay` (`{:x}` and `{:X}` only) write encoded text through `core::fmt` without an intermediate string.
- **Serde (`serde`):** `serialize` and `deserialize_array` integrate with `#[serde(with = "...")]`; `deserialize` returns a byte vector when `alloc` is also enabled.

| Feature   | Effect                                                                                             |
| --------- | -------------------------------------------------------------------------------------------------- |
| Default   | No features or dependencies; caller-buffer, in-place, fixed-array, streaming, and formatting APIs. |
| `alloc`   | Adds owned and append helpers for `String` and `Vec<u8>`.                                          |
| `std`     | Enables `alloc`, I/O adapters, and runtime x86/x86_64 CPU feature detection.                       |
| `serde`   | Adds `serde_core` serialization and fixed-array deserialization, including without an allocator.   |

## Uninitialized output buffers

Use `encode_into_uninit` or `decode_into_uninit` when the destination is an uninitialized stack buffer or a vector's spare capacity. Use the returned slice to access the result. For padded Base64, it can be shorter than the decoding size estimate.

```rust
use core::mem::MaybeUninit;

use bases::Base64;

const WIRE: &[u8] = b"Zg==";
let mut output = [MaybeUninit::uninit(); Base64::max_decoded_len(WIRE.len())];
let decoded = Base64::decode_into_uninit(WIRE, &mut output).unwrap();
assert_eq!(decoded, b"f");
```

When passing `Vec::spare_capacity_mut()`, the returned slice borrows the new bytes and the vector keeps its original length. To append decoded bytes to a `Vec<u8>` with automatic length management, use `decode_append` with the `alloc` feature. A decoding error may leave partial writes and reports no initialized prefix.

## Streaming and errors

```rust
use bases::Base64;

let mut encoder = Base64::encoder();
let mut output = [0_u8; 4];
assert_eq!(encoder.update(b"fo", &mut output).unwrap(), "");
assert_eq!(encoder.update(b"o", &mut output).unwrap(), "Zm9v");
assert_eq!(encoder.finish(&mut output).unwrap(), "");
```

Successful updates consume the complete input chunk. Successful `finish()` consumes the stream and handles the final fragment. On failure, `FinishError::into_inner()` recovers the unchanged encoder or decoder for retrying with a larger output buffer. Callers that do not need recovery can use `?` to convert `FinishError` into the underlying `EncodeError`, `DecodeError`, or `DecodeSliceError`, discarding the stream.

A decoding error can arrive after earlier output has been written. Treat a streamed destination as complete only after finalization or reading to EOF succeeds.

## Strictness and security

Decoders reject whitespace, bytes outside the selected alphabet, misplaced or forbidden padding, and non-zero unused trailing bits. Padded and unpadded protocols use different profile types; case handling follows the selected profile.

Base encoding is not encryption. Bases does not provide constant-time guarantees and uses data-dependent branches, table lookups, and private scalar or SIMD backends. The [architecture guide](ARCHITECTURE.md) describes backend dispatch and performance contracts.

## Minimum Supported Rust Version (MSRV)

This crate is built against the latest stable release, and its minimum supported rustc version is 1.88.0.

The policy is that the minimum Rust version required to use this crate can be increased in minor version updates. For example, if Bases 1.0 requires Rust 1.20.0, then Bases 1.0.z for all values of z will also require Rust 1.20.0 or newer. However, Bases 1.y for y > 0 may require a newer minimum version of Rust.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
