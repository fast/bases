# Architecture

Bases separates observable encoding semantics from execution strategy.

## Public layers

- Parameterized zero-sized family types (`base16::Base16`, `base32::Base32`, `base64::Base64`) combined with `policy` markers (`Padded`, `Unpadded`, `PaddedOptional`, `UnpaddedOptional`, `Lower`, `Upper`, `Mixed`, `Strict`) define an alphabet, padding policy, and case policy, alongside ten crate-root aliases (`Base16Lower`, `Base16Upper`, `Base32`, `Base32Unpadded`, `Base32Hex`, `Base32HexUnpadded`, `Base64`, `Base64Unpadded`, `Base64Url`, `Base64UrlUnpadded`).
- Concrete family types provide direct inherent methods, including `const fn` sizing and constructors (`new`, `name`, `encoded_len`, `max_decoded_len`, `display`, `encoder`, `decoder`), while the sealed `Codec` trait provides the shared contract for generic `C: Codec` callers.
- Caller-buffer operations accept initialized slices (`encode_into`, `decode_into`) or `MaybeUninit<u8>` slices (`encode_into_uninit`, `decode_into_uninit`). `validate`, `decode_in_place`, and `decode_array` also operate without allocation. The `alloc` feature adds `encode`, `encode_append`, `decode`, and transactional `decode_append`. The `serde` feature adds `serialize`, `deserialize_array`, and, with `alloc`, `deserialize`.
- `Encoder<C>` and `Decoder<C>` preserve one partial quantum for incremental operation without allocation; `std` adds `bases::io::{EncoderWriter, DecoderReader}` over `std::io::{Write, Read}`.
- Formatting adapters (`EncodedDisplay` and explicit-case `HexDisplay`) stream encoded output through `core::fmt`.

## Backend boundary

Each profile maps at compile time to private backend dispatch functions that pair scalar reference implementations with architecture-specific SIMD kernels:

- `aarch64` NEON (`target_feature = "neon"`) for Base16, Base32, and Base64.
- `x86` and `x86_64` AVX2 and SSSE3 for Base16, Base32, and Base64, using runtime `is_x86_feature_detected!` dispatch under `std` or compile-time `target_feature` selection in `no_std`.
- `wasm32` `simd128` (`target_feature = "simd128"`) for Base16 and Base64 encoding and decoding, and for Base16, Base32, and Base64 validation.

SIMD backends accelerate bulk inputs and fall back to the scalar reference implementations for short inputs or tails and under `cfg!(miri)`. Backend selection is private, and every SIMD kernel preserves exact scalar output, error precedence, error byte offsets, and stream-state contracts.

The workspace denies `unsafe_code` by default. Scoped allowances cover architecture intrinsics and internal buffer operations, including writing into uninitialized storage and converting initialized ASCII bytes to strings. These operations document their safety invariants, with `missing_safety_doc` and `undocumented_unsafe_blocks` enforced.

## Performance contract

The primary API writes into caller-owned slices, arrays, or in-place mutable buffers without allocating. Sizing methods let callers reuse exact encoding buffers and safe upper-bound decoding buffers. Successful stream updates consume the complete input slice; failures leave stream state unchanged so callers can retry with more output capacity.

Owned, appended, and fixed-array results are written directly into their destination storage. Only initialized output is exposed to callers; spare capacity does not require zeroing or a second copy. Formatting encodes complete input quanta into a bounded scratch buffer before writing to the formatter.

In-place decoding (`decode_in_place`) writes the decoded prefix back over the encoded buffer and leaves buffer contents unspecified on `DecodeError`. Constant-time execution and in-place encoding are not provided.
