# Changelog

All significant changes to this project will be documented in this file.

## Unreleased

### Features

* Add `decoded_len` for exact buffer sizing and `check` for validation without decoded output to all profiles and the `Codec` trait. `decoded_len` checks length and trailing padding shape without fully validating the input; `check` preserves decoding errors and byte offsets.
* Add `policy::PaddedOptional` and `policy::UnpaddedOptional` for Base32 and Base64. Both accept complete padding or none when decoding and emit padded and unpadded output, respectively.
* Add `encode_into_uninit` and `decode_into_uninit` to all profiles and the `Codec` trait. They write into caller-provided `MaybeUninit<u8>` storage and return the initialized prefix.

## v0.1.1 (2026-11-17)

### Improvements

* Include a self-contained usage guide and architecture documentation in the crate package, and organize repository navigation for examples, benchmarks, and release notes.

### Performance improvements

* Avoid redundant initialization, copying, and UTF-8 validation when encoding or decoding owned results, appending to existing buffers, and decoding fixed-size arrays.
* Reuse decoded-length and padding analysis across decoding paths to avoid repeated validation.
* Reduce formatting and incremental streaming overhead by processing complete encoding quanta directly.
* Improve scalar and SIMD handling of short inputs and tails across Base16, Base32, and Base64, including simpler x86 Base64 character mapping and Base16 decoding.
* Let `io::EncoderWriter` process larger input chunks according to the encoding profile, and avoid redundant buffer initialization across short reads in the streaming example.

## v0.1.0 (2026-11-15)

This is the initial release.
