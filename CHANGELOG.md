# CHANGELOG

All significant changes to this project will be documented in this file.

## v0.1.1 (2026-11-17)

- Write owned, appended, and fixed-array results directly into their destination storage, avoiding redundant initialization, copying, and UTF-8 validation.
- Reuse decoded-length and padding analysis across decoding paths, and encode formatted output in complete quanta without an intermediate stream encoder.
- Improve scalar and SIMD handling of short inputs and tails across Base16, Base32, and Base64, including simpler x86 Base64 character mapping and Base16 decoding.
- Size I/O writer chunks for each encoding profile and retain reusable output buffers across short reads in the streaming example.
- Include the architecture guide, changelog, benchmark guide, and linked example sources in the crate package so README links resolve within the published version.
- Expand contract tests across encoding profiles, buffer boundaries, malformed inputs, streaming retries, I/O, and Serde, including fixed-array Serde support without `alloc`.

## v0.1.0 (2026-11-15)

This is the initial release.
