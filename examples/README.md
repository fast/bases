# Examples

Runnable examples live in this workspace crate. Each program uses local sample data and can run without arguments from the repository root:

```shell
cargo run --package examples --example url_cursor
```

| Example                                       | Application                                                                                 | APIs to start with                                       |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| [`base64_field`](src/base64_field.rs)         | Carry binary data in a JSON string field and reject a malformed field.                      | `Base64::encode`, `Base64::decode`, `DecodeError`        |
| [`url_cursor`](src/url_cursor.rs)             | Encode a record ID into an unpadded URL-safe pagination cursor and recover it.              | `Base64UrlUnpadded::encode`, `decode_array`              |
| [`reusable_buffers`](src/reusable_buffers.rs) | Reuse output buffers across a batch of records with different encoding profiles.            | `C: Codec`, sizing helpers, `encode_into`, `decode_into` |
| [`stream_io`](src/stream_io.rs)               | Convert a file or network stream with bounded buffers and explicit finalization.            | `Encoder<C>`, `Decoder<C>`, `update`, `finish`           |
| [`format_bytes`](src/format_bytes.rs)         | Format binary IDs and payloads in diagnostic output without an intermediate encoded string. | `HexDisplay` (`{:x}`/`{:X}`), `Base64::display`          |
| [`profiles`](src/profiles.rs)                 | Compare Base16, Base32, Base64, and Base64URL representations.                              | `Base16Lower`, `Base16Upper`, `Base32`, `Base64`         |

## Stream I/O

The `stream_io` example also accepts `encode` or `decode` to transform standard input into standard output using padded Base64:

```shell
printf 'hello, world' | cargo run --quiet --package examples --example stream_io -- encode
printf 'aGVsbG8sIHdvcmxk' | cargo run --quiet --package examples --example stream_io -- decode
```

Its `encode` and `decode` helpers accept any `Read` and `Write`, including files and sockets. Successful updates consume the complete input chunk; `finish()` handles the final fragment at EOF. A decoding error can arrive after earlier output has been written, so callers should treat the destination as complete only after the whole operation succeeds.
