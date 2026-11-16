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

use bases::Base16Lower;
use bases::Base16Upper;
use bases::Base32;
use bases::Base32HexUnpadded;
use bases::Base64;
use bases::Base64UrlUnpadded;
use bases::Decoder;
use bases::EncodeError;
use bases::EncodedDisplay;
use bases::Encoder;
use bases::base16;
use bases::base32;
use bases::policy;

type Base16LowerStrict = base16::Base16<policy::Lower, policy::Strict>;
type Base32HexLowerMixedUnpadded =
    base32::Base32<base32::ExtendedHex, policy::Unpadded, policy::Lower, policy::Mixed>;

const CONST_NAME_BASE16_STRICT: &str = Base16LowerStrict::name();
const CONST_NAME_RFC5155: &str = Base32HexLowerMixedUnpadded::name();
const CONST_ENCODED_LEN: usize = match Base64::encoded_len(5) {
    Ok(len) => len,
    Err(_) => panic!("encoded length fits in usize"),
};
const CONST_OVERFLOW: Result<usize, EncodeError> = Base16Upper::encoded_len(usize::MAX);
const CONST_MAX_DECODED_LEN: usize = Base32HexUnpadded::max_decoded_len(16);
const CONST_DISPLAY: EncodedDisplay<'static, Base64UrlUnpadded> = Base64UrlUnpadded::display(b"hi");
const CONST_ENCODER: Encoder<Base32> = Base32::encoder();
const CONST_DECODER: Decoder<Base16Lower> = Base16Lower::decoder();

#[test]
fn const_fn_inherent_apis_evaluate_at_compile_time() {
    assert_eq!(CONST_NAME_BASE16_STRICT, "base16-lower-strict");
    assert_eq!(CONST_NAME_RFC5155, "base32hex-lower-mixed-unpadded");
    assert_eq!(CONST_ENCODED_LEN, 8);
    assert_eq!(CONST_OVERFLOW, Err(EncodeError::LengthOverflow));
    assert_eq!(CONST_MAX_DECODED_LEN, 10);
    assert_eq!(CONST_DISPLAY.to_string(), "aGk");

    let mut encoded = [0_u8; CONST_ENCODED_LEN];
    let text = Base64::encode_into(b"hello", &mut encoded).unwrap();
    assert_eq!(text, "aGVsbG8=");

    let mut enc = CONST_ENCODER;
    let mut enc_out = [0_u8; 8];
    assert_eq!(enc.update(b"fo", &mut enc_out).unwrap(), "");
    assert_eq!(enc.finish(&mut enc_out).unwrap(), "MZXQ====");

    let mut dec = CONST_DECODER;
    let mut dec_out = [0_u8; 4];
    assert_eq!(dec.update(b"DeAd", &mut dec_out).unwrap(), [0xde, 0xad]);
    assert_eq!(dec.finish(&mut dec_out).unwrap(), b"");
}

#[test]
fn base32_extended_hex_lower_mixed_unpadded_matches_rfc5155() {
    // RFC 5155 Appendix A NSEC3 hashes (H(example) and H(ns1.example)).
    for nsec3_hash in [
        "0p9mhaveqvm6t7vbl5lop2u3t2rp3tom",
        "2t7b4g4vsa5smi47k61mv5bv1a22bojr",
    ] {
        let raw_digest = data_encoding::BASE32HEX_NOPAD
            .decode(nsec3_hash.to_ascii_uppercase().as_bytes())
            .expect("valid RFC 5155 Base32Hex vector");
        let encoded = Base32HexLowerMixedUnpadded::encode(&raw_digest);
        assert_eq!(encoded, nsec3_hash);
        assert_eq!(
            Base32HexLowerMixedUnpadded::decode(encoded.to_ascii_uppercase()).unwrap(),
            raw_digest
        );
    }
}
