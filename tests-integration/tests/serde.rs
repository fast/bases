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
use bases::base16;
use bases::policy;
use serde::Deserialize;
use serde::Serialize;

type Base16UpperStrict = base16::Base16<policy::Upper, policy::Strict>;

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
struct Record {
    #[serde(with = "bases::Base64")]
    payload: Vec<u8>,
    #[serde(with = "bases::Base64UrlUnpadded")]
    cursor: Vec<u8>,
    #[serde(with = "bases::Base32")]
    tag: Vec<u8>,
    #[serde(
        serialize_with = "Base16Lower::serialize",
        deserialize_with = "Base16Lower::deserialize_array"
    )]
    digest: [u8; 4],
    #[serde(
        serialize_with = "Base16UpperStrict::serialize",
        deserialize_with = "Base16UpperStrict::deserialize_array"
    )]
    mac: [u8; 2],
}

#[test]
fn serde_roundtrips_vec_and_fixed_arrays() {
    let record = Record {
        payload: vec![0x66, 0x6f, 0x6f, 0xff, 0xff],
        cursor: vec![0xfb, 0xff],
        tag: b"f".to_vec(),
        digest: [0xde, 0xad, 0xbe, 0xef],
        mac: [0xab, 0xcd],
    };

    let json = serde_json::to_string(&record).unwrap();
    assert_eq!(
        json,
        r#"{"payload":"Zm9v//8=","cursor":"-_8","tag":"MY======","digest":"deadbeef","mac":"ABCD"}"#
    );

    let decoded: Record = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, record);
}

#[test]
fn serde_accepts_json_escaped_encoded_strings() {
    // Escaped slash (`\/`) and `\uXXXX` escapes are unescaped by serde_json before decoding.
    let json = r#"{"payload":"\u005am9v\/\/8=","cursor":"-_8","tag":"\u004dY======","digest":"\u0064eadbeef","mac":"ABCD"}"#;
    let decoded: Record = serde_json::from_str(json).unwrap();
    assert_eq!(decoded.payload, vec![0x66, 0x6f, 0x6f, 0xff, 0xff]);
    assert_eq!(decoded.tag, b"f");
    assert_eq!(decoded.digest, [0xde, 0xad, 0xbe, 0xef]);
}

#[test]
fn serde_rejects_malformed_wrong_length_and_wrong_type_inputs() {
    let bad_cases = [
        // Invalid byte in Base64 payload.
        r#"{"payload":"Zm!v","cursor":"-_8","tag":"MY======","digest":"deadbeef","mac":"ABCD"}"#,
        // Non-canonical trailing bits in Base64 payload.
        r#"{"payload":"Zh==","cursor":"-_8","tag":"MY======","digest":"deadbeef","mac":"ABCD"}"#,
        // Padding rejected on unpadded Base64UrlUnpadded cursor.
        r#"{"payload":"Zm9v","cursor":"-_8=","tag":"MY======","digest":"deadbeef","mac":"ABCD"}"#,
        // Short digest array (2 bytes instead of 4).
        r#"{"payload":"Zm9v","cursor":"-_8","tag":"MY======","digest":"dead","mac":"ABCD"}"#,
        // Long digest array (6 bytes instead of 4).
        r#"{"payload":"Zm9v","cursor":"-_8","tag":"MY======","digest":"deadbeefcafe","mac":"ABCD"}"#,
        // Lowercase rejected by Base16UpperStrict mac field.
        r#"{"payload":"Zm9v","cursor":"-_8","tag":"MY======","digest":"deadbeef","mac":"abcd"}"#,
        // Wrong JSON type (array instead of string).
        r#"{"payload":[1,2,3],"cursor":"-_8","tag":"MY======","digest":"deadbeef","mac":"ABCD"}"#,
        // Wrong JSON type (integer instead of string) for fixed array.
        r#"{"payload":"Zm9v","cursor":"-_8","tag":"MY======","digest":1234,"mac":"ABCD"}"#,
    ];

    for bad in bad_cases {
        assert!(serde_json::from_str::<Record>(bad).is_err(), "input: {bad}");
    }
}
