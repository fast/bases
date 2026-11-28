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

use bases::Codec;
use data_encoding::Encoding;
use googletest::prelude::*;
use serde::de::value::BorrowedBytesDeserializer;
use serde::de::value::BytesDeserializer;
use serde::de::value::Error;
use serde_json::Value;
use serde_json::value::Serializer;

use crate::payload;

pub fn check<C: Codec>(reference: &Encoding) {
    for len in (0..=16).chain([31, 32, 33, 63, 64, 65, 1025]) {
        let raw = payload(len);
        let expected = reference.encode(&raw);
        let json = C::serialize(&raw, Serializer).unwrap();
        assert_that!(json, eq(&Value::String(expected.clone())));
        assert_that!(C::deserialize(&json).unwrap(), eq(&raw));
        assert_that!(C::deserialize(json).unwrap(), eq(&raw));
        assert_eq!(
            C::deserialize(BorrowedBytesDeserializer::<Error>::new(expected.as_bytes())).unwrap(),
            raw
        );
        assert_eq!(
            C::deserialize(BytesDeserializer::<Error>::new(expected.as_bytes())).unwrap(),
            raw
        );

        if !raw.is_empty() {
            let mut bad = expected.into_bytes();
            bad[0] = 0xff;
            assert!(C::deserialize(BytesDeserializer::<Error>::new(&bad)).is_err());
            bad[0] = b'!';
            assert!(C::deserialize(Value::String(String::from_utf8(bad).unwrap())).is_err());
        }
    }
    for wrong in [
        Value::Null,
        Value::Bool(true),
        Value::from(42),
        Value::Array(vec![]),
        serde_json::json!({}),
    ] {
        assert!(C::deserialize(wrong.clone()).is_err());
        assert!(C::deserialize_array::<_, 0>(wrong).is_err());
    }
    arrays::<C, 0>(reference);
    arrays::<C, 1>(reference);
    arrays::<C, 2>(reference);
    arrays::<C, 3>(reference);
    arrays::<C, 4>(reference);
    arrays::<C, 5>(reference);
    arrays::<C, 16>(reference);
    arrays::<C, 32>(reference);
    arrays::<C, 65>(reference);
}

fn arrays<C: Codec, const N: usize>(reference: &Encoding) {
    let raw = payload(N);
    let wire = reference.encode(&raw);
    assert_eq!(
        C::deserialize_array::<_, N>(Value::String(wire.clone()))
            .unwrap()
            .as_slice(),
        raw
    );
    assert_eq!(
        C::deserialize_array::<_, N>(BytesDeserializer::<Error>::new(wire.as_bytes()))
            .unwrap()
            .as_slice(),
        raw
    );
    for len in [N.saturating_sub(1), N + 1] {
        if len != N {
            let wrong = Value::String(reference.encode(&payload(len)));
            assert!(C::deserialize_array::<_, N>(wrong).is_err());
        }
    }
}
