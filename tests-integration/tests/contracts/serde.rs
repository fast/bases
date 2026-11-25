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
use bases::serde::Array;
use bases::serde::OptionalArray;
use bases::serde::OptionalVec;
use data_encoding::Encoding;
use googletest::prelude::*;
use serde::Deserializer;
use serde::de;
use serde::de::Visitor;
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
            OptionalVec::<C>::serialize(&Some(raw.clone()), Serializer).unwrap(),
            Value::String(expected.clone())
        );
        assert_eq!(
            OptionalVec::<C>::deserialize(Value::String(expected.clone())).unwrap(),
            Some(raw.clone())
        );
        assert_eq!(
            OptionalVec::<C>::deserialize(Present(BorrowedBytesDeserializer::<Error>::new(
                expected.as_bytes(),
            )))
            .unwrap(),
            Some(raw.clone())
        );
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
            let bad = Value::String(String::from_utf8(bad).unwrap());
            assert!(C::deserialize(bad.clone()).is_err());
            assert!(OptionalVec::<C>::deserialize(bad).is_err());
        }
    }
    assert_eq!(
        OptionalVec::<C>::serialize(&None, Serializer).unwrap(),
        Value::Null
    );
    assert_eq!(OptionalVec::<C>::deserialize(Value::Null).unwrap(), None);
    for wrong in [
        Value::Null,
        Value::Bool(true),
        Value::from(42),
        Value::Array(vec![]),
        serde_json::json!({}),
    ] {
        assert!(C::deserialize(wrong.clone()).is_err());
        assert!(C::deserialize_array::<_, 0>(wrong.clone()).is_err());
        assert!(Array::<C, 0>::deserialize(wrong.clone()).is_err());
        if wrong != Value::Null {
            assert!(OptionalVec::<C>::deserialize(wrong.clone()).is_err());
            assert!(OptionalArray::<C, 0>::deserialize(wrong).is_err());
        }
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
    let array: &[u8; N] = raw.as_slice().try_into().unwrap();
    assert_eq!(
        OptionalArray::<C, N>::serialize(&Some(*array), Serializer).unwrap(),
        Value::String(wire.clone())
    );
    assert_eq!(
        OptionalArray::<C, N>::deserialize(Value::String(wire.clone())).unwrap(),
        Some(*array)
    );
    assert_eq!(
        OptionalArray::<C, N>::deserialize(Present(BytesDeserializer::<Error>::new(
            wire.as_bytes()
        )))
        .unwrap(),
        Some(*array)
    );
    assert_eq!(
        OptionalArray::<C, N>::serialize(&None, Serializer).unwrap(),
        Value::Null
    );
    assert_eq!(
        OptionalArray::<C, N>::deserialize(Value::Null).unwrap(),
        None
    );
    assert_eq!(
        Array::<C, N>::serialize(array, Serializer).unwrap(),
        Value::String(wire.clone())
    );
    assert_eq!(
        Array::<C, N>::deserialize(Value::String(wire.clone()))
            .unwrap()
            .as_slice(),
        raw
    );
    assert_eq!(
        Array::<C, N>::deserialize(BorrowedBytesDeserializer::<Error>::new(wire.as_bytes()))
            .unwrap()
            .as_slice(),
        raw
    );
    assert_eq!(
        Array::<C, N>::deserialize(BytesDeserializer::<Error>::new(wire.as_bytes()))
            .unwrap()
            .as_slice(),
        raw
    );
    if N > 0 {
        let mut invalid = wire.as_bytes().to_vec();
        invalid[0] = 0xff;
        assert!(Array::<C, N>::deserialize(BytesDeserializer::<Error>::new(&invalid)).is_err());
        assert!(
            OptionalArray::<C, N>::deserialize(Present(BytesDeserializer::<Error>::new(&invalid)))
                .is_err()
        );
    }
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
            assert!(C::deserialize_array::<_, N>(wrong.clone()).is_err());
            assert!(Array::<C, N>::deserialize(wrong.clone()).is_err());
            assert!(OptionalArray::<C, N>::deserialize(wrong).is_err());
        }
    }
}

struct Present<D>(D);

impl<'de, D: Deserializer<'de>> Deserializer<'de> for Present<D> {
    type Error = D::Error;

    fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Self::Error> {
        Err(de::Error::custom("expected deserialize_option"))
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_some(self.0)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct newtype_struct seq tuple tuple_struct
        map struct enum identifier ignored_any
    }
}
