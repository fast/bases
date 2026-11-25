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

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use serde_core::Deserializer;
use serde_core::Serialize;
use serde_core::Serializer;
use serde_core::de::DeserializeSeed;
use serde_core::de::Error;
use serde_core::de::Visitor;

use crate::Codec;
use crate::codec;

pub fn serialize<C, T, S>(bytes: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    C: Codec,
    T: ?Sized + AsRef<[u8]>,
    S: Serializer,
{
    serializer.collect_str(&C::display(bytes.as_ref()))
}

pub fn serialize_option<C, T, S>(bytes: &Option<T>, serializer: S) -> Result<S::Ok, S::Error>
where
    C: Codec,
    T: AsRef<[u8]>,
    S: Serializer,
{
    struct Encoded<'a, C>(&'a [u8], PhantomData<C>);

    impl<C: Codec> Serialize for Encoded<'_, C> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serialize::<C, _, S>(self.0, serializer)
        }
    }

    match bytes {
        Some(bytes) => serializer.serialize_some(&Encoded::<C>(bytes.as_ref(), PhantomData)),
        None => serializer.serialize_none(),
    }
}

pub fn deserialize_optional_array<'de, C, D, const N: usize>(
    deserializer: D,
) -> Result<Option<[u8; N]>, D::Error>
where
    C: Codec,
    D: Deserializer<'de>,
{
    struct ArraySeed<C, const N: usize>(PhantomData<C>);

    impl<'de, C: Codec, const N: usize> DeserializeSeed<'de> for ArraySeed<C, N> {
        type Value = [u8; N];

        fn deserialize<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserialize_array::<C, D, N>(deserializer)
        }
    }

    deserialize_option(deserializer, ArraySeed::<C, N>(PhantomData))
}

#[cfg(feature = "alloc")]
pub fn deserialize_optional_vec<'de, C, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
where
    C: Codec,
    D: Deserializer<'de>,
{
    struct VecSeed<C>(PhantomData<C>);

    impl<'de, C: Codec> DeserializeSeed<'de> for VecSeed<C> {
        type Value = Vec<u8>;

        fn deserialize<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserialize::<C, D>(deserializer)
        }
    }

    deserialize_option(deserializer, VecSeed::<C>(PhantomData))
}

fn deserialize_option<'de, D, S>(deserializer: D, seed: S) -> Result<Option<S::Value>, D::Error>
where
    D: Deserializer<'de>,
    S: DeserializeSeed<'de>,
{
    struct OptionVisitor<S>(S);

    impl<'de, S: DeserializeSeed<'de>> Visitor<'de> for OptionVisitor<S> {
        type Value = Option<S::Value>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an optional encoded string")
        }

        fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            self.0.deserialize(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(OptionVisitor(seed))
}

pub fn deserialize_array<'de, C, D, const N: usize>(deserializer: D) -> Result<[u8; N], D::Error>
where
    C: Codec,
    D: Deserializer<'de>,
{
    struct ArrayVisitor<C, const N: usize>(PhantomData<C>);

    impl<C: Codec, const N: usize> Visitor<'_> for ArrayVisitor<C, N> {
        type Value = [u8; N];

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a {} encoded string", C::name())
        }

        fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
            codec::decode_array::<C, N>(value.as_bytes()).map_err(E::custom)
        }

        fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
            codec::decode_array::<C, N>(value).map_err(E::custom)
        }
    }

    deserializer.deserialize_str(ArrayVisitor::<C, N>(PhantomData))
}

#[cfg(feature = "alloc")]
pub fn deserialize<'de, C, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    C: Codec,
    D: Deserializer<'de>,
{
    struct VecVisitor<C>(PhantomData<C>);

    impl<C: Codec> Visitor<'_> for VecVisitor<C> {
        type Value = Vec<u8>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a {} encoded string", C::name())
        }

        fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
            codec::decode::<C>(value.as_bytes()).map_err(E::custom)
        }

        fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
            codec::decode::<C>(value).map_err(E::custom)
        }
    }

    deserializer.deserialize_str(VecVisitor::<C>(PhantomData))
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use serde::Deserialize;
    use serde::Serialize;

    use crate::Base16Lower;
    use crate::Base64;
    use crate::Base64UrlUnpadded;

    #[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
    struct Packet {
        #[serde(with = "crate::Base64")]
        payload: Vec<u8>,
        #[serde(
            serialize_with = "crate::Base16Lower::serialize",
            deserialize_with = "crate::Base16Lower::deserialize_array"
        )]
        digest: [u8; 4],
        #[serde(with = "crate::Base64UrlUnpadded")]
        cursor: Vec<u8>,
    }

    #[test]
    fn serde_with_profile_roundtrip() {
        let packet = Packet {
            payload: vec![0x66, 0x6f, 0x6f],
            digest: [0xde, 0xad, 0xbe, 0xef],
            cursor: vec![0xfb, 0xff],
        };

        let json = serde_json::to_string(&packet).unwrap();
        assert_eq!(
            json,
            r#"{"payload":"Zm9v","digest":"deadbeef","cursor":"-_8"}"#
        );

        let decoded: Packet = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn serde_rejects_invalid_length_and_data() {
        let bad_len = r#"{"payload":"Zm9v","digest":"dead","cursor":"-_8"}"#;
        assert!(serde_json::from_str::<Packet>(bad_len).is_err());

        let bad_char = r#"{"payload":"Zm!v","digest":"deadbeef","cursor":"-_8"}"#;
        assert!(serde_json::from_str::<Packet>(bad_char).is_err());

        let _ = (Base64::new(), Base16Lower::new(), Base64UrlUnpadded::new());
    }
}

#[cfg(all(test, not(feature = "alloc")))]
mod tests_no_alloc {
    use serde::Deserialize;
    use serde::Serialize;

    use crate::Base64;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Digest {
        #[serde(
            serialize_with = "Base64::serialize",
            deserialize_with = "Base64::deserialize_array"
        )]
        bytes: [u8; 3],
    }

    #[test]
    fn fixed_array_serde_without_alloc() {
        let digest = Digest { bytes: *b"foo" };
        let wire = r#"{"bytes":"Zm9v"}"#;
        assert_eq!(serde_json::to_string(&digest).unwrap(), wire);
        assert_eq!(serde_json::from_str::<Digest>(wire).unwrap(), digest);
        for invalid in [
            r#"{"bytes":"Zg=="}"#,
            r#"{"bytes":"Zm9vYg=="}"#,
            r#"{"bytes":"Zm!v"}"#,
        ] {
            assert!(serde_json::from_str::<Digest>(invalid).is_err());
        }
    }
}
