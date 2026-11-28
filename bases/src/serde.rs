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

//! Serde adapters for encoded byte fields.
//!
//! Optional adapters preserve Serde's `Some` and `None` representation. In JSON,
//! these are encoded strings and `null`. An empty encoded string is a present
//! value and must satisfy the selected profile and array length.
//!
//! Use `#[serde(default, with = "...")]` to accept a missing optional field as
//! `None`, and `skip_serializing_if = "Option::is_none"` to omit it when absent.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::marker::PhantomData;

use serde_core::Deserializer;
use serde_core::Serializer;

use crate::Codec;

pub(crate) mod internal;

/// A Serde adapter for `[u8; N]` fields encoded with profile `C`.
///
/// Use this type with `#[serde(with = "...")]` directly or through a type alias.
/// Serialization accepts exactly `[u8; N]`, and deserialization rejects input
/// that does not decode to exactly `N` bytes.
///
/// The adapter works without `alloc`.
/// Allocation by the serializer or deserializer depends on the format.
///
/// # Examples
///
/// ```
/// use bases::Base16Lower;
/// use bases::serde::Array;
/// use serde::Deserialize;
/// use serde::Serialize;
///
/// type DigestHex = Array<Base16Lower, 4>;
///
/// #[derive(Debug, PartialEq, Serialize, Deserialize)]
/// struct Record {
///     #[serde(with = "DigestHex")]
///     digest: [u8; 4],
/// }
///
/// let record = Record {
///     digest: [0xde, 0xad, 0xbe, 0xef],
/// };
/// let json = r#"{"digest":"deadbeef"}"#;
/// assert_eq!(serde_json::to_string(&record).unwrap(), json);
/// assert_eq!(serde_json::from_str::<Record>(json).unwrap(), record);
/// ```
pub struct Array<C: Codec, const N: usize>(PhantomData<C>);

impl<C: Codec, const N: usize> Array<C, N> {
    /// Serializes a byte array as an encoded string using profile `C`.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `serializer`.
    #[inline]
    pub fn serialize<S: Serializer>(bytes: &[u8; N], serializer: S) -> Result<S::Ok, S::Error> {
        internal::serialize::<C, _, S>(bytes, serializer)
    }

    /// Deserializes an encoded string into a byte array using profile `C`.
    ///
    /// # Errors
    ///
    /// Returns a deserialization error if the input is not a valid encoded
    /// string for this profile or does not decode to exactly `N` bytes.
    #[inline]
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; N], D::Error> {
        internal::deserialize_array::<C, D, N>(deserializer)
    }
}

/// A Serde adapter for `Option<[u8; N]>` fields encoded with profile `C`.
///
/// Present values follow [`Array`]'s encoding and exact-length rules. Absent
/// values use Serde's `None` representation. This adapter works without `alloc`.
///
/// # Examples
///
/// ```
/// use bases::Base16Lower;
/// use bases::serde::OptionalArray;
/// use serde::Deserialize;
/// use serde::Serialize;
///
/// type MaybeDigest = OptionalArray<Base16Lower, 4>;
///
/// #[derive(Debug, PartialEq, Serialize, Deserialize)]
/// struct Record {
///     #[serde(default, with = "MaybeDigest", skip_serializing_if = "Option::is_none")]
///     digest: Option<[u8; 4]>,
/// }
///
/// let absent = Record { digest: None };
/// assert_eq!(serde_json::from_str::<Record>("{}").unwrap(), absent);
/// assert_eq!(serde_json::to_string(&absent).unwrap(), "{}");
/// let json = r#"{"digest":"deadbeef"}"#;
/// let present = Record {
///     digest: Some([0xde, 0xad, 0xbe, 0xef]),
/// };
/// assert_eq!(serde_json::from_str::<Record>(json).unwrap(), present);
/// assert_eq!(serde_json::to_string(&present).unwrap(), json);
/// ```
pub struct OptionalArray<C: Codec, const N: usize>(PhantomData<C>);

impl<C: Codec, const N: usize> OptionalArray<C, N> {
    /// Serializes an optional byte array using profile `C` for present values.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `serializer`.
    #[inline]
    pub fn serialize<S: Serializer>(
        bytes: &Option<[u8; N]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        internal::serialize_option::<C, _, S>(bytes, serializer)
    }

    /// Deserializes an optional encoded string into a byte array using profile `C`.
    ///
    /// # Errors
    ///
    /// Returns a deserialization error if a present value is not valid for the
    /// profile or does not decode to exactly `N` bytes.
    #[inline]
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<[u8; N]>, D::Error> {
        internal::deserialize_optional_array::<C, D, N>(deserializer)
    }
}

/// A Serde adapter for `Option<Vec<u8>>` fields encoded with profile `C`.
///
/// Present values follow the profile's encoding rules. Absent values use
/// Serde's `None` representation; an empty encoded string decodes to `Some(Vec::new())`.
///
/// # Examples
///
/// ```
/// use serde::Deserialize;
/// use serde::Serialize;
///
/// #[derive(Debug, PartialEq, Serialize, Deserialize)]
/// struct Record {
///     #[serde(default, with = "bases::serde::OptionalVec::<bases::Base64>")]
///     payload: Option<Vec<u8>>,
/// }
///
/// let absent = Record { payload: None };
/// assert_eq!(serde_json::from_str::<Record>("{}").unwrap(), absent);
/// assert_eq!(
///     serde_json::to_string(&absent).unwrap(),
///     r#"{"payload":null}"#
/// );
/// let empty = Record {
///     payload: Some(Vec::new()),
/// };
/// let json = r#"{"payload":""}"#;
/// assert_eq!(serde_json::from_str::<Record>(json).unwrap(), empty);
/// assert_eq!(serde_json::to_string(&empty).unwrap(), json);
/// ```
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub struct OptionalVec<C: Codec>(PhantomData<C>);

#[cfg(feature = "alloc")]
impl<C: Codec> OptionalVec<C> {
    /// Serializes an optional byte vector using profile `C` for present values.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `serializer`.
    #[inline]
    pub fn serialize<S: Serializer>(
        bytes: &Option<Vec<u8>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        internal::serialize_option::<C, _, S>(bytes, serializer)
    }

    /// Deserializes an optional encoded string into a byte vector using profile `C`.
    ///
    /// # Errors
    ///
    /// Returns a deserialization error if a present value is not valid for the profile.
    #[inline]
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<u8>>, D::Error> {
        internal::deserialize_optional_vec::<C, D>(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde::Serialize;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Record {
        #[serde(with = "crate::serde::Array::<crate::Base64, 1>")]
        byte: [u8; 1],
        #[serde(with = "crate::serde::Array::<crate::Base16Lower, 0>")]
        empty: [u8; 0],
    }

    #[test]
    fn array_adapter_without_alloc() {
        let record = Record {
            byte: *b"f",
            empty: [],
        };
        let json = r#"{"byte":"Zg==","empty":""}"#;
        assert_eq!(serde_json::to_string(&record).unwrap(), json);
        assert_eq!(serde_json::from_str::<Record>(json).unwrap(), record);
        for invalid in [
            r#"{"byte":"","empty":""}"#,
            r#"{"byte":"Zm8=","empty":""}"#,
            r#"{"byte":"!g==","empty":""}"#,
            r#"{"byte":"Zh==","empty":""}"#,
            r#"{"byte":"Zg","empty":""}"#,
            r#"{"byte":"Zg==","empty":"00"}"#,
            r#"{"byte":null,"empty":""}"#,
            r#"{"byte":[102],"empty":""}"#,
        ] {
            assert!(
                serde_json::from_str::<Record>(invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct OptionalRecord {
        #[serde(default, with = "crate::serde::OptionalArray::<crate::Base64, 1>")]
        byte: Option<[u8; 1]>,
        #[serde(
            default,
            with = "crate::serde::OptionalArray::<crate::Base16Lower, 0>",
            skip_serializing_if = "Option::is_none"
        )]
        empty: Option<[u8; 0]>,
    }

    #[test]
    fn optional_array_without_alloc() {
        for (record, json) in [
            (
                OptionalRecord {
                    byte: None,
                    empty: None,
                },
                r#"{"byte":null}"#,
            ),
            (
                OptionalRecord {
                    byte: Some(*b"f"),
                    empty: Some([]),
                },
                r#"{"byte":"Zg==","empty":""}"#,
            ),
        ] {
            assert_eq!(serde_json::to_string(&record).unwrap(), json);
            assert_eq!(
                serde_json::from_str::<OptionalRecord>(json).unwrap(),
                record
            );
        }
        assert_eq!(
            serde_json::from_str::<OptionalRecord>("{}").unwrap(),
            OptionalRecord {
                byte: None,
                empty: None
            }
        );
        for invalid in [
            r#"{"byte":""}"#,
            r#"{"byte":"Zm8="}"#,
            r#"{"byte":"!g=="}"#,
            r#"{"byte":"Zh=="}"#,
            r#"{"byte":"Zg"}"#,
            r#"{"empty":"00"}"#,
            r#"{"byte":false}"#,
            r#"{"byte":[102]}"#,
        ] {
            assert!(
                serde_json::from_str::<OptionalRecord>(invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
