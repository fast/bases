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

//! Policy markers for encoding and decoding profiles.
//!
//! For Base32 and Base64, [`PaddedOptional`] and [`UnpaddedOptional`] select the encoded
//! output. Both accept input with complete padding or with the padding omitted.
//!
//! ```
//! use bases::base64;
//! use bases::policy;
//!
//! type Padded = base64::Base64<base64::Standard, policy::PaddedOptional>;
//! type Unpadded = base64::Base64<base64::Standard, policy::UnpaddedOptional>;
//!
//! let mut output = [0; 4];
//! assert_eq!(Padded::encode_into(b"f", &mut output).unwrap(), "Zg==");
//! assert_eq!(Unpadded::encode_into(b"f", &mut output).unwrap(), "Zg");
//!
//! for input in ["Zg", "Zg=="] {
//!     assert_eq!(Padded::decode_array(input), Ok(*b"f"));
//!     assert_eq!(Unpadded::decode_array(input), Ok(*b"f"));
//! }
//! assert!(Padded::decode_array::<1>("Zg=").is_err());
//! assert!(Unpadded::decode_array::<1>("Zg=").is_err());
//! ```

/// Padding policy that requires and emits canonical `=` padding.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Padded;

/// Padding policy that omits and rejects `=` padding.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Unpadded;

/// Emits canonical `=` padding. Decoding accepts complete padding or none.
///
/// For example, standard Base64 encodes `b"f"` as `Zg==` and accepts both `Zg==` and `Zg`.
/// Partial padding such as `Zg=` is invalid. See [`UnpaddedOptional`] for unpadded output.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PaddedOptional;

/// Omits `=` padding. Decoding accepts complete padding or none.
///
/// For example, standard Base64 encodes `b"f"` as `Zg` and accepts both `Zg` and `Zg==`.
/// Partial padding such as `Zg=` is invalid. See [`PaddedOptional`] for padded output.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct UnpaddedOptional;

/// Letter case policy that emits lowercase ASCII letters.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Lower;

/// Letter case policy that emits uppercase ASCII letters.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Upper;

/// Decoding case policy that accepts both uppercase and lowercase ASCII letters.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Mixed;

/// Decoding case policy that accepts only the configured letter case.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Strict;

impl sealed::PaddingPolicy for Padded {
    const EMIT_PADDING: bool = true;
    const DECODE_PADDING_OPTIONAL: bool = false;
}

impl sealed::PaddingPolicy for Unpadded {
    const EMIT_PADDING: bool = false;
    const DECODE_PADDING_OPTIONAL: bool = false;
}

impl sealed::PaddingPolicy for PaddedOptional {
    const EMIT_PADDING: bool = true;
    const DECODE_PADDING_OPTIONAL: bool = true;
}

impl sealed::PaddingPolicy for UnpaddedOptional {
    const EMIT_PADDING: bool = false;
    const DECODE_PADDING_OPTIONAL: bool = true;
}

impl sealed::LetterCase for Lower {
    const UPPERCASE: bool = false;
}

impl sealed::LetterCase for Upper {
    const UPPERCASE: bool = true;
}

impl sealed::DecodeCasePolicy for Mixed {
    const STRICT: bool = false;
}

impl sealed::DecodeCasePolicy for Strict {
    const STRICT: bool = true;
}

pub(crate) mod sealed {
    pub trait PaddingPolicy: 'static {
        const EMIT_PADDING: bool;
        const DECODE_PADDING_OPTIONAL: bool;
        const ACCEPTS_PADDING: bool = Self::EMIT_PADDING || Self::DECODE_PADDING_OPTIONAL;

        #[inline]
        fn decode_padded(input: &[u8]) -> bool {
            if Self::DECODE_PADDING_OPTIONAL {
                input.last() == Some(&b'=')
            } else {
                Self::EMIT_PADDING
            }
        }
    }

    macro_rules! profile_name {
        ($base:literal, $padding:ty) => {
            match (
                <$padding>::EMIT_PADDING,
                <$padding>::DECODE_PADDING_OPTIONAL,
            ) {
                (true, false) => $base,
                (false, false) => concat!($base, "-unpadded"),
                (true, true) => concat!($base, "-padded-optional"),
                (false, true) => concat!($base, "-unpadded-optional"),
            }
        };
    }
    pub(crate) use profile_name;

    pub trait LetterCase: 'static {
        const UPPERCASE: bool;
    }

    pub trait DecodeCasePolicy: 'static {
        const STRICT: bool;
    }
}
