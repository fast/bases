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

use crate::policy::sealed::DecodePadding;

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
    const ENCODE_PADDED: bool = true;
    const DECODE_PADDING: DecodePadding = DecodePadding::Required;
}

impl sealed::PaddingPolicy for Unpadded {
    const ENCODE_PADDED: bool = false;
    const DECODE_PADDING: DecodePadding = DecodePadding::Forbidden;
}

impl sealed::PaddingPolicy for PaddedOptional {
    const ENCODE_PADDED: bool = true;
    const DECODE_PADDING: DecodePadding = DecodePadding::Optional;
}

impl sealed::PaddingPolicy for UnpaddedOptional {
    const ENCODE_PADDED: bool = false;
    const DECODE_PADDING: DecodePadding = DecodePadding::Optional;
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
    #[derive(Clone, Copy)]
    pub enum DecodePadding {
        Required,
        Forbidden,
        Optional,
    }

    impl DecodePadding {
        #[inline]
        pub fn uses_padded_decoder(self, input: &[u8]) -> bool {
            match self {
                Self::Required => true,
                Self::Forbidden => false,
                Self::Optional => input.last() == Some(&b'='),
            }
        }
    }

    pub trait PaddingPolicy: 'static {
        const ENCODE_PADDED: bool;
        const DECODE_PADDING: DecodePadding;
    }

    macro_rules! profile_name {
        ($base:literal, $padding:ty) => {{
            use $crate::policy::sealed::DecodePadding;
            match <$padding>::DECODE_PADDING {
                DecodePadding::Required => $base,
                DecodePadding::Forbidden => concat!($base, "-unpadded"),
                DecodePadding::Optional if <$padding>::ENCODE_PADDED => {
                    concat!($base, "-padded-optional")
                }
                DecodePadding::Optional => concat!($base, "-unpadded-optional"),
            }
        }};
    }
    pub(crate) use profile_name;

    pub trait LetterCase: 'static {
        const UPPERCASE: bool;
    }

    pub trait DecodeCasePolicy: 'static {
        const STRICT: bool;
    }
}
