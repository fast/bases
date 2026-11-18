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

use crate::policy::sealed::PaddingMode;

/// Padding policy that requires and emits canonical `=` padding.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Padded;

/// Padding policy that omits and rejects `=` padding.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Unpadded;

/// Omits padding when encoding and accepts either complete canonical padding or none when decoding.
///
/// For example, gRPC binary metadata uses this policy with the standard Base64 alphabet.
///
/// ```
/// use bases::base64;
/// use bases::policy;
///
/// type Metadata = base64::Base64<base64::Standard, policy::OptionalPadding>;
///
/// let mut output = [0; 4];
/// assert_eq!(Metadata::encode_into(b"f", &mut output).unwrap(), "Zg");
/// assert_eq!(Metadata::decode_array("Zg"), Ok(*b"f"));
/// assert_eq!(Metadata::decode_array("Zg=="), Ok(*b"f"));
/// assert!(Metadata::decode_array::<1>("Zg=").is_err());
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct OptionalPadding;

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
    const MODE: PaddingMode = PaddingMode::Required;
}

impl sealed::PaddingPolicy for Unpadded {
    const MODE: PaddingMode = PaddingMode::Forbidden;
}

impl sealed::PaddingPolicy for OptionalPadding {
    const MODE: PaddingMode = PaddingMode::Optional;
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
    pub enum PaddingMode {
        Required,
        Forbidden,
        Optional,
    }

    impl PaddingMode {
        #[inline]
        pub fn decode_padded(self, input: &[u8]) -> bool {
            match self {
                Self::Required => true,
                Self::Forbidden => false,
                Self::Optional => input.last() == Some(&b'='),
            }
        }
    }

    pub trait PaddingPolicy: 'static {
        const MODE: PaddingMode;
        const EMIT_PADDING: bool = matches!(Self::MODE, PaddingMode::Required);
    }

    pub trait LetterCase: 'static {
        const UPPERCASE: bool;
    }

    pub trait DecodeCasePolicy: 'static {
        const STRICT: bool;
    }
}
