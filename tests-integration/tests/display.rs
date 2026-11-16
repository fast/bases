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

use core::fmt;
use core::fmt::Write;
use std::array;

use bases::Base32;
use bases::Base64;
use bases::HexDisplay;

#[test]
fn display_writes_the_complete_message() {
    let input: [u8; 1021] = array::from_fn(|index| index as u8);
    assert!(Base32::display(&[]).to_string().is_empty());
    assert_eq!(
        format!("{:.3}", Base32::display(&input)),
        data_encoding::BASE32.encode(&input)
    );
    assert_eq!(
        format!("{:X}", HexDisplay(&input)),
        data_encoding::HEXUPPER.encode(&input)
    );
    assert_eq!(
        format!("{:x}", HexDisplay(&input)),
        data_encoding::HEXLOWER.encode(&input)
    );
}

#[test]
fn hex_display_ignores_width_and_alternate_flags() {
    let id = HexDisplay(&[0x00, 0xab, 0xcd]);
    assert_eq!(format!("{id:>12X}"), "00ABCD");
    assert_eq!(format!("{id:>12x}"), "00abcd");
    assert_eq!(format!("{id:#X}"), "00ABCD");
    assert_eq!(format!("{id:#x}"), "00abcd");
}

#[test]
fn display_propagates_formatter_errors() {
    struct RejectData;

    impl fmt::Write for RejectData {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            if value.is_empty() {
                Ok(())
            } else {
                Err(fmt::Error)
            }
        }
    }

    for input in [b"foo".as_slice(), b"f"] {
        assert!(write!(RejectData, "{}", Base64::display(input)).is_err());
    }
}

#[test]
fn display_handles_profile_cutoffs_and_chunk_boundaries() {
    let buf: [u8; 1025] = array::from_fn(|index| (index * 37 + 13) as u8);
    for len in [
        1, 30, 31, 32, 33, 39, 40, 41, 47, 48, 49, 240, 512, 640, 768, 1024, 1025,
    ] {
        let slice = &buf[..len];
        assert_eq!(
            format!("{:x}", HexDisplay(slice)),
            data_encoding::HEXLOWER.encode(slice)
        );
        assert_eq!(
            Base32::display(slice).to_string(),
            data_encoding::BASE32.encode(slice)
        );
        assert_eq!(
            Base64::display(slice).to_string(),
            data_encoding::BASE64.encode(slice)
        );
    }
}
