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

//! Private SIMD CPU feature tokens and target-feature dispatch macro.

/// Zero-sized proof that AArch64 NEON is available.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[derive(Clone, Copy)]
pub struct NeonToken(());

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl NeonToken {
    /// Summons a token proving compile-time NEON availability.
    #[inline(always)]
    pub const fn summon() -> Self {
        Self(())
    }
}

/// Zero-sized proof that x86/x86_64 SSSE3 is available.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx2"),
    any(feature = "std", target_feature = "ssse3")
))]
#[derive(Clone, Copy)]
pub struct Ssse3Token(());

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx2"),
    feature = "std",
    test
))]
impl Ssse3Token {
    /// Detects SSSE3 availability for direct SSSE3 tests.
    #[inline(always)]
    pub fn detect() -> Option<Self> {
        if std::arch::is_x86_feature_detected!("ssse3") {
            Some(Self(()))
        } else {
            None
        }
    }
}

/// Zero-sized proof that x86/x86_64 AVX2 is available.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "std", target_feature = "avx2")
))]
#[derive(Clone, Copy)]
pub struct Avx2Token(());

/// Selected x86/x86_64 SIMD capability token.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
))]
#[derive(Clone, Copy)]
pub enum X86Token {
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    Avx2(Avx2Token),
    #[cfg(not(target_feature = "avx2"))]
    Ssse3(Ssse3Token),
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
))]
impl X86Token {
    /// Selects the widest available x86 SIMD feature token at compile time or runtime.
    #[inline(always)]
    pub fn detect() -> Option<Self> {
        #[cfg(target_feature = "avx2")]
        {
            Some(Self::Avx2(Avx2Token(())))
        }

        #[cfg(all(not(target_feature = "avx2"), feature = "std"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                Some(Self::Avx2(Avx2Token(())))
            } else if std::arch::is_x86_feature_detected!("ssse3") {
                Some(Self::Ssse3(Ssse3Token(())))
            } else {
                None
            }
        }

        #[cfg(all(
            not(target_feature = "avx2"),
            not(feature = "std"),
            target_feature = "ssse3"
        ))]
        {
            Some(Self::Ssse3(Ssse3Token(())))
        }
    }
}

/// Wraps a `#[target_feature]` function into an `#[inline(always)]` safe trampoline gated by a
/// matching ZST feature token.
///
/// The macro binds each token type to its exact `#[target_feature]` string and limits the
/// `unsafe` block to invoking the inner function after verifying the token's concrete type.
macro_rules! simd_fn {
    (
        @emit ($feature:literal, $token_ty:path)
        [ $($meta:meta),* ]
        $vis:vis fn $name:ident
        [ $(const $const_param:ident : $const_ty:ty),* ]
        (
            $token:ident
            $(, $arg:ident : $arg_ty:ty)* $(,)?
        ) $(-> $ret:ty)? $body:block
    ) => {
        $(#[$meta])*
        #[inline(always)]
        $vis fn $name < $(const $const_param : $const_ty),* > (
            $token: $token_ty
            $(, $arg: $arg_ty)*
        ) $(-> $ret)? {
            #[inline]
            #[target_feature(enable = $feature)]
            fn inner < $(const $const_param : $const_ty),* > (
                $($arg: $arg_ty),*
            ) $(-> $ret)? $body

            let _: $token_ty = $token;
            #[allow(unsafe_code, clippy::undocumented_unsafe_blocks)]
            // SAFETY: The caller provided a ZST feature token proving `$feature` is available.
            unsafe { inner::< $($const_param),* >($($arg),*) }
        }
    };
    (
        @resolve
        [ $($meta:meta),* ]
        $vis:vis fn $name:ident
        $const_params:tt
        ( $token:ident : NeonToken $(, $arg:ident : $arg_ty:ty)* $(,)? ) $(-> $ret:ty)? $body:block
    ) => {
        $crate::backend::simd::simd_fn! {
            @emit ("neon", $crate::backend::simd::NeonToken)
            [ $($meta),* ]
            $vis fn $name
            $const_params
            ( $token $(, $arg : $arg_ty)* ) $(-> $ret)? $body
        }
    };
    (
        @resolve
        [ $($meta:meta),* ]
        $vis:vis fn $name:ident
        $const_params:tt
        ( $token:ident : Ssse3Token $(, $arg:ident : $arg_ty:ty)* $(,)? ) $(-> $ret:ty)? $body:block
    ) => {
        $crate::backend::simd::simd_fn! {
            @emit ("ssse3", $crate::backend::simd::Ssse3Token)
            [ $($meta),* ]
            $vis fn $name
            $const_params
            ( $token $(, $arg : $arg_ty)* ) $(-> $ret)? $body
        }
    };
    (
        @resolve
        [ $($meta:meta),* ]
        $vis:vis fn $name:ident
        $const_params:tt
        ( $token:ident : Avx2Token $(, $arg:ident : $arg_ty:ty)* $(,)? ) $(-> $ret:ty)? $body:block
    ) => {
        $crate::backend::simd::simd_fn! {
            @emit ("avx2", $crate::backend::simd::Avx2Token)
            [ $($meta),* ]
            $vis fn $name
            $const_params
            ( $token $(, $arg : $arg_ty)* ) $(-> $ret)? $body
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident
        ( $($args:tt)* ) $(-> $ret:ty)? $body:block
    ) => {
        $crate::backend::simd::simd_fn! {
            @resolve
            [ $($meta),* ]
            $vis fn $name [] ( $($args)* ) $(-> $ret)? $body
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident
        < $(const $const_param:ident : $const_ty:ty),+ $(,)? >
        ( $($args:tt)* ) $(-> $ret:ty)? $body:block
    ) => {
        $crate::backend::simd::simd_fn! {
            @resolve
            [ $($meta),* ]
            $vis fn $name [ $(const $const_param : $const_ty),+ ] ( $($args)* ) $(-> $ret)? $body
        }
    };
}

pub(crate) use simd_fn;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn simd_fn_neon_enables_target_feature() {
        #[target_feature(enable = "neon")]
        fn require_neon() {}

        simd_fn! {
            fn add_offset<const OFFSET: usize>(_token: NeonToken, value: usize) -> usize {
                // Calling a `#[target_feature(enable = "neon")]` fn without `unsafe` proves at
                // compile time that `simd_fn!` enabled `neon` on the generated inner function.
                require_neon();
                value + OFFSET
            }
        }

        assert_eq!(add_offset::<7>(NeonToken::summon(), 35), 42);
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(feature = "std", target_feature = "avx2", target_feature = "ssse3")
    ))]
    #[test]
    fn simd_fn_x86_enables_target_feature() {
        #[cfg(any(feature = "std", target_feature = "avx2"))]
        #[target_feature(enable = "avx2")]
        fn require_avx2() {}

        #[cfg(not(target_feature = "avx2"))]
        #[target_feature(enable = "ssse3")]
        fn require_ssse3() {}

        simd_fn! {
            #[cfg(any(feature = "std", target_feature = "avx2"))]
            fn add_avx2<const OFFSET: usize>(_token: Avx2Token, value: usize) -> usize {
                require_avx2();
                value + OFFSET
            }
        }
        simd_fn! {
            #[cfg(not(target_feature = "avx2"))]
            fn add_ssse3<const OFFSET: usize>(_token: Ssse3Token, value: usize) -> usize {
                require_ssse3();
                value + OFFSET
            }
        }

        let result = match X86Token::detect() {
            #[cfg(any(feature = "std", target_feature = "avx2"))]
            Some(X86Token::Avx2(token)) => Some(add_avx2::<7>(token, 35)),
            #[cfg(not(target_feature = "avx2"))]
            Some(X86Token::Ssse3(token)) => Some(add_ssse3::<7>(token, 35)),
            None => None,
        };
        if let Some(value) = result {
            assert_eq!(value, 42);
        }
    }
}
