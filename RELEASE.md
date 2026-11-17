# Releasing Bases

Releases are published manually to crates.io from a clean commit that has passed CI.

## Prepare

1. Update the version in `bases/Cargo.toml` and refresh `Cargo.lock`.
2. Add a versioned section to the [changelog](CHANGELOG.md) for user-visible changes.
3. Run `cargo x lint`, `cargo x check`, and `cargo x test`.
4. From the clean release commit, review `cargo package --list --package bases --locked` and run `cargo publish --dry-run --package bases --all-features --locked`. Cargo builds the archive and verifies that the extracted crate compiles without uploading it. Check that the package contains its README, architecture guide, and license, with relative documentation links resolving inside the package.

## Publish

Publish the verified commit with `cargo publish --package bases --locked`. After crates.io accepts the immutable package, create and push a signed `vX.Y.Z` tag for that same commit.
