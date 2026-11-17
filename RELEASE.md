# Releasing Bases

Releases are published manually to crates.io. For the first release, check that the crate name is still available and the repository linked in the package metadata is accessible to its intended readers.

## Prepare

1. Update the version in `bases/Cargo.toml` and refresh `Cargo.lock`.
2. Move user-visible entries from `Unreleased` in `CHANGELOG.md` into a versioned section.
3. Run `cargo x lint`, `cargo x check`, `cargo x test`, and `cargo x package`. The package command builds the archive, verifies its contents and documentation, and tests the extracted crate. Use `cargo x package --allow-dirty` to validate uncommitted changes during development.
4. Run `cargo publish --dry-run --package bases --locked` from the clean release commit to verify the package without uploading it.

## Publish

Publish the verified commit with `cargo publish --package bases --locked`. After crates.io accepts the immutable package, create and push a signed `vX.Y.Z` tag for that same commit.
