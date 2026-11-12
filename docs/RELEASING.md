# Releasing Vespasian

Releases come from version tags in [chrisvoncsefalvay/vespasian](https://github.com/chrisvoncsefalvay/vespasian). The default branch is `main`.

## Repository setup

1. Enable GitHub Actions in the repository.
2. Under Settings → Pages, choose **GitHub Actions** as the source.
3. Create the `crates-io` environment for the publication workflow.
4. For the public launch, set the repository visibility to Public so source, recordings and release downloads share the same public URLs.

The Pages workflow builds `dist/site` with `python scripts/build_site.py` and deploys changes from `main`. Its address is [chrisvoncsefalvay.github.io/vespasian](https://chrisvoncsefalvay.github.io/vespasian/).

## Prepare a version

Update the version in `Cargo.toml`, run `cargo check` to refresh `Cargo.lock`, and add its changes to `CHANGELOG.md`. The site reads the Cargo version during its build.

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
python -m unittest discover -s python -p 'test_*.py'
python -m unittest discover -s scripts -p '*_test.py'
python scripts/build_site.py
cargo package --locked
```

Commit the version, then tag and push it:

```sh
git tag -a v0.3.0 -m 'Vespasian 0.3.0'
git push origin main
git push origin v0.3.0
```

## Build the release

The **Release** workflow runs the full test matrix, builds native binaries, packages the crate, and creates a GitHub release draft. An existing tag can also be selected through Actions → Release → Run workflow.

| Platform | Release asset |
| --- | --- |
| Linux x64 | `vespasian-0.3.0-x86_64-unknown-linux-gnu.tar.gz` |
| Linux ARM64 | `vespasian-0.3.0-aarch64-unknown-linux-gnu.tar.gz` |
| macOS Apple Silicon | `vespasian-0.3.0-aarch64-apple-darwin.tar.gz` |
| Windows x64 | `vespasian-0.3.0-x86_64-pc-windows-msvc.zip` |
| Windows ARM64 | `vespasian-0.3.0-aarch64-pc-windows-msvc.zip` |

Each archive contains a `vespasian-0.3.0` directory with the executable, examples and documentation. Windows ARM64 uses an x64 Python runtime through [Windows emulation supported by uv](https://docs.astral.sh/uv/concepts/python-versions/#transparent-x86_64-emulation-on-aarch64).

The draft includes `vespasian-0.3.0.crate`, `SHA256SUMS.txt` and `release.json`. The manifest joins all five platform archives and the crate to the same Git commit. Packaging checks executable architecture, version and file hashes.

Download the draft assets, check their hashes, and run the extracted executable:

```sh
sha256sum --check SHA256SUMS.txt
./vespasian-0.3.0/vespasian --version
./vespasian-0.3.0/vespasian setup
./vespasian-0.3.0/vespasian doctor
```

On macOS use `shasum -a 256 -c SHA256SUMS.txt`. On Windows use `Get-FileHash -Algorithm SHA256` and run `vespasian.exe`. Select **Publish release** after checking the draft assets and release notes.

## Publish the crate

For the first publication, add a crates.io token as the `CARGO_REGISTRY_TOKEN` secret in the `crates-io` environment. Run **Publish crate** with the release tag and authentication set to `token`.

Configure [Trusted Publishing](https://crates.io/docs/trusted-publishing) in the crate settings with these values:

| Setting | Value |
| --- | --- |
| Repository owner | `chrisvoncsefalvay` |
| Repository | `vespasian` |
| Workflow | `publish-crate.yml` |
| Environment | `crates-io` |

Revoke the initial token and remove its GitHub secret. For later versions, select `trusted` in **Publish crate**. The workflow checks the tag, tests the package, runs a publication dry run, then publishes with a short-lived token from the [crates.io authentication action](https://github.com/rust-lang/crates-io-auth-action).

Confirm installation from the registry:

```sh
cargo install vespasian --locked --version 0.3.0
vespasian --version
```
