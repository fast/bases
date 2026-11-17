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

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use clap::Subcommand;

const PACKAGE_NAME: &str = "bases";

const CHECK_FEATURE_CONFIGS: &[&[&str]] = &[
    &["--no-default-features"],
    &["--no-default-features", "--features", "alloc"],
    &["--no-default-features", "--features", "std"],
    &["--no-default-features", "--features", "serde"],
    &["--no-default-features", "--features", "alloc,serde"],
    &["--all-features"],
];

const TEST_CONFIGS: &[&[&str]] = &[
    &["--workspace", "--all-features"],
    &["--package", PACKAGE_NAME, "--no-default-features"],
    &[
        "--package",
        PACKAGE_NAME,
        "--no-default-features",
        "--features",
        "alloc,serde",
    ],
    &[
        "--package",
        PACKAGE_NAME,
        "--no-default-features",
        "--features",
        "serde",
    ],
    &[
        "--package",
        "tests-integration",
        "--release",
        "--all-features",
    ],
];

#[derive(Parser)]
struct Command {
    #[command(subcommand)]
    sub: SubCommand,
}

impl Command {
    fn run(self) {
        match self.sub {
            SubCommand::Bench(cmd) => cmd.run(),
            SubCommand::Build(cmd) => cmd.run(),
            SubCommand::Check(cmd) => cmd.run(),
            SubCommand::Lint(cmd) => cmd.run(),
            SubCommand::Package(cmd) => cmd.run(),
            SubCommand::Test(cmd) => cmd.run(),
        }
    }
}

#[derive(Subcommand)]
enum SubCommand {
    #[command(about = "Run workspace benchmarks.")]
    Bench(CommandBench),
    #[command(about = "Compile workspace packages.")]
    Build(CommandBuild),
    #[command(about = "Check bases under its feature configurations.")]
    Check(CommandCheck),
    #[command(about = "Run workspace quality checks.")]
    Lint(CommandLint),
    #[command(about = "Build, inspect, and test the bases release package.")]
    Package(CommandPackage),
    #[command(about = "Run workspace tests.")]
    Test(CommandTest),
}

#[derive(Parser)]
struct CommandBench {
    #[arg(long, help = "Compile benchmarks without running them.")]
    no_run: bool,
    #[arg(long, requires = "no_run", help = "Emit Cargo JSON build metadata.")]
    json: bool,
    #[arg(
        last = true,
        conflicts_with = "no_run",
        help = "Arguments for the benchmark runner."
    )]
    args: Vec<String>,
}

impl CommandBench {
    fn run(self) {
        let mut cmd = cargo();
        cmd.args(["bench", "--package", "benchmarks", "--bench", "throughput"]);
        if self.no_run {
            cmd.arg("--no-run");
        }
        if self.json {
            cmd.arg("--message-format=json-render-diagnostics");
        }
        if !self.args.is_empty() {
            cmd.arg("--").args(self.args);
        }
        run_command(cmd);
    }
}

#[derive(Parser)]
struct CommandBuild {
    #[arg(long, help = "Assert that `Cargo.lock` will remain unchanged.")]
    locked: bool,
}

impl CommandBuild {
    fn run(self) {
        let mut cmd = cargo();
        cmd.args([
            "build",
            "--workspace",
            "--all-features",
            "--tests",
            "--examples",
            "--benches",
            "--bins",
        ]);
        if self.locked {
            cmd.arg("--locked");
        }
        run_command(cmd);
    }
}

#[derive(Parser)]
struct CommandCheck;

impl CommandCheck {
    fn run(self) {
        for args in CHECK_FEATURE_CONFIGS {
            run_command(make_check_cmd(args));
        }
    }
}

#[derive(Parser)]
struct CommandTest {
    #[arg(long, help = "Run tests serially and do not capture output.")]
    no_capture: bool,
}

impl CommandTest {
    fn run(self) {
        for args in TEST_CONFIGS {
            let mut cmd = cargo();
            cmd.arg("test").args(*args);
            add_test_output_args(&mut cmd, self.no_capture);
            run_command(cmd);
        }
    }
}

#[derive(Parser)]
#[command(name = "lint")]
struct CommandLint {
    #[arg(long, help = "Automatically apply lint and formatting suggestions.")]
    fix: bool,
}

impl CommandLint {
    fn run(self) {
        run_command(make_clippy_cmd(self.fix));
        run_command(make_format_cmd(self.fix));
        run_command(make_taplo_cmd(self.fix));
        run_command(make_typos_cmd());
        run_command(make_hawkeye_cmd(self.fix));
        run_command(make_doc_cmd());
    }
}

#[derive(Parser)]
struct CommandPackage {
    #[arg(
        long,
        help = "Allow packaging uncommitted changes for local validation."
    )]
    allow_dirty: bool,
}

impl CommandPackage {
    fn run(self) {
        let workspace = Path::new(env!("CARGO_WORKSPACE_DIR"));
        let mut metadata_cmd = cargo();
        metadata_cmd.args(["metadata", "--format-version", "1", "--no-deps", "--locked"]);
        let metadata: serde_json::Value = serde_json::from_str(&command_output(metadata_cmd))
            .expect("Cargo metadata must be valid JSON");
        let package = metadata["packages"]
            .as_array()
            .expect("Cargo metadata must contain packages")
            .iter()
            .find(|package| package["name"].as_str() == Some(PACKAGE_NAME))
            .expect("bases must be a workspace package");
        let version = package["version"]
            .as_str()
            .expect("package version missing");
        let package_dir = Path::new(
            metadata["target_directory"]
                .as_str()
                .expect("Cargo target directory missing"),
        )
        .join("package")
        .join(format!("{PACKAGE_NAME}-{version}"));

        let mut documents: Vec<PathBuf> = [
            "README.md",
            "ARCHITECTURE.md",
            "CHANGELOG.md",
            "benchmarks/README.md",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect();
        for entry in fs::read_dir(workspace.join("examples/src")).expect("cannot read examples") {
            let path = entry.expect("cannot read example entry").path();
            if path.extension().is_some_and(|extension| extension == "rs") {
                documents.push(path.strip_prefix(workspace).unwrap().to_path_buf());
            }
        }
        documents.sort();

        let mut list_cmd = cargo();
        list_cmd.args(["package", "--package", PACKAGE_NAME, "--locked", "--list"]);
        if self.allow_dirty {
            list_cmd.arg("--allow-dirty");
        }
        let files: BTreeSet<PathBuf> = command_output(list_cmd)
            .lines()
            .map(PathBuf::from)
            .collect();
        for required in ["Cargo.lock", "Cargo.toml", "LICENSE", "src/lib.rs"] {
            assert!(
                files.contains(Path::new(required)),
                "missing package file: {required}"
            );
        }
        for document in &documents {
            assert!(
                files.contains(document),
                "missing package document: {}",
                document.display()
            );
        }
        for file in &files {
            let development_only = [
                "benchmarks",
                "examples",
                "tests-integration",
                "xtask",
                "target",
            ]
            .iter()
            .any(|directory| file.starts_with(directory));
            assert!(
                !development_only || documents.contains(file),
                "development-only file in package: {}",
                file.display()
            );
        }

        let mut package_cmd = cargo();
        package_cmd.args(["package", "--package", PACKAGE_NAME, "--locked"]);
        if self.allow_dirty {
            package_cmd.arg("--allow-dirty");
        }
        run_command(package_cmd);

        for document in &documents {
            let path = package_dir.join(document);
            let metadata = fs::symlink_metadata(&path).expect("cannot inspect packaged document");
            assert!(
                metadata.is_file(),
                "expected a regular file: {}",
                path.display()
            );
            assert!(
                fs::read(&path).expect("cannot read packaged document")
                    == fs::read(workspace.join(document)).expect("cannot read source document"),
                "packaged document differs from its source: {}",
                document.display()
            );
        }

        let mut test_cmd = cargo();
        test_cmd
            .arg("test")
            .arg("--manifest-path")
            .arg(package_dir.join("Cargo.toml"));
        test_cmd.args(["--all-features", "--locked"]);
        run_command(test_cmd);
    }
}

fn command_output(mut cmd: process::Command) -> String {
    println!("{cmd:?}");
    let output = cmd
        .stderr(process::Stdio::inherit())
        .output()
        .expect("failed to execute process");
    assert!(output.status.success(), "command failed: {}", output.status);
    String::from_utf8(output.stdout).expect("command output must be UTF-8")
}

fn find_command(command: &str) -> process::Command {
    match which::which(command) {
        Ok(executable) => {
            let mut cmd = process::Command::new(executable);
            cmd.current_dir(Path::new(env!("CARGO_WORKSPACE_DIR")));
            cmd
        }
        Err(err) => panic!("{command} not found: {err}"),
    }
}

fn ensure_installed(binary: &str, crate_name: &str) {
    if which::which(binary).is_err() {
        let mut cmd = cargo();
        cmd.args(["install", crate_name]);
        run_command(cmd);
    }
}

fn cargo() -> process::Command {
    find_command("cargo")
}

fn run_command(mut cmd: process::Command) {
    println!("{cmd:?}");
    let status = cmd.status().expect("failed to execute process");
    assert!(status.success(), "command failed: {status}");
}

fn add_test_output_args(cmd: &mut process::Command, no_capture: bool) {
    if no_capture {
        cmd.args(["--", "--nocapture", "--test-threads=1"]);
    }
}

fn make_check_cmd(extra_args: &[&str]) -> process::Command {
    let mut cmd = cargo();
    cmd.env("RUSTFLAGS", "-D warnings");
    cmd.args(["check", "--package", PACKAGE_NAME, "--all-targets"]);
    cmd.args(extra_args);
    cmd
}

fn make_format_cmd(fix: bool) -> process::Command {
    let mut cmd = cargo();
    cmd.args(["+nightly", "fmt", "--all"]);
    if !fix {
        cmd.args(["--", "--check"]);
    }
    cmd
}

fn make_clippy_cmd(fix: bool) -> process::Command {
    let mut cmd = cargo();
    cmd.args([
        "+nightly",
        "clippy",
        "--workspace",
        "--all-targets",
        "--all-features",
    ]);
    if fix {
        cmd.args(["--allow-staged", "--allow-dirty", "--fix"]);
    } else {
        cmd.args(["--", "-D", "warnings"]);
    }
    cmd
}

fn make_hawkeye_cmd(fix: bool) -> process::Command {
    ensure_installed("hawkeye", "hawkeye");
    let mut cmd = find_command("hawkeye");
    if fix {
        cmd.arg("format");
    } else {
        cmd.arg("check");
    }
    cmd
}

fn make_typos_cmd() -> process::Command {
    ensure_installed("typos", "typos-cli");
    find_command("typos")
}

fn make_taplo_cmd(fix: bool) -> process::Command {
    ensure_installed("taplo", "taplo-cli");
    let mut cmd = find_command("taplo");
    if fix {
        cmd.arg("format");
    } else {
        cmd.args(["format", "--check"]);
    }
    cmd
}

fn make_doc_cmd() -> process::Command {
    let mut cmd = cargo();
    cmd.env("RUSTDOCFLAGS", "-D warnings");
    cmd.args([
        "doc",
        "--package",
        PACKAGE_NAME,
        "--all-features",
        "--no-deps",
    ]);
    cmd
}

fn main() {
    Command::parse().run();
}
