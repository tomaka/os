// Copyright (C) 2019-2020  Pierre Krieger
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

/// Configuration for building the kernel.
#[derive(Debug)]
pub struct Config<'a> {
    /// Path to the `Cargo.toml` of the standalone kernel.
    pub kernel_cargo_toml: &'a Path,

    /// If true, compiles with `--release`.
    pub release: bool,

    /// Name of the target to pass as `--target`.
    pub target_name: &'a str,

    /// JSON target specifications.
    pub target_specs: &'a str,

    /// Link script to pass to the linker.
    pub link_script: &'a str,
}

/// Successful build.
#[derive(Debug)]
pub struct BuildOutput {
    /// Path to the output of the compilation.
    pub out_kernel_path: PathBuf,
}

/// Error that can happen during the build.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Could not start Cargo: {0}")]
    CargoNotFound(io::Error),

    #[error("Error while building the kernel")]
    BuildError,

    #[error("Failed to get metadata about the kernel Cargo.toml")]
    MetadataFailed,

    #[error("kernel_cargo_toml must not point to a workspace")]
    UnexpectedWorkspace,

    #[error("No binary target found at the kernel standalone path")]
    NoBinTarget,

    #[error("Multiple binary targets found")]
    MultipleBinTargets,

    #[error("{0}")]
    Io(#[from] io::Error),
}

/// Builds the kernel.
pub fn build(cfg: Config) -> Result<BuildOutput, Error> {
    assert_ne!(cfg.target_name, "debug");
    assert_ne!(cfg.target_name, "release");

    // Get the package ID of the package requested by the user.
    let pkg_id = {
        let output = Command::new("cargo")
            .arg("read-manifest")
            .arg("--manifest-path")
            .arg(cfg.kernel_cargo_toml)
            .output()
            .map_err(Error::CargoNotFound)?;
        if !output.status.success() {
            return Err(Error::MetadataFailed);
        }
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        json.as_object()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned()
    };

    // Determine the path to the file that Cargo will generate.
    let (output_file, target_dir_with_target, bin_target) = {
        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(&cfg.kernel_cargo_toml)
            .no_deps()
            .exec()
            .map_err(|_| Error::MetadataFailed)?;

        let package = metadata
            .packages
            .iter()
            .find(|p| p.id.repr == pkg_id)
            .unwrap();

        let bin_target = {
            let mut iter = package
                .targets
                .iter()
                .filter(|t| t.kind.iter().any(|k| k == "bin"));
            let target = iter.next().ok_or(Error::NoBinTarget)?;
            if iter.next().is_some() {
                return Err(Error::MultipleBinTargets);
            }
            target
        };

        let output_file = metadata
            .target_directory
            .join(cfg.target_name)
            .join(if cfg.release { "release" } else { "debug" })
            .join(bin_target.name.clone());

        let target_dir_with_target = metadata.target_directory.join(cfg.target_name);

        (output_file, target_dir_with_target, bin_target.name.clone())
    };

    // Create and fill the directory for the target specifications.
    fs::create_dir_all(&target_dir_with_target)?;
    fs::write(
        (&target_dir_with_target).join(format!("{}.json", cfg.target_name)),
        cfg.target_specs.as_bytes(),
    )?;
    fs::write(
        (&target_dir_with_target).join("link.ld"),
        cfg.link_script.as_bytes(),
    )?;

    // Actually build the kernel.
    let build_status = Command::new("cargo")
        .arg("build")
        .args(&["-Z", "build-std=core,alloc"]) // TODO: nightly only; cc https://github.com/tomaka/redshirt/issues/300
        .env("RUST_TARGET_PATH", &target_dir_with_target)
        .env(
            &format!(
                "CARGO_TARGET_{}_RUSTFLAGS",
                cfg.target_name.replace("-", "_").to_uppercase()
            ),
            format!(
                "-Clink-arg=--script -Clink-arg={}",
                target_dir_with_target.join("link.ld").display()
            ),
        )
        .arg("--bin")
        .arg(bin_target)
        .arg("--target")
        .arg(cfg.target_name)
        .arg("--manifest-path")
        .arg(cfg.kernel_cargo_toml)
        .args(if cfg.release {
            &["--release"][..]
        } else {
            &[][..]
        })
        .status()
        .map_err(Error::CargoNotFound)?;
    // TODO: should we make it configurable where the stdout/stderr outputs go?
    if !build_status.success() {
        return Err(Error::BuildError);
    }

    assert!(output_file.exists());

    Ok(BuildOutput {
        out_kernel_path: output_file,
    })
}
