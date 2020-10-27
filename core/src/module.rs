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

use core::fmt;

/// Represents a successfully-parsed binary.
///
/// This is the equivalent of an [ELF](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format)
/// or a [PE](https://en.wikipedia.org/wiki/Portable_Executable).
pub struct Module {
    #[cfg(not(target_arch = "x86_64"))]
    inner: wasmi::Module,
    #[cfg(target_arch = "x86_64")]
    bytes: Vec<u8>,
    hash: ModuleHash,
}

/// Hash of a module.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ModuleHash([u8; 32]);

/// Error that can happen when calling [`ModuleHash::from_bytes`].
#[derive(Debug)]
pub struct FromBytesError {}

/// Error that can happen when calling [`ModuleHash::from_base58`].
#[derive(Debug)]
pub struct FromBase58Error {}

impl Module {
    /// Parses a module from WASM bytes.
    pub fn from_bytes(buffer: impl AsRef<[u8]>) -> Result<Self, FromBytesError> {
        let buffer = buffer.as_ref();
        let hash = ModuleHash::from_bytes(buffer);

        Ok(Module {
            #[cfg(not(target_arch = "x86_64"))]
            inner: wasmi::Module::from_buffer(buffer.as_ref()).map_err(|_| FromBytesError {})?,
            #[cfg(target_arch = "x86_64")]
            bytes: buffer.to_owned(),
            hash,
        })
    }

    /// Returns a reference to the internal module.
    #[cfg(not(target_arch = "x86_64"))]
    pub(crate) fn as_ref(&self) -> &wasmi::Module {
        &self.inner
    }

    /// Returns the Wasm binary.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn as_ref(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the hash of that module.
    ///
    /// This gives the same result as calling `ModuleHash::from_bytes` on the original input.
    pub fn hash(&self) -> &ModuleHash {
        &self.hash
    }
}

impl From<[u8; 32]> for ModuleHash {
    fn from(hash: [u8; 32]) -> ModuleHash {
        ModuleHash(hash)
    }
}

impl From<ModuleHash> for [u8; 32] {
    fn from(hash: ModuleHash) -> [u8; 32] {
        hash.0
    }
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Module({})", bs58::encode(&self.hash.0).into_string())
    }
}

impl ModuleHash {
    /// Returns the hash of the given bytes.
    pub fn from_bytes(buffer: impl AsRef<[u8]>) -> Self {
        ModuleHash(blake3::hash(buffer.as_ref()).into())
    }

    /// Decodes the given base58-encoded string into a hash.
    ///
    /// See also https://en.wikipedia.org/wiki/Base58.
    // TODO: check that we return an error if the string is too long
    pub fn from_base58(encoding: &str) -> Result<Self, FromBase58Error> {
        let mut out = [0; 32];
        let written = bs58::decode(encoding)
            .into(&mut out)
            .map_err(|_| FromBase58Error {})?;
        let mut out2 = [0; 32];
        out2[32 - written..].copy_from_slice(&out[..written]);
        Ok(ModuleHash(out2))
    }
}

impl fmt::Debug for ModuleHash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ModuleHash({})", bs58::encode(&self.0).into_string())
    }
}

impl fmt::Display for FromBase58Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FromBase58Error")
    }
}

impl fmt::Display for FromBytesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FromBytesError")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_wat_works() {
        let _ = from_wat!(local, "(module)");
    }

    #[test]
    fn simple_wat_works() {
        let _ = from_wat!(
            local,
            r#"
            (module
                (func $add (param i32 i32) (result i32)
                    get_local 0
                    get_local 1
                    i32.add)
                (export "add" (func $add)))
            "#
        );
    }
}
