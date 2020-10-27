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

//! Core components of a redshirt kernel.
//!
//! The main structure of this crate is [`System`]. A [`System`] represents a running operating
//! system, and is a collection of *programs* exchanging messages between each others.
//!
//! > **Note**: A [`System`] doesn't run automatically in the background. You must call the
//! >           [`System::run`] function repeatedly in order for the programs within the
//! >           [`System`] to advance.
//!
//! There exists two kinds of programs within a [`System`]:
//!
//! - *Wasm* programs. They are written in [the Wasm language](https://webassembly.org/), or, more
//! likely, written in a programming language then compiled to Wasm. In order to start a Wasm
//! program, build a [`Module`] then pass it either to [`SystemBuilder::with_startup_process`]
//! if you're building a [`System`], or to [`System::execute`] if the [`System`] is already
//! constructed.
//!
//! - *Native* programs. They are directly written in Rust. In order to start a native program,
//! you must pass an object that implements the [`NativeProgramRef`](native::NativeProgramRef)
//! trait to [`SystemBuilder::with_native_program`] when building the [`System`].
//!
//! Each program within a [`System`] gets attributed a single [`Pid`] that identifies it.
//!
//! # Messages
//!
//! As part of their execution, programs (both Wasm and native) can emit *messages*. A message can
//! accept either zero or one response. Each emitted message gets attribute a single unique
//! [`MessageId`] that identifies it.
//!
//! In order to emit a message, you must pass three main information:
//!
//! - The hash of a target *interface* (more on that below).
//! - The body of the message, opaque to the `redshirt_core` crate. The way it must be interpreted
//! depends on the target *interface*.
//! - Whether or not a response is expected.
//!
//! Contrary to many other operating systems, messages don't target a specific program, but rather
//! an *interface* that can be referred to with an [`InterfaceHash`].
//!
//! There exists a few *interfaces* hardcoded within the [`System`]. When a program emits a
//! message targetting one of these interfaces, the message will be treated (and answer, if
//! necessary) by the [`System`] itself. Here is a list:
//!
//! - `interface`. The interface named `interface` allows programs to register themselves as
//! provider of an interface. If a program then emits a message targetting the interface, then
//! the registered program will be in charge of treating the message.
//! - `threads`. The interface named `threads` provides a few utilities related to multithreading
//! (TODO: this isn't really done yet)
//!
//! > **Note**: A very common workflow for a program is, immediately after it starts, to emit a
//! >           message on the `interface` interface in order to register itself as the handler of
//! >           a specific interface. It will then be in charge of processing the messages coming
//! >           on that registered interface.
//!
//! > **Note**: Only one program at a time can be registered as an interface handler. This is done
//! >           in a first-come-first-serve manner. If a second program tries to register itself
//! >           for the same interface, the second registration will fail.
//!
//! # Wasm programs isolation
//!
//! Wasm programs are isolated entirely within their virtual machine, and have no access to the
//! outside except for passing messages around.
//! Any action requiring intervention from the hardware can only be done directly by a
//! *native program*.
//!
//! Because of their non-isolated nature, the list of *native programs* should be composed only
//! of the strict minimum, and can't be changed once the [`System`] has been constructed.
//!
//! # Lazy interfaces registration
//!
//! Since programs all start simultaneously at the system initialization, and because we don't
//! know in advance which interface(s) (if any) a program will register and when, it is possible
//! for a program to emit a message on an interface that has no registered handler but that will
//! have one soon in the future.
//!
//! > **Example**: Program A is an HTTP server and wants to open a TCP socket to start listening
//! >              for incoming connections. To do so, program A emits a message of the interface
//! >              named `tcp`. Program B is the network manager and registers itself as the
//! >              provider of the `tcp` interface. Since A and B start at the same time, it is
//! >              possible for A to emit its message before B has registered itself.
//!
//! In order to solve this problem, emitting a message will block the execution of the current
//! thread until a handler is available for the target interface. It is possible, when emitting
//! a message, to disable this behaviour and fail immediately if no handler is registered.
//!
//! Additionally, no timeout mechanism exists. In other words, if no program registers itself as
//! the handler of an interface for which a message has been emitted, then the sending thread
//! will block forever.
//!
//! > **Note**: As a general rule in IT, the only two timeout values that make sense are *0*
//! >           and *infinite*.
//!
//! While this hasn't been implemented yet, the best way to deal with this kind of situation is
//! to somehow report to the user the list of programs being stuck waiting for an interface
//! handler.
//!

#![feature(asm, global_asm, naked_functions)]
#![feature(new_uninit)] // TODO: no; definitely can be circumvented too
#![warn(missing_docs)]
//#![deny(unsafe_code)] // TODO: 🤷
#![allow(dead_code)] // TODO: temporary during development

// The crate uses the stdlib for testing purposes.
// TODO: restore no_std
//#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub use self::module::Module;
pub use self::system::{System, SystemBuilder, SystemRunOutcome};
pub use primitives::{ValueType, WasmValue};
pub use redshirt_syscalls::{
    Decode, Encode, EncodedMessage, InterfaceHash, InvalidMessageIdErr, MessageId, Pid, ThreadId,
};

/// Compiles a WASM module and includes it similar to `include_bytes!`.
///
/// Must be passed the path to a directory containing a `Cargo.toml`.
/// Can be passed an optional second argument containing the binary name to compile. Mandatory if
/// the package contains multiple binaries.
#[cfg(feature = "nightly")]
#[cfg_attr(docsrs, doc(cfg(feature = "nightly")))]
// TODO: enable unconditonally after https://github.com/rust-lang/rust/issues/43781
pub use redshirt_core_proc_macros::build_wasm_module;

#[doc(hidden)]
pub use redshirt_core_proc_macros::wat_to_bin;

/// Builds a [`Module`](module::Module) from a WASM text representation.
///
/// The WASM text representation is parsed and transformed at compile time.
#[macro_export]
macro_rules! from_wat {
    // TODO: also build the hash at compile-time? https://github.com/tomaka/redshirt/issues/218
    // TODO: we need this hack with a special `local` tag because of macro paths resolution issues
    (local, $wat:expr) => {{
        $crate::Module::from_bytes(redshirt_core_proc_macros::wat_to_bin!($wat)).unwrap()
    }};
    ($wat:expr) => {{
        $crate::Module::from_bytes($crate::wat_to_bin!($wat)).unwrap()
    }};
}

mod id_pool;

pub mod extrinsics;
pub mod module;
pub mod native;
pub mod primitives;
pub mod scheduler;
pub mod system;
