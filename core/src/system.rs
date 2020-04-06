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

use crate::module::{Module, ModuleHash};
use crate::native::{self, NativeProgramMessageIdWrite as _};
use crate::scheduler::{Core, CoreBuilder, CoreRunOutcome, NewErr};

use alloc::vec::Vec;
use core::{cell::RefCell, iter, num::NonZeroU64, sync::atomic, task::Poll};
use crossbeam_queue::SegQueue;
use futures::prelude::*;
use hashbrown::HashSet;
use nohash_hasher::BuildNoHashHasher;
use redshirt_syscalls::{Decode, Encode, MessageId, Pid};

/// Main struct that handles a system, including the scheduler, program loader,
/// inter-process communication, and so on.
///
/// Natively handles the "interface" interface.  TODO: indicate hashes
pub struct System<'a> {
    /// Inner system with inter-process communications.
    core: Core,

    /// Collection of programs. Each is assigned a `Pid` that is reserved within `core`.
    /// Can communicate with the WASM programs that are within `core`.
    native_programs: native::NativeProgramsCollection<'a>,

    /// PID of the program that handles the `loader` interface, or `0` is no such program exists
    /// yet.
    // TODO: add timeout for loader interface availability?
    loader_pid: atomic::AtomicU64,

    /// List of programs to load if the loader interface handler is available.
    programs_to_load: SegQueue<ModuleHash>,

    /// "Virtual" pid for the process that sends messages towards the loader.
    load_source_virtual_pid: Pid,

    /// Set of messages that we emitted of requests to load a program from the loader interface.
    /// All these messages expect a `redshirt_loader_interface::ffi::LoadResponse` as answer.
    // TODO: call shink_to_fit from time to time
    loading_programs: RefCell<HashSet<MessageId, BuildNoHashHasher<u64>>>,
}

/// Prototype for a [`System`].
pub struct SystemBuilder<'a> {
    /// Builder for the inner core.
    core: CoreBuilder,

    /// Native programs.
    native_programs: native::NativeProgramsCollection<'a>,

    /// "Virtual" pid for handling messages on the `interface` interface.
    interface_interface_pid: Pid,

    /// "Virtual" pid for the process that sends messages towards the loader.
    load_source_virtual_pid: Pid,

    /// List of programs to start executing immediately after construction.
    startup_processes: Vec<Module>,

    /// Same field as [`System::programs_to_load`].
    programs_to_load: SegQueue<ModuleHash>,
}

/// Outcome of running the [`System`] once.
#[derive(Debug)]
pub enum SystemRunOutcome {
    /// A program has ended, either successfully or after an error.
    ProgramFinished {
        /// Identifier of the process that has stopped.
        pid: Pid,
        /// Either `Ok(())` if the main thread has ended, or the error that happened in the
        /// process.
        // TODO: change error type
        outcome: Result<(), wasmi::Error>,
    },
}

#[derive(Debug)]
enum RunOnceOutcome {
    Report(SystemRunOutcome),
    Idle,
    LoopAgain,
    LoopAgainNow,
}

impl<'a> System<'a> {
    /// Start executing a program.
    pub fn execute(&self, program: &Module) -> Result<Pid, NewErr> {
        Ok(self.core.execute(program)?.pid())
    }

    /// Runs the [`System`] once and returns the outcome.
    ///
    /// > **Note**: For now, it can a long time for this `Future` to be `Ready` because it is also
    /// >           waiting for the native programs to produce events in case there's nothing to
    /// >           do. In other words, this function can be seen more as a generator that whose
    /// >           `Future` becomes `Ready` only when something needs to be notified.
    pub fn run<'b>(&'b self) -> impl Future<Output = SystemRunOutcome> + 'b {
        // TODO: We use a `poll_fn` because async/await don't work in no_std yet.
        future::poll_fn(move |cx| {
            loop {
                // If we have a handler for the loader interface, start loading pending programs.
                if let Some(_) = NonZeroU64::new(self.loader_pid.load(atomic::Ordering::Relaxed)) {
                    while let Ok(hash) = self.programs_to_load.pop() {
                        // TODO: can this not fail if the handler crashed in parallel in a
                        // multithreaded situation?
                        let message_id = self.core.emit_interface_message_answer(
                            self.load_source_virtual_pid,
                            redshirt_loader_interface::ffi::INTERFACE,
                            redshirt_loader_interface::ffi::LoaderMessage::Load(From::from(hash)),
                        );
                        self.loading_programs.borrow_mut().insert(message_id);
                    }
                }

                let run_once_outcome = self.run_once();

                if let RunOnceOutcome::Report(out) = run_once_outcome {
                    return Poll::Ready(out);
                }

                if let RunOnceOutcome::LoopAgainNow = run_once_outcome {
                    continue;
                }

                let next_event = self.native_programs.next_event();
                futures::pin_mut!(next_event);
                let event = match next_event.poll(cx) {
                    Poll::Ready(ev) => ev,
                    Poll::Pending => {
                        if let RunOnceOutcome::LoopAgain = run_once_outcome {
                            continue;
                        }
                        return Poll::Pending;
                    }
                };

                match event {
                    native::NativeProgramsCollectionEvent::Emit {
                        interface,
                        emitter_pid,
                        message,
                        message_id_write,
                    } => {
                        // The native programs want to emit a message in the kernel.
                        if let Some(message_id_write) = message_id_write {
                            let message_id = self.core.emit_interface_message_answer(
                                emitter_pid,
                                interface,
                                message,
                            );
                            message_id_write.acknowledge(message_id);
                        } else {
                            self.core.emit_interface_message_no_answer(
                                emitter_pid,
                                interface,
                                message,
                            );
                        }
                    }
                    native::NativeProgramsCollectionEvent::CancelMessage { message_id } => {
                        // The native programs want to cancel a previously-emitted message.
                        self.core.cancel_message(message_id);
                    }
                    native::NativeProgramsCollectionEvent::Answer { message_id, answer } => {
                        self.core.answer_message(message_id, answer);
                    }
                }
            }
        })
    }

    fn run_once(&self) -> RunOnceOutcome {
        match self.core.run() {
            CoreRunOutcome::Idle => return RunOnceOutcome::Idle,

            CoreRunOutcome::ProgramFinished { pid, outcome, .. } => {
                self.loader_pid
                    .compare_and_swap(u64::from(pid), 0, atomic::Ordering::AcqRel);
                self.native_programs.process_destroyed(pid);
                return RunOnceOutcome::Report(SystemRunOutcome::ProgramFinished {
                    pid,
                    outcome: outcome.map(|_| ()).map_err(|err| err.into()),
                });
            }

            CoreRunOutcome::ThreadWaitUnavailableInterface { .. } => {} // TODO: lazy-loading

            CoreRunOutcome::MessageResponse {
                message_id,
                response,
                ..
            } => {
                if self.loading_programs.borrow_mut().remove(&message_id) {
                    let redshirt_loader_interface::ffi::LoadResponse { result } =
                        Decode::decode(response.unwrap()).unwrap();
                    // TODO: don't unwrap
                    let module = Module::from_bytes(&result.expect("loader returned error"))
                        .expect("module isn't proper wasm");
                    match self.core.execute(&module) {
                        Ok(_) => {}
                        Err(_) => panic!(),
                    }
                } else {
                    self.native_programs.message_response(message_id, response);
                }
            }

            CoreRunOutcome::ReservedPidInterfaceMessage {
                pid,
                message_id,
                interface,
                message,
            } if interface == redshirt_interface_interface::ffi::INTERFACE => {
                // Handling messages on the `interface` interface.
                match redshirt_interface_interface::ffi::InterfaceMessage::decode(message) {
                    Ok(redshirt_interface_interface::ffi::InterfaceMessage::Register(
                        interface_hash,
                    )) => {
                        // Set the process as interface handler, if possible.
                        let result = self.core.set_interface_handler(interface_hash.clone(), pid);
                        let response =
                            redshirt_interface_interface::ffi::InterfaceRegisterResponse {
                                result: result.clone().map_err(|()| redshirt_interface_interface::ffi::InterfaceRegisterError::AlreadyRegistered),
                            };
                        if let Some(message_id) = message_id {
                            self.core.answer_message(message_id, Ok(response.encode()));
                        }

                        // Special handling if the registered interface is the loader.
                        if result.is_ok()
                            && interface_hash == redshirt_loader_interface::ffi::INTERFACE
                        {
                            debug_assert_ne!(u64::from(pid), 0);
                            self.loader_pid
                                .swap(u64::from(pid), atomic::Ordering::AcqRel);
                            return RunOnceOutcome::LoopAgainNow;
                        }
                    }
                    Err(_) => {
                        if let Some(message_id) = message_id {
                            self.core.answer_message(message_id, Err(()));
                        }
                    }
                }
            }

            CoreRunOutcome::ReservedPidInterfaceMessage {
                pid,
                message_id,
                interface,
                message,
            } => {
                self.native_programs
                    .interface_message(interface, message_id, pid, message);
            }
        }

        RunOnceOutcome::LoopAgain
    }
}

impl<'a> SystemBuilder<'a> {
    /// Starts a new builder.
    pub fn new() -> Self {
        // We handle some low-level interfaces here.
        let mut core = Core::new();
        let interface_interface_pid = core.reserve_pid();
        let load_source_virtual_pid = core.reserve_pid();

        SystemBuilder {
            core,
            interface_interface_pid,
            load_source_virtual_pid,
            startup_processes: Vec::new(),
            programs_to_load: SegQueue::new(),
            native_programs: native::NativeProgramsCollection::new(),
        }
    }

    /// Registers native code that can communicate with the WASM programs.
    pub fn with_native_program<T>(mut self, program: T) -> Self
    where
        T: Send + 'a,
        for<'r> &'r T: native::NativeProgramRef<'r>,
    {
        self.native_programs.push(self.core.reserve_pid(), program);
        self
    }

    /// Adds a process to the list of processes that the [`System`] must start as part of the
    /// startup process.
    ///
    /// > **Note**: The startup processes are started in the order in which they are added here,
    /// >           but you should not rely on this fact for making the system work.
    ///
    /// By default, the list is empty. Should at least contain a process that handles the `loader`
    /// interface.
    pub fn with_startup_process(mut self, process: impl Into<Module>) -> Self {
        let process = process.into();
        self.startup_processes.push(process);
        self
    }

    /// Shortcut for calling [`with_main_program`](SystemBuilder::with_main_program) multiple
    /// times.
    pub fn with_main_programs(self, hashes: impl IntoIterator<Item = ModuleHash>) -> Self {
        for hash in hashes {
            self.programs_to_load.push(hash);
        }
        self
    }

    /// Adds a program that the [`System`] must execute after startup. Can be called multiple times
    /// to add multiple programs.
    ///
    /// The program will be loaded through the `loader` interface. The loading starts as soon as
    /// the `loader` interface has been registered by one of the processes passed to
    /// [`with_startup_process`](SystemBuilder::with_startup_process).
    ///
    /// Messages are sent to the `loader` interface in the order in which this function has been
    /// called. The implementation of `loader`, however, might not deliver the responses in the
    /// same order.
    pub fn with_main_program(self, hash: ModuleHash) -> Self {
        self.with_main_programs(iter::once(hash))
    }

    /// Builds the [`System`].
    ///
    /// Returns an error if any of the programs passed through
    /// [`SystemBuilder::with_startup_process`] fails to start.
    pub fn build(self) -> Result<System<'a>, NewErr> {
        let core = self.core.build();

        // We ask the core to redirect messages for the `interface` interface towards our
        // "virtual" `Pid`.
        match core.set_interface_handler(
            redshirt_interface_interface::ffi::INTERFACE,
            self.interface_interface_pid,
        ) {
            Ok(()) => {}
            Err(_) => unreachable!(),
        };

        for program in self.startup_processes {
            core.execute(&program)?;
        }

        Ok(System {
            core,
            native_programs: self.native_programs,
            loader_pid: atomic::AtomicU64::new(0),
            load_source_virtual_pid: self.load_source_virtual_pid,
            loading_programs: RefCell::new(Default::default()),
            programs_to_load: self.programs_to_load,
        })
    }
}

impl<'a> Default for SystemBuilder<'a> {
    fn default() -> Self {
        SystemBuilder::new()
    }
}
