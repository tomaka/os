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

use crate::module::Module;
use crate::native::{self, NativeProgramMessageIdWrite as _};
use crate::scheduler::{Core, CoreBuilder, CoreRunOutcome};
use alloc::{vec, vec::Vec};
use core::{cell::RefCell, task::Poll};
use crossbeam_queue::SegQueue;
use fnv::FnvBuildHasher;
use futures::prelude::*;
use hashbrown::{hash_map::Entry, HashMap, HashSet};
use nohash_hasher::BuildNoHashHasher;
use redshirt_syscalls::{Decode, Encode, EncodedMessage, MessageId, Pid};
use smallvec::SmallVec;

/// Main struct that handles a system, including the scheduler, program loader,
/// inter-process communication, and so on.
///
/// Natively handles the "interface" and "threads" interfaces.  TODO: indicate hashes
pub struct System {
    /// Inner system with inter-process communications.
    core: Core,

    /// List of active futexes. The keys of this hashmap are process IDs and memory addresses, and
    /// the values of this hashmap are a list of "wait" messages to answer once the corresponding
    /// futex is woken up.
    ///
    /// Lists of messages must never be empty.
    ///
    /// Messages are always pushed at the back of the list. Therefore the first element is the
    /// oldest message.
    ///
    /// See the "threads" interface for documentation about what a futex is.
    futex_waits: RefCell<HashMap<(Pid, u32), SmallVec<[MessageId; 4]>, FnvBuildHasher>>,

    /// Collection of programs. Each is assigned a `Pid` that is reserved within `core`.
    /// Can communicate with the WASM programs that are within `core`.
    native_programs: native::NativeProgramsCollection<'static>,

    /// List of programs to load as soon as a loader interface handler is available.
    ///
    /// As soon as a handler for the "loader" interface is registered, we start loading the
    /// programs in this list. Afterwards, the list will always be empty.
    ///
    /// This list is only filled at initialization and then never pushed again.
    // TODO: add timeout for loader interface availability
    main_programs: SegQueue<[u8; 32]>,

    /// "Virtual" pid for the process that sends messages towards the loader.
    loading_virtual_pid: Pid,

    /// Set of messages that we emitted of requests to load a program from the loader interface.
    /// All these messages expect a `redshirt_loader_interface::ffi::LoadResponse` as answer.
    // TODO: call shink_to_fit from time to time
    loading_programs: RefCell<HashSet<MessageId, BuildNoHashHasher<u64>>>,
}

/// Prototype for a [`System`].
pub struct SystemBuilder {
    /// Builder for the inner core.
    core: CoreBuilder,

    /// Native programs.
    native_programs: native::NativeProgramsCollection<'static>,

    /// "Virtual" pid for handling messages on the `interface` interface.
    interface_interface_pid: Pid,

    /// "Virtual" pid for handling messages on the `threads` interface.
    threads_interface_pid: Pid,

    /// "Virtual" pid for the process that sends messages towards the loader.
    loading_virtual_pid: Pid,

    /// List of programs to start executing immediately after construction.
    startup_processes: Vec<Module>,

    /// Same field as [`System::main_programs`].
    main_programs: SegQueue<[u8; 32]>,
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

impl System {
    /// Start executing a program.
    pub fn execute(&self, program: &Module) -> Pid {
        self.core
            .execute(program)
            .expect("failed to start startup program")
            .pid() // TODO: don't unwrap
    }

    /// Runs the [`System`] once and returns the outcome.
    ///
    /// > **Note**: For now, can block a long time because it's waiting for the native programs
    /// >           produce events in case there's nothing to do. In other words, this function
    /// >           can be seen as a generator that returns only when something needs to be
    /// >           notified.
    pub fn run<'b>(&'b self) -> impl Future<Output = SystemRunOutcome> + 'b {
        // TODO: We use a `poll_fn` because async/await don't work in no_std yet.
        future::poll_fn(move |cx| loop {
            if let Some(out) = self.run_once() {
                return Poll::Ready(out);
            }

            let next_event = self.native_programs.next_event();
            futures::pin_mut!(next_event);
            let event = match next_event.poll(cx) {
                Poll::Ready(ev) => ev,
                Poll::Pending => return Poll::Pending,
            };

            match event {
                native::NativeProgramsCollectionEvent::Emit {
                    interface,
                    emitter_pid,
                    message,
                    message_id_write,
                } => {
                    if let Some(message_id_write) = message_id_write {
                        let message_id = self.core.emit_interface_message_answer(
                            emitter_pid,
                            interface,
                            message,
                        );
                        message_id_write.acknowledge(message_id);
                    } else {
                        self.core
                            .emit_interface_message_no_answer(emitter_pid, interface, message);
                    }
                }
                native::NativeProgramsCollectionEvent::CancelMessage { .. } => unimplemented!(),
                native::NativeProgramsCollectionEvent::Answer { message_id, answer } => {
                    self.core.answer_message(message_id, answer);
                }
            }
        })
    }

    fn run_once(&self) -> Option<SystemRunOutcome> {
        // TODO: remove loop?
        loop {
            match self.core.run() {
                CoreRunOutcome::ProgramFinished { pid, outcome, .. } => {
                    self.native_programs.process_destroyed(pid);
                    return Some(SystemRunOutcome::ProgramFinished {
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
                        let module = Module::from_bytes(&result.unwrap()).unwrap();
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
                } if interface == redshirt_threads_interface::ffi::INTERFACE => {
                    let msg: redshirt_threads_interface::ffi::ThreadsMessage =
                        Decode::decode(message).unwrap();
                    match msg {
                        redshirt_threads_interface::ffi::ThreadsMessage::New(new_thread) => {
                            assert!(message_id.is_none());
                            self.core
                                .process_by_id(pid)
                                .unwrap()
                                .start_thread(
                                    new_thread.fn_ptr,
                                    vec![wasmi::RuntimeValue::I32(new_thread.user_data as i32)],
                                )
                                .unwrap();
                        }
                        redshirt_threads_interface::ffi::ThreadsMessage::FutexWake(mut wake) => {
                            assert!(message_id.is_none());
                            let mut futex_waits = self.futex_waits.borrow_mut();
                            if let Some(list) = futex_waits.get_mut(&(pid, wake.addr)) {
                                while wake.nwake > 0 && !list.is_empty() {
                                    wake.nwake -= 1;
                                    let message_id = list.remove(0);
                                    self.core
                                        .answer_message(message_id, Ok(EncodedMessage(Vec::new())));
                                }

                                if list.is_empty() {
                                    futex_waits.remove(&(pid, wake.addr));
                                }
                            }
                            // TODO: implement
                        }
                        redshirt_threads_interface::ffi::ThreadsMessage::FutexWait(wait) => {
                            if let Some(message_id) = message_id {
                                // TODO: val_cmp
                                match self.futex_waits.borrow_mut().entry((pid, wait.addr)) {
                                    Entry::Occupied(mut e) => e.get_mut().push(message_id),
                                    Entry::Vacant(e) => {
                                        e.insert({
                                            let mut sv = SmallVec::new();
                                            sv.push(message_id);
                                            sv
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                CoreRunOutcome::ReservedPidInterfaceMessage {
                    pid,
                    message_id,
                    interface,
                    message,
                } if interface == redshirt_interface_interface::ffi::INTERFACE => {
                    let msg =
                        redshirt_interface_interface::ffi::InterfaceMessage::decode(message).ok();
                    match msg {
                        Some(redshirt_interface_interface::ffi::InterfaceMessage::Register(
                            interface_hash,
                        )) => {
                            let result = self.core
                                .set_interface_handler(interface_hash.clone(), pid)
                                .map_err(|()| redshirt_interface_interface::ffi::InterfaceRegisterError::AlreadyRegistered);
                            let response =
                                redshirt_interface_interface::ffi::InterfaceRegisterResponse {
                                    result,
                                };
                            if let Some(message_id) = message_id {
                                self.core.answer_message(message_id, Ok(response.encode()));
                            }

                            if interface_hash == redshirt_loader_interface::ffi::INTERFACE {
                                while let Ok(hash) = self.main_programs.pop() {
                                    let msg =
                                        redshirt_loader_interface::ffi::LoaderMessage::Load(hash);
                                    let id = self.core.emit_interface_message_answer(
                                        self.loading_virtual_pid,
                                        redshirt_loader_interface::ffi::INTERFACE,
                                        msg,
                                    );
                                    self.loading_programs.borrow_mut().insert(id);
                                }
                            }
                        }
                        None => {}
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

                CoreRunOutcome::Idle => return None,
            }
        }
    }
}

impl SystemBuilder {
    /// Starts a new builder.
    pub fn new() -> Self {
        // We handle some low-level interfaces here.
        let mut core = Core::new();
        let interface_interface_pid = core.reserve_pid();
        let threads_interface_pid = core.reserve_pid();
        let loading_virtual_pid = core.reserve_pid();

        SystemBuilder {
            core,
            interface_interface_pid,
            threads_interface_pid,
            loading_virtual_pid,
            startup_processes: Vec::new(),
            main_programs: SegQueue::new(),
            native_programs: native::NativeProgramsCollection::new(),
        }
    }

    /// Registers native code that can communicate with the WASM programs.
    pub fn with_native_program<T>(mut self, program: T) -> Self
    where
        T: Send + 'static,
        for<'r> &'r T: native::NativeProgramRef<'r>,
    {
        self.native_programs.push(self.core.reserve_pid(), program);
        self
    }

    /// Adds a process to the list of processes that the [`System`] must start as part of the
    /// startup process.
    ///
    /// The startup processes are started in the order in which they are added here.
    ///
    /// By default, the list is empty. Should at least contain a process that handles the `loader`
    /// interface.
    pub fn with_startup_process(mut self, process: impl Into<Module>) -> Self {
        let process = process.into();
        self.startup_processes.push(process);
        self
    }

    /// Adds a program that the [`System`] must execute after startup. Can be called multiple times
    /// to add multiple programs.
    ///
    /// The program will be loaded through the `loader` interface. The loading starts as soon as
    /// the `loader` interface has been registered by one of the processes passed to
    /// [`with_startup_process`](SystemBuilder::with_startup_process).
    pub fn with_main_program(self, hash: [u8; 32]) -> Self {
        self.main_programs.push(hash);
        self
    }

    /// Builds the [`System`].
    pub fn build(self) -> System {
        let core = self.core.build();

        // We ask the core to redirect messages for the `interface` and `threads` interfaces
        // towards our "virtual" `Pid`s.
        match core.set_interface_handler(
            redshirt_interface_interface::ffi::INTERFACE,
            self.interface_interface_pid,
        ) {
            Ok(()) => {}
            Err(_) => unreachable!(),
        };
        match core.set_interface_handler(
            redshirt_threads_interface::ffi::INTERFACE,
            self.threads_interface_pid,
        ) {
            Ok(()) => {}
            Err(_) => unreachable!(),
        };

        for program in self.startup_processes {
            core.execute(&program)
                .expect("failed to start startup program"); // TODO:
        }

        System {
            core,
            native_programs: self.native_programs,
            futex_waits: RefCell::new(Default::default()),
            loading_virtual_pid: self.loading_virtual_pid,
            loading_programs: RefCell::new(Default::default()),
            main_programs: self.main_programs,
        }
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        SystemBuilder::new()
    }
}
