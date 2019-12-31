// Copyright (C) 2019  Pierre Krieger
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

use crate::id_pool::IdPool;
use crate::module::Module;
use crate::scheduler::{processes, vm};
use crate::sig;
use crate::signature::Signature;
use crate::InterfaceHash;

use alloc::{borrow::Cow, collections::VecDeque, vec, vec::Vec};
use byteorder::{ByteOrder as _, LittleEndian};
use core::{convert::TryFrom, iter, mem};
use crossbeam_queue::SegQueue;
use hashbrown::{hash_map::Entry, HashMap, HashSet};
use redshirt_syscalls_interface::{Encode, EncodedMessage, MessageId, Pid, ThreadId};
use smallvec::SmallVec;

/// Handles scheduling processes and inter-process communications.
pub struct Core {
    /// Queue of events to return in priority when `run` is called.
    pending_events: SegQueue<CoreRunOutcomeInner>,

    /// List of running processes.
    processes: processes::ProcessesCollection<Extrinsic, Process, Thread>,

    /// List of `Pid`s that have been reserved during the construction.
    ///
    /// Never modified after initialization.
    reserved_pids: HashSet<Pid>,

    /// For each interface, which program is fulfilling it.
    interfaces: HashMap<InterfaceHash, InterfaceState>,

    /// Pool of identifiers for messages.
    message_id_pool: IdPool,

    /// List of messages that have been emitted by a process and that are waiting for a response.
    // TODO: doc about hash safety
    // TODO: call shrink_to from time to time
    messages_to_answer: HashMap<MessageId, Pid>,
}

/// Which way an interface is handled.
#[derive(Debug, Clone, PartialEq, Eq)]
enum InterfaceState {
    /// Interface has been registered using [`Core::set_interface_handler`].
    Process(Pid),
    /// Interface hasn't been registered yet, but has been requested.
    Requested {
        /// List of threads waiting for this interface. All the threads in this list must be in
        /// the [`Thread::InterfaceNotAvailableWait`] state.
        threads: SmallVec<[ThreadId; 4]>,
        /// Other messages waiting to be delivered to this interface.
        other: Vec<(Pid, Option<MessageId>, EncodedMessage)>,
    },
}

/// Possible function available to processes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Extrinsic {
    NextMessage,
    EmitMessage,
    EmitMessageError,
    EmitAnswer,
    CancelMessage,
}

/// Prototype for a `Core` under construction.
pub struct CoreBuilder {
    /// See the corresponding field in `Core`.
    reserved_pids: HashSet<Pid>,
    /// Builder for the [`processes`][Core::processes] field in `Core`.
    inner_builder: processes::ProcessesCollectionBuilder<Extrinsic>,
}

/// Outcome of calling [`run`](Core::run).
// TODO: #[derive(Debug)]
pub enum CoreRunOutcome<'a> {
    /// A program has stopped, either because the main function has stopped or a problem has
    /// occurred.
    ProgramFinished {
        /// Id of the program that has stopped.
        pid: Pid,

        /// List of messages emitted using [`Core::emit_interface_message_answer`] that were
        /// supposed to be handled by the process that has just terminated.
        unhandled_messages: Vec<MessageId>,

        /// List of messages for which a [`CoreRunOutcome::InterfaceMessage`] has been emitted
        /// but that no loner need answering.
        cancelled_messages: Vec<MessageId>,

        /// List of interfaces that were registered by th process and no longer are.
        unregistered_interfaces: Vec<InterfaceHash>,

        /// How the program ended. If `Ok`, it has gracefully terminated. If `Err`, something
        /// bad happened.
        // TODO: force Ok to i32?
        outcome: Result<Option<wasmi::RuntimeValue>, wasmi::Trap>,
    },

    /// Thread has tried to emit a message on an interface that isn't registered. The thread is
    /// now in sleep mode. You can either wake it up by calling [`set_interface_handler`], or
    /// resume the thread with an "interface not available error" by calling . // TODO
    ThreadWaitUnavailableInterface {
        /// Thread that emitted the message.
        thread: CoreThread<'a>,

        /// Interface that the thread is trying to access.
        interface: InterfaceHash,
    },

    /// A process has emitted a message on an interface registered with a reserved PID.
    ReservedPidInterfaceMessage {
        pid: Pid,
        message_id: Option<MessageId>,
        interface: InterfaceHash,
        message: EncodedMessage,
    },

    /// Response to a message emitted using [`Core::emit_interface_message_answer`].
    MessageResponse {
        message_id: MessageId,
        response: Result<EncodedMessage, ()>,
    },

    /// Nothing to do. No thread is ready to run.
    Idle,
}

/// Because of lifetime issues, this is the same as `CoreRunOutcome` but that holds `Pid`s instead
/// of `CoreProcess`es.
// TODO: remove this enum and solve borrowing issues
enum CoreRunOutcomeInner {
    ProgramFinished {
        pid: Pid,
        unhandled_messages: Vec<MessageId>,
        cancelled_messages: Vec<MessageId>,
        unregistered_interfaces: Vec<InterfaceHash>,
        outcome: Result<Option<wasmi::RuntimeValue>, wasmi::Trap>,
    },
    ThreadWaitUnavailableInterface {
        thread: ThreadId,
        interface: InterfaceHash,
    },
    ReservedPidInterfaceMessage {
        // TODO: `pid` is redundant with `message_id`; should just be a better API with an `Event` handle struct
        pid: Pid,
        message_id: Option<MessageId>,
        interface: InterfaceHash,
        message: EncodedMessage,
    },
    MessageResponse {
        message_id: MessageId,
        response: Result<EncodedMessage, ()>,
    },
    LoopAgain,
    Idle,
}

/// Additional information about a process.
#[derive(Debug)]
struct Process {
    /// Messages available for retrieval by the process by calling `next_message`.
    ///
    /// Note that the [`ResponseMessage::index_in_list`](redshirt_syscalls_interface::ffi::ResponseMessage::index_in_list)
    /// and [`InterfaceMessage::index_in_list`](redshirt_syscalls_interface::ffi::InterfaceMessage::index_in_list) fields are
    /// set to a dummy value, and must be filled before actually delivering the message.
    // TODO: call shrink_to_fit from time to time
    messages_queue: VecDeque<redshirt_syscalls_interface::ffi::Message>,

    /// Interfaces that the process has registered.
    registered_interfaces: SmallVec<[InterfaceHash; 1]>,

    /// List of interfaces that this process has used. When the process dies, we notify all the
    /// handlers about it.
    used_interfaces: HashSet<InterfaceHash>,

    /// List of messages that the process has emitted and that are waiting for an answer.
    emitted_messages: SmallVec<[MessageId; 8]>,

    /// List of messages that the process is expected to answer.
    messages_to_answer: SmallVec<[MessageId; 8]>,
}

/// Additional information about a thread. Must be consistent with the actual state of the thread.
#[derive(Debug, PartialEq, Eq)]
enum Thread {
    /// Thread is ready to run.
    ReadyToRun,

    /// The thread is sleeping and waiting for a message to come.
    ///
    /// Note that this can be set even if the `messages_queue` is not empty, in the case where
    /// the thread is waiting only on messages that aren't in the queue.
    MessageWait(MessageWait),

    /// The thread called `emit_message` and wants to emit a message on an interface for which no
    /// handler was available at the time.
    InterfaceNotAvailableWait {
        /// Interface we want to emit the message on.
        interface: InterfaceHash,
        /// Identifier of the message if it expects an answer.
        message_id: Option<MessageId>,
        /// Message itself. Needs to be delivered to the handler once it is registered.
        message: EncodedMessage,
    },

    /// The thread is sleeping and waiting for an external extrinsic.
    ExtrinsicWait,

    /// Thread has been interrupted, and the call is being processed right now.
    InProcess,
}

/// How a process is waiting for messages.
#[derive(Debug, Clone, PartialEq, Eq)] // TODO: remove Clone
struct MessageWait {
    /// Identifiers of the messages we are waiting upon. Duplicate of what is in the process's
    /// memory.
    msg_ids: Vec<MessageId>,
    /// Offset within the memory of the process where the list of messages to wait upon is
    /// located. This is necessary as we have to zero.
    msg_ids_ptr: u32,
    /// Offset within the memory of the process where to write the received message.
    out_pointer: u32,
    /// Size of the memory of the process dedicated to receiving the message.
    out_size: u32,
}

/// Access to a process within the core.
pub struct CoreProcess<'a> {
    /// Access to the process within the inner collection.
    process: processes::ProcessesCollectionProc<'a, Process, Thread>,
}

/// Access to a thread within the core.
pub struct CoreThread<'a> {
    /// Access to the thread within the inner collection.
    thread: processes::ProcessesCollectionThread<'a, Process, Thread>,
}

impl Core {
    // TODO: figure out borrowing issues and remove that Clone
    /// Initialies a new `Core`.
    pub fn new() -> CoreBuilder {
        CoreBuilder {
            reserved_pids: HashSet::new(),
            inner_builder: processes::ProcessesCollectionBuilder::default()
                .with_extrinsic(
                    "redshirt",
                    "next_message",
                    sig!((I32, I32, I32, I32, I32) -> I32),
                    Extrinsic::NextMessage,
                )
                .with_extrinsic(
                    "redshirt",
                    "emit_message",
                    sig!((I32, I32, I32, I32, I32, I32) -> I32),
                    Extrinsic::EmitMessage,
                )
                .with_extrinsic(
                    "redshirt",
                    "emit_message_error",
                    sig!((I32)),
                    Extrinsic::EmitMessageError,
                )
                .with_extrinsic(
                    "redshirt",
                    "emit_answer",
                    sig!((I32, I32, I32)),
                    Extrinsic::EmitAnswer,
                )
                .with_extrinsic(
                    "redshirt",
                    "cancel_message",
                    sig!((I32)),
                    Extrinsic::CancelMessage,
                ),
        }
    }

    /// Run the core once.
    // TODO: make multithreaded
    pub fn run(&mut self) -> CoreRunOutcome {
        loop {
            break match self.run_inner() {
                CoreRunOutcomeInner::Idle => CoreRunOutcome::Idle,
                CoreRunOutcomeInner::LoopAgain => continue,
                CoreRunOutcomeInner::ProgramFinished {
                    pid,
                    unhandled_messages,
                    cancelled_messages,
                    unregistered_interfaces,
                    outcome,
                } => CoreRunOutcome::ProgramFinished {
                    pid,
                    unhandled_messages,
                    cancelled_messages,
                    unregistered_interfaces,
                    outcome,
                },
                CoreRunOutcomeInner::ThreadWaitUnavailableInterface { thread, interface } => {
                    CoreRunOutcome::ThreadWaitUnavailableInterface {
                        thread: CoreThread {
                            thread: self.processes.thread_by_id(thread).unwrap(),
                        },
                        interface,
                    }
                }
                CoreRunOutcomeInner::ReservedPidInterfaceMessage {
                    pid,
                    message_id,
                    interface,
                    message,
                } => CoreRunOutcome::ReservedPidInterfaceMessage {
                    pid,
                    message_id,
                    interface,
                    message,
                },
                CoreRunOutcomeInner::MessageResponse {
                    message_id,
                    response,
                } => CoreRunOutcome::MessageResponse {
                    message_id,
                    response,
                },
            };
        }
    }

    /// Because of lifetime issues, we return an enum that holds `Pid`s instead of `CoreProcess`es.
    /// Then `run` does the conversion in order to have a good API.
    // TODO: make multithreaded
    fn run_inner(&mut self) -> CoreRunOutcomeInner {
        if let Ok(ev) = self.pending_events.pop() {
            return ev;
        }

        match self.processes.run() {
            processes::RunOneOutcome::ProcessFinished {
                pid,
                outcome,
                dead_threads,
                user_data,
            } => {
                debug_assert_eq!(dead_threads[0].1, Thread::ReadyToRun);
                for (dead_thread_id, dead_thread_state) in dead_threads {
                    match dead_thread_state {
                        _ => {} // TODO:
                    }
                }

                // Unregister the interfaces this program had registered.
                let mut unregistered_interfaces = Vec::new();
                for interface in user_data.registered_interfaces {
                    let _interface = self.interfaces.remove(&interface);
                    debug_assert_eq!(_interface, Some(InterfaceState::Process(pid)));
                    unregistered_interfaces.push(interface);
                }

                // Cancelling messages that the process had emitted.
                // TODO: this only handles messages emitted through the external API
                let mut cancelled_messages = Vec::new();
                for emitted_message in user_data.emitted_messages {
                    let _emitter = self.messages_to_answer.remove(&emitted_message);
                    debug_assert_eq!(_emitter, Some(pid));
                    cancelled_messages.push(emitted_message);
                }

                // Notify interface handlers about the process stopping.
                for interface in user_data.used_interfaces {
                    match self.interfaces.get(&interface) {
                        Some(InterfaceState::Process(p)) => {
                            if let Some(mut process) = self.processes.process_by_id(*p) {
                                let message =
                                    redshirt_syscalls_interface::ffi::Message::ProcessDestroyed(
                                        redshirt_syscalls_interface::ffi::ProcessDestroyedMessage {
                                            index_in_list: 0,
                                            pid: pid.into(),
                                        },
                                    );

                                process.user_data().messages_queue.push_back(message);
                                try_resume_message_wait(process);
                            } // TODO: notify externals as well?
                        }
                        None => unreachable!(),
                        _ => {}
                    }
                }

                // TODO: also, what do we do with the pending messages and all?

                CoreRunOutcomeInner::ProgramFinished {
                    pid,
                    unregistered_interfaces,
                    // TODO: this only handles messages emitted through the external API
                    unhandled_messages: user_data.messages_to_answer.to_vec(), // TODO: to_vec overhead
                    cancelled_messages,
                    outcome,
                }
            }

            processes::RunOneOutcome::ThreadFinished { user_data, .. } => {
                debug_assert_eq!(user_data, Thread::ReadyToRun);
                // TODO: report?
                CoreRunOutcomeInner::LoopAgain
            }

            processes::RunOneOutcome::Interrupted {
                mut thread,
                id: Extrinsic::NextMessage,
                params,
            } => {
                debug_assert_eq!(*thread.user_data(), Thread::ReadyToRun);
                *thread.user_data() = Thread::InProcess;
                // TODO: refactor a bit to first parse the parameters and then update `self`
                extrinsic_next_message(&mut thread, params);
                CoreRunOutcomeInner::LoopAgain
            }

            processes::RunOneOutcome::Interrupted {
                mut thread,
                id: Extrinsic::EmitMessage,
                params,
            } => {
                debug_assert_eq!(*thread.user_data(), Thread::ReadyToRun);
                *thread.user_data() = Thread::InProcess;

                // TODO: lots of unwraps here
                assert_eq!(params.len(), 6);
                let interface: InterfaceHash = {
                    let addr = params[0].try_into::<i32>().unwrap() as u32;
                    InterfaceHash::from(
                        <[u8; 32]>::try_from(&thread.read_memory(addr, 32).unwrap()[..]).unwrap(),
                    )
                };
                let message = {
                    let addr = params[1].try_into::<i32>().unwrap() as u32;
                    let num_bufs = params[2].try_into::<i32>().unwrap() as u32;
                    let mut out_msg = Vec::new();
                    for buf_n in 0..num_bufs {
                        let sub_buf_ptr = thread.read_memory(addr + 8 * buf_n, 4).unwrap();
                        let sub_buf_ptr = LittleEndian::read_u32(&sub_buf_ptr);
                        let sub_buf_sz = thread.read_memory(addr + 8 * buf_n + 4, 4).unwrap();
                        let sub_buf_sz = LittleEndian::read_u32(&sub_buf_sz);
                        out_msg.extend_from_slice(
                            &thread.read_memory(sub_buf_ptr, sub_buf_sz).unwrap(),
                        );
                    }
                    EncodedMessage(out_msg)
                };
                let needs_answer = params[3].try_into::<i32>().unwrap() != 0;
                let allow_delay = params[4].try_into::<i32>().unwrap() != 0;
                let emitter_pid = thread.pid();
                let message_id = if needs_answer {
                    let message_id_write = params[5].try_into::<i32>().unwrap() as u32;
                    let new_message_id = loop {
                        let id: MessageId = self.message_id_pool.assign();
                        if u64::from(id) == 0 || u64::from(id) == 1 {
                            continue;
                        }
                        match self.messages_to_answer.entry(id) {
                            Entry::Occupied(_) => continue,
                            Entry::Vacant(e) => e.insert(emitter_pid),
                        };
                        break id;
                    };
                    let mut buf = [0; 8];
                    LittleEndian::write_u64(&mut buf, From::from(new_message_id));
                    thread.write_memory(message_id_write, &buf).unwrap();
                    // TODO: thread.user_data().;
                    // TODO: thread.process().user_data().emitted_messages.push();
                    Some(new_message_id)
                } else {
                    None
                };

                thread
                    .process_user_data()
                    .used_interfaces
                    .insert(interface.clone());

                match (self.interfaces.get_mut(&interface), allow_delay) {
                    (Some(InterfaceState::Process(pid)), _) => {
                        *thread.user_data() = Thread::ReadyToRun;
                        thread.resume(Some(wasmi::RuntimeValue::I32(0)));

                        if let Some(mut process) = self.processes.process_by_id(*pid) {
                            let message = redshirt_syscalls_interface::ffi::Message::Interface(
                                redshirt_syscalls_interface::ffi::InterfaceMessage {
                                    interface: interface.into(),
                                    index_in_list: 0,
                                    message_id,
                                    emitter_pid: emitter_pid.into(),
                                    actual_data: message.0,
                                },
                            );

                            let mut process = self.processes.process_by_id(*pid).unwrap();
                            process.user_data().messages_queue.push_back(message);
                            try_resume_message_wait(process);
                            CoreRunOutcomeInner::LoopAgain
                        } else {
                            CoreRunOutcomeInner::ReservedPidInterfaceMessage {
                                pid: emitter_pid,
                                message_id,
                                interface,
                                message,
                            }
                        }
                    }
                    (None, false) | (Some(InterfaceState::Requested { .. }), false) => {
                        *thread.user_data() = Thread::ReadyToRun;
                        thread.resume(Some(wasmi::RuntimeValue::I32(1)));
                        CoreRunOutcomeInner::LoopAgain
                    }
                    (Some(InterfaceState::Requested { threads, .. }), true) => {
                        *thread.user_data() = Thread::InterfaceNotAvailableWait {
                            interface: interface.clone(),
                            message_id,
                            message,
                        };
                        threads.push(thread.tid());
                        CoreRunOutcomeInner::ThreadWaitUnavailableInterface {
                            thread: thread.tid(),
                            interface,
                        }
                    }
                    (None, true) => {
                        *thread.user_data() = Thread::InterfaceNotAvailableWait {
                            interface: interface.clone(),
                            message_id,
                            message,
                        };
                        self.interfaces.insert(
                            interface.clone(),
                            InterfaceState::Requested {
                                threads: iter::once(thread.tid()).collect(),
                                other: Vec::new(),
                            },
                        );
                        CoreRunOutcomeInner::ThreadWaitUnavailableInterface {
                            thread: thread.tid(),
                            interface,
                        }
                    }
                }
            }

            processes::RunOneOutcome::Interrupted {
                mut thread,
                id: Extrinsic::EmitAnswer,
                params,
            } => {
                debug_assert_eq!(*thread.user_data(), Thread::ReadyToRun);
                *thread.user_data() = Thread::InProcess;

                // TODO: lots of unwraps here
                assert_eq!(params.len(), 3);
                let msg_id = {
                    let addr = params[0].try_into::<i32>().unwrap() as u32;
                    let buf = thread.read_memory(addr, 8).unwrap();
                    MessageId::from(byteorder::LittleEndian::read_u64(&buf))
                };
                let message = {
                    let addr = params[1].try_into::<i32>().unwrap() as u32;
                    let sz = params[2].try_into::<i32>().unwrap() as u32;
                    EncodedMessage(thread.read_memory(addr, sz).unwrap())
                };
                let pid = thread.pid();
                thread.resume(None);
                self.answer_message_inner(msg_id, Ok(message))
                    .unwrap_or(CoreRunOutcomeInner::LoopAgain)
            }

            processes::RunOneOutcome::Interrupted {
                mut thread,
                id: Extrinsic::EmitMessageError,
                params,
            } => {
                debug_assert_eq!(*thread.user_data(), Thread::ReadyToRun);
                *thread.user_data() = Thread::InProcess;

                // TODO: lots of unwraps here
                assert_eq!(params.len(), 1);
                let msg_id = {
                    let addr = params[0].try_into::<i32>().unwrap() as u32;
                    let buf = thread.read_memory(addr, 8).unwrap();
                    MessageId::from(byteorder::LittleEndian::read_u64(&buf))
                };

                self.messages_to_answer.remove(&msg_id);
                thread.resume(None);

                let pid = thread.pid();
                self.answer_message_inner(msg_id, Err(()))
                    .unwrap_or(CoreRunOutcomeInner::LoopAgain)
            }

            processes::RunOneOutcome::Interrupted {
                mut thread,
                id: Extrinsic::CancelMessage,
                params,
            } => unimplemented!(),

            processes::RunOneOutcome::Idle => CoreRunOutcomeInner::Idle,
        }
    }

    /// Returns an object granting access to a process, if it exists.
    pub fn process_by_id(&mut self, pid: Pid) -> Option<CoreProcess> {
        let p = self.processes.process_by_id(pid)?;
        Some(CoreProcess { process: p })
    }

    /// Returns an object granting access to a thread, if it exists.
    pub fn thread_by_id(&mut self, thread: ThreadId) -> Option<CoreThread> {
        let thread = self.processes.thread_by_id(thread)?;
        Some(CoreThread { thread })
    }

    // TODO: better API
    pub fn set_interface_handler(
        &mut self,
        interface: InterfaceHash,
        process: Pid,
    ) -> Result<(), ()> {
        if self.processes.process_by_id(process).is_none() {
            if !self.reserved_pids.contains(&process) {
                return Err(());
            }
        } else {
            debug_assert!(!self.reserved_pids.contains(&process));
        }

        let (thread_ids, other_messages) = match self.interfaces.entry(interface.clone()) {
            Entry::Vacant(e) => {
                e.insert(InterfaceState::Process(process));
                return Ok(());
            }
            Entry::Occupied(mut e) => {
                // Check whether interface was already registered.
                if let InterfaceState::Requested { .. } = *e.get_mut() {
                } else {
                    return Err(());
                };
                match mem::replace(e.get_mut(), InterfaceState::Process(process)) {
                    InterfaceState::Requested { threads, other } => (threads, other),
                    _ => unreachable!(),
                }
            }
        };

        // Send the `other_messages`.
        // TODO: should we preserve the order w.r.t. `threads`?
        for (emitter_pid, message_id, message_data) in other_messages {
            let message = redshirt_syscalls_interface::ffi::Message::Interface(
                redshirt_syscalls_interface::ffi::InterfaceMessage {
                    interface: interface.clone().into(),
                    index_in_list: 0,
                    message_id,
                    emitter_pid,
                    actual_data: message_data.0,
                },
            );

            self.processes
                .process_by_id(process)
                .unwrap()
                .user_data()
                .messages_queue
                .push_back(message);
        }

        // Now process the threads that were waiting for this interface to be registered.
        for thread_id in thread_ids {
            let mut thread = self.processes.thread_by_id(thread_id).unwrap();
            let thread_user_data = mem::replace(thread.user_data(), Thread::ReadyToRun);
            if let Thread::InterfaceNotAvailableWait {
                interface: int,
                message_id,
                message,
            } = thread_user_data
            {
                assert_eq!(interface, int);

                thread.resume(Some(wasmi::RuntimeValue::I32(0)));
                let emitter_pid = thread.pid().into();

                if let Some(mut interface_handler_proc) = self.processes.process_by_id(process) {
                    let message = redshirt_syscalls_interface::ffi::Message::Interface(
                        redshirt_syscalls_interface::ffi::InterfaceMessage {
                            interface: interface.clone().into(),
                            index_in_list: 0,
                            message_id,
                            emitter_pid,
                            actual_data: message.0,
                        },
                    );

                    interface_handler_proc
                        .user_data()
                        .messages_queue
                        .push_back(message);
                // TODO: try_resume_message_wait(interface_handler_proc);
                } else {
                    self.pending_events
                        .push(CoreRunOutcomeInner::ReservedPidInterfaceMessage {
                            pid: emitter_pid,
                            message_id,
                            interface: interface.clone(),
                            message,
                        });
                }
            } else {
                // State inconsistency in the core.
                unreachable!()
            }
        }

        // TODO: do we have to resume `process`?

        Ok(())
    }

    /// Emits a message for the handler of the given interface.
    ///
    /// The message doesn't expect any answer.
    // TODO: better API
    pub fn emit_interface_message_no_answer<'a>(
        &mut self,
        emitter_pid: Pid,
        interface: InterfaceHash,
        message: impl Encode,
    ) {
        assert!(self.reserved_pids.contains(&emitter_pid));
        let _out = self.emit_interface_message_inner(emitter_pid, interface, message, false);
        debug_assert!(_out.is_none());
    }

    /// Emits a message for the handler of the given interface.
    ///
    /// The message does expect an answer. The answer will be sent back as
    /// [`MessageResponse`](CoreRunOutcome::MessageResponse) event.
    // TODO: better API
    pub fn emit_interface_message_answer<'a>(
        &mut self,
        emitter_pid: Pid,
        interface: InterfaceHash,
        message: impl Encode,
    ) -> MessageId {
        assert!(self.reserved_pids.contains(&emitter_pid));
        self.emit_interface_message_inner(emitter_pid, interface, message, true)
            .unwrap()
    }

    fn emit_interface_message_inner<'a>(
        &mut self,
        emitter_pid: Pid,
        interface: InterfaceHash,
        message: impl Encode,
        needs_answer: bool,
    ) -> Option<MessageId> {
        let (message_id, messages_to_answer_entry) = if needs_answer {
            loop {
                let id: MessageId = self.message_id_pool.assign();
                if u64::from(id) == 0 || u64::from(id) == 1 {
                    continue;
                }
                match self.messages_to_answer.entry(id) {
                    Entry::Vacant(e) => break (Some(id), Some(e)),
                    Entry::Occupied(_) => continue,
                };
            }
        } else {
            (None, None)
        };

        let pid = match self.interfaces.entry(interface.clone()).or_insert_with(|| {
            InterfaceState::Requested {
                threads: SmallVec::new(),
                other: Vec::new(),
            }
        }) {
            InterfaceState::Process(pid) => *pid,
            InterfaceState::Requested { other, .. } => {
                other.push((emitter_pid, message_id, message.encode()));
                return message_id;
            }
        };

        if let Some(mut process) = self.processes.process_by_id(pid) {
            let message = redshirt_syscalls_interface::ffi::Message::Interface(
                redshirt_syscalls_interface::ffi::InterfaceMessage {
                    interface: interface.into(),
                    message_id,
                    emitter_pid,
                    index_in_list: 0,
                    actual_data: message.encode().0.to_vec(),
                },
            );

            process.user_data().messages_queue.push_back(message);
            try_resume_message_wait(process);
        } else {
            assert!(self.reserved_pids.contains(&emitter_pid));
            self.pending_events
                .push(CoreRunOutcomeInner::ReservedPidInterfaceMessage {
                    pid: emitter_pid,
                    message_id: None,
                    interface,
                    message: message.encode(),
                });
        };

        if let Some(messages_to_answer_entry) = messages_to_answer_entry {
            messages_to_answer_entry.insert(emitter_pid);
        }
        message_id
    }

    ///
    ///
    /// It is forbidden to answer messages created using [`emit_interface_message_answer`] or
    /// [`emit_interface_message_no_answer`]. Only messages generated by processes can be answered
    /// through this method.
    // TODO: better API
    pub fn answer_message(&mut self, message_id: MessageId, response: Result<EncodedMessage, ()>) {
        let ret = self.answer_message_inner(message_id, response);
        assert!(ret.is_none());
    }

    // TODO: better API
    fn answer_message_inner(
        &mut self,
        message_id: MessageId,
        response: Result<EncodedMessage, ()>,
    ) -> Option<CoreRunOutcomeInner> {
        if let Some(emitter_pid) = self.messages_to_answer.remove(&message_id) {
            if let Some(mut process) = self.processes.process_by_id(emitter_pid) {
                let actual_message = redshirt_syscalls_interface::ffi::Message::Response(
                    redshirt_syscalls_interface::ffi::ResponseMessage {
                        message_id,
                        // We a dummy value here and fill it up later when actually delivering the message.
                        index_in_list: 0,
                        actual_data: response.map(|r| r.0.to_vec()),
                    },
                );

                process.user_data().messages_queue.push_back(actual_message);
                process
                    .user_data()
                    .emitted_messages
                    .retain(|m| *m != message_id);
                try_resume_message_wait(process);
                None
            } else {
                Some(CoreRunOutcomeInner::MessageResponse {
                    message_id,
                    response,
                })
            }
        } else {
            // TODO: what to do here?
            panic!("no process found with that event")
        }
    }

    /// Start executing the module passed as parameter.
    ///
    /// Each import of the [`Module`](crate::module::Module) is resolved.
    pub fn execute(&mut self, module: &Module) -> Result<CoreProcess, vm::NewErr> {
        let proc_metadata = Process {
            messages_queue: VecDeque::new(),
            registered_interfaces: SmallVec::new(),
            used_interfaces: HashSet::new(),
            emitted_messages: SmallVec::new(),
            messages_to_answer: SmallVec::new(),
        };

        let process = self
            .processes
            .execute(module, proc_metadata, Thread::ReadyToRun)?;

        Ok(CoreProcess { process })
    }
}

impl<'a> CoreProcess<'a> {
    /// Returns the [`Pid`] of the process.
    pub fn pid(&self) -> Pid {
        self.process.pid()
    }

    /// Adds a new thread to the process, starting the function with the given index and passing
    /// the given parameters.
    // TODO: don't expose wasmi::RuntimeValue
    pub fn start_thread(
        self,
        fn_index: u32,
        params: Vec<wasmi::RuntimeValue>,
    ) -> Result<CoreThread<'a>, vm::StartErr> {
        let thread = self
            .process
            .start_thread(fn_index, params, Thread::ReadyToRun)?;
        Ok(CoreThread { thread })
    }

    /// Kills the process immediately.
    pub fn abort(self) {
        self.process.abort(); // TODO: clean up
    }
}

impl<'a> CoreThread<'a> {
    /// Returns the [`ThreadId`] of the thread.
    pub fn tid(&mut self) -> ThreadId {
        self.thread.tid()
    }

    /// Returns the [`Pid`] of the process associated to this thread.
    pub fn pid(&self) -> Pid {
        self.thread.pid()
    }
}

impl CoreBuilder {
    /// Allocates a `Pid` that will not be used by any process.
    ///
    /// > **Note**: As of the writing of this comment, this feature is only ever used to allocate
    /// >           `Pid`s that last forever. There is therefore no corresponding "unreserve_pid"
    /// >           method that frees such an allocated `Pid`. If there is ever a need to free
    /// >           these `Pid`s, such a method should be added.
    pub fn reserve_pid(&mut self) -> Pid {
        let pid = self.inner_builder.reserve_pid();
        let _was_inserted = self.reserved_pids.insert(pid);
        debug_assert!(_was_inserted);
        pid
    }

    /// Turns the builder into a [`Core`].
    pub fn build(mut self) -> Core {
        self.reserved_pids.shrink_to_fit();

        Core {
            pending_events: SegQueue::new(),
            processes: self.inner_builder.build(),
            interfaces: Default::default(),
            reserved_pids: self.reserved_pids,
            message_id_pool: IdPool::new(),
            messages_to_answer: HashMap::default(),
        }
    }
}

/// Called when a thread calls the `next_message` extrinsic.
///
/// Tries to resume the thread by fetching a message from the queue.
///
/// Returns an error if the extrinsic call was invalid.
fn extrinsic_next_message(
    thread: &mut processes::ProcessesCollectionThread<Process, Thread>,
    params: Vec<wasmi::RuntimeValue>,
) -> Result<(), ()> {
    // TODO: lots of conversions here
    assert_eq!(params.len(), 5);

    let msg_ids_ptr = params[0].try_into::<i32>().ok_or(())? as u32;
    let msg_ids = {
        let addr = msg_ids_ptr;
        let len = params[1].try_into::<i32>().ok_or(())? as u32;
        let mem = thread.read_memory(addr, len * 8)?;
        let mut out = vec![0u64; len as usize];
        byteorder::LittleEndian::read_u64_into(&mem, &mut out);
        out.into_iter().map(MessageId::from).collect::<Vec<_>>() // TODO: meh
    };

    let out_pointer = params[2].try_into::<i32>().ok_or(())? as u32;
    let out_size = params[3].try_into::<i32>().ok_or(())? as u32;
    let block = params[4].try_into::<i32>().ok_or(())? != 0;

    assert!(*thread.user_data() == Thread::InProcess);
    *thread.user_data() = Thread::MessageWait(MessageWait {
        msg_ids,
        msg_ids_ptr,
        out_pointer,
        out_size,
    });

    try_resume_message_wait_thread(thread);

    // If `block` is false, we put the thread to sleep anyway, then wake it up again here.
    if !block && *thread.user_data() != Thread::ReadyToRun {
        debug_assert!(if let Thread::MessageWait(_) = thread.user_data() {
            true
        } else {
            false
        });
        *thread.user_data() = Thread::ReadyToRun;
        thread.resume(Some(wasmi::RuntimeValue::I32(0)));
    }

    Ok(())
}

/// If any of the threads of the given process is waiting for a message to arrive, checks the
/// queue and tries to resume said thread.
fn try_resume_message_wait(process: processes::ProcessesCollectionProc<Process, Thread>) {
    // TODO: is it a good strategy to just go through threads in linear order? what about
    //       round-robin-ness instead?
    let mut thread = process.main_thread();

    loop {
        try_resume_message_wait_thread(&mut thread);
        match thread.next_thread() {
            Some(t) => thread = t,
            None => break,
        };
    }
}

/// If the given thread is waiting for a message to arrive, checks the queue and tries to resume
/// said thread.
// TODO: in order to call this function, we essentially have to put the state machine in a "bad"
// state (message in queue and thread would accept said message); not great
fn try_resume_message_wait_thread(
    thread: &mut processes::ProcessesCollectionThread<Process, Thread>,
) {
    if thread.process_user_data().messages_queue.is_empty() {
        return;
    }

    let msg_wait = match thread.user_data() {
        Thread::MessageWait(ref wait) => wait.clone(), // TODO: don't clone?
        _ => return,
    };

    // Try to find a message in the queue that matches something the user is waiting for.
    let mut index_in_queue = 0;
    let index_in_msg_ids = loop {
        if index_in_queue >= thread.process_user_data().messages_queue.len() {
            // No message found.
            return;
        }

        // For that message in queue, grab the value that must be in `msg_ids` in order to match.
        let msg_id = match &thread.process_user_data().messages_queue[index_in_queue] {
            redshirt_syscalls_interface::ffi::Message::Interface(_) => MessageId::from(1),
            redshirt_syscalls_interface::ffi::Message::ProcessDestroyed(_) => MessageId::from(1),
            redshirt_syscalls_interface::ffi::Message::Response(response) => {
                debug_assert!(u64::from(response.message_id) >= 2);
                response.message_id
            }
        };

        if let Some(p) = msg_wait.msg_ids.iter().position(|id| *id == msg_id.into()) {
            break p as u32;
        }

        index_in_queue += 1;
    };

    // If we reach here, we have found a message that matches what the user wants.

    // Adjust the `index_in_list` field of the message to match what we have.
    match thread.process_user_data().messages_queue[index_in_queue] {
        redshirt_syscalls_interface::ffi::Message::Response(ref mut response) => {
            response.index_in_list = index_in_msg_ids;
        }
        redshirt_syscalls_interface::ffi::Message::Interface(ref mut interface) => {
            interface.index_in_list = index_in_msg_ids;
        }
        redshirt_syscalls_interface::ffi::Message::ProcessDestroyed(ref mut proc_destr) => {
            proc_destr.index_in_list = index_in_msg_ids;
        }
    }

    // Turn said message into bytes.
    // TODO: would be great to not do that every single time
    let msg_bytes = thread.process_user_data().messages_queue[index_in_queue]
        .clone()
        .encode();

    // TODO: don't use as
    if msg_wait.out_size as usize >= msg_bytes.0.len() {
        // Write the message in the process's memory.
        thread
            .write_memory(msg_wait.out_pointer, &msg_bytes.0)
            .unwrap();
        // Zero the corresponding entry in the messages to wait upon.
        thread
            .write_memory(msg_wait.msg_ids_ptr + index_in_msg_ids * 8, &[0; 8])
            .unwrap();
        // Pop the message from the queue, so that we don't deliver it twice.
        thread
            .process_user_data()
            .messages_queue
            .remove(index_in_queue);
    }

    *thread.user_data() = Thread::ReadyToRun;
    thread.resume(Some(wasmi::RuntimeValue::I32(msg_bytes.0.len() as i32))); // TODO: don't use as
}
