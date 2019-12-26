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

use crate::native::traits::{NativeProgramEvent, NativeProgramMessageIdWrite, NativeProgramRef};

use alloc::{boxed::Box, vec::Vec};
use core::{mem, pin::Pin, task::Context, task::Poll};
use futures::prelude::*;
use hashbrown::{hash_map::Entry, HashMap, HashSet};
use redshirt_interface_interface::ffi::InterfaceMessage;
use redshirt_syscalls_interface::{Decode as _, EncodedMessage, MessageId, Pid};
use spin::Mutex;

/// Collection of objects that implement the [`NativeProgram`] trait.
pub struct NativeProgramsCollection {
    // TODO: add ` + 'a` in the `Box`, to allow non-'static programs
    processes: HashMap<Pid, Box<dyn AdapterAbstract + Send>>,
}

/// Event generated by a [`NativeProgram`].
pub enum NativeProgramsCollectionEvent<'a> {
    /// Request to emit a message.
    Emit {
        interface: [u8; 32],
        pid: Pid,
        message: EncodedMessage,
        message_id_write: Option<NativeProgramsCollectionMessageIdWrite<'a>>,
    },
    /// Request to cancel a previously-emitted message.
    CancelMessage { message_id: MessageId },
    Answer {
        message_id: MessageId,
        answer: Result<EncodedMessage, ()>,
    },
}

pub struct NativeProgramsCollectionMessageIdWrite<'collec> {
    write: Box<dyn AbstractMessageIdWrite + 'collec>,
}

/// Wraps around a [`NativeProgram`].
struct Adapter<T> {
    inner: T,
    registered_interfaces: Mutex<HashSet<[u8; 32]>>,
    expected_responses: Mutex<HashSet<MessageId>>,
}

/// Abstracts over [`Adapter`] so that we can box it.
trait AdapterAbstract {
    fn poll_next_event<'a>(
        &'a self,
        cx: &mut Context,
    ) -> Poll<NativeProgramEvent<Box<dyn AbstractMessageIdWrite + 'a>>>;
    fn deliver_interface_message(
        &self,
        interface: [u8; 32],
        message_id: Option<MessageId>,
        emitter_pid: Pid,
        message: EncodedMessage,
    ) -> Result<(), EncodedMessage>;
    fn deliver_response(
        &self,
        message_id: MessageId,
        response: Result<EncodedMessage, ()>,
    ) -> Result<(), Result<EncodedMessage, ()>>;
    fn process_destroyed(&self, pid: Pid);
}

trait AbstractMessageIdWrite {
    fn acknowledge(&mut self, id: MessageId);
}

struct MessageIdWriteAdapter<'a, T> {
    inner: Option<T>,
    expected_responses: &'a Mutex<HashSet<MessageId>>,
}

impl NativeProgramsCollection {
    pub fn new() -> Self {
        NativeProgramsCollection {
            processes: HashMap::new(),
        }
    }

    /// Adds a program to the collection.
    ///
    /// # Panic
    ///
    /// Panics if the `pid` already exists in this collection.
    ///
    pub fn push<T>(&mut self, pid: Pid, program: T)
    where
        T: Send + 'static,
        for<'r> &'r T: NativeProgramRef<'r>,
    {
        let adapter = Box::new(Adapter {
            inner: program,
            registered_interfaces: Mutex::new(HashSet::new()),
            expected_responses: Mutex::new(HashSet::new()),
        });

        match self.processes.entry(pid) {
            Entry::Occupied(_) => panic!(),
            Entry::Vacant(e) => e.insert(adapter),
        };

        // We assume that `push` is only ever called at initialization.
        self.processes.shrink_to_fit();
    }

    pub fn next_event<'collec>(
        &'collec self,
    ) -> impl Future<Output = NativeProgramsCollectionEvent<'collec>> + 'collec {
        future::poll_fn(move |cx| {
            for (pid, process) in self.processes.iter() {
                match process.poll_next_event(cx) {
                    Poll::Pending => {}
                    Poll::Ready(NativeProgramEvent::Emit {
                        interface,
                        message_id_write,
                        message,
                    }) => {
                        return Poll::Ready(NativeProgramsCollectionEvent::Emit {
                            pid: *pid,
                            interface,
                            message,
                            message_id_write: message_id_write
                                .map(|w| NativeProgramsCollectionMessageIdWrite { write: w }),
                        })
                    }
                    Poll::Ready(NativeProgramEvent::CancelMessage { message_id }) => {
                        return Poll::Ready(NativeProgramsCollectionEvent::CancelMessage {
                            message_id,
                        })
                    }
                    Poll::Ready(NativeProgramEvent::Answer { message_id, answer }) => {
                        return Poll::Ready(NativeProgramsCollectionEvent::Answer {
                            message_id,
                            answer,
                        })
                    }
                }
            }

            Poll::Pending
        })
    }

    /// Notify the [`NativeProgram`] that a message has arrived on one of the interface that it
    /// has registered.
    pub fn interface_message(
        &self,
        interface: [u8; 32],
        message_id: Option<MessageId>,
        emitter_pid: Pid,
        mut message: EncodedMessage,
    ) {
        for process in self.processes.values() {
            let mut msg = mem::replace(&mut message, EncodedMessage(Vec::new()));
            match process.deliver_interface_message(interface, message_id, emitter_pid, msg) {
                Ok(_) => return,
                Err(msg) => message = msg,
            }
        }

        panic!() // TODO: what to do here?
    }

    /// Notify the [`NativeProgram`]s that the program with the given [`Pid`] has terminated.
    pub fn process_destroyed(&mut self, pid: Pid) {
        for process in self.processes.values() {
            process.process_destroyed(pid);
        }
    }

    /// Notify the appropriate [`NativeProgram`] of a response to a message that it has previously
    /// emitted.
    pub fn message_response(
        &self,
        message_id: MessageId,
        mut response: Result<EncodedMessage, ()>,
    ) {
        for process in self.processes.values() {
            let mut msg = mem::replace(&mut response, Ok(EncodedMessage(Vec::new())));
            match process.deliver_response(message_id, msg) {
                Ok(_) => return,
                Err(msg) => response = msg,
            }
        }

        panic!() // TODO: what to do here?
    }
}

impl<T> AdapterAbstract for Adapter<T>
where
    for<'r> &'r T: NativeProgramRef<'r>,
{
    fn poll_next_event<'a>(
        &'a self,
        cx: &mut Context,
    ) -> Poll<NativeProgramEvent<Box<dyn AbstractMessageIdWrite + 'a>>> {
        let future = (&self.inner).next_event();
        futures::pin_mut!(future);
        match future.poll(cx) {
            Poll::Ready(NativeProgramEvent::Emit {
                interface,
                message_id_write,
                message,
            }) => {
                if interface == redshirt_interface_interface::ffi::INTERFACE {
                    // TODO: check whether registration succeeds, but hard if `message_id_write` is `None
                    if let Ok(msg) = InterfaceMessage::decode(message.clone()) {
                        let InterfaceMessage::Register(to_reg) = msg;
                        let mut registered_interfaces = self.registered_interfaces.lock();
                        registered_interfaces.insert(to_reg);
                    }
                }

                let message_id_write = message_id_write.map(|inner| {
                    Box::new(MessageIdWriteAdapter {
                        inner: Some(inner),
                        expected_responses: &self.expected_responses,
                    }) as Box<_>
                });

                Poll::Ready(NativeProgramEvent::Emit {
                    interface,
                    message,
                    message_id_write,
                })
            }
            Poll::Ready(NativeProgramEvent::CancelMessage { message_id }) => {
                Poll::Ready(NativeProgramEvent::CancelMessage { message_id })
            }
            Poll::Ready(NativeProgramEvent::Answer { message_id, answer }) => {
                Poll::Ready(NativeProgramEvent::Answer { message_id, answer })
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn deliver_interface_message(
        &self,
        interface: [u8; 32],
        message_id: Option<MessageId>,
        emitter_pid: Pid,
        message: EncodedMessage,
    ) -> Result<(), EncodedMessage> {
        let registered_interfaces = self.registered_interfaces.lock();
        if registered_interfaces.contains(&interface) {
            self.inner
                .interface_message(interface, message_id, emitter_pid, message);
            Ok(())
        } else {
            Err(message)
        }
    }

    fn deliver_response(
        &self,
        message_id: MessageId,
        response: Result<EncodedMessage, ()>,
    ) -> Result<(), Result<EncodedMessage, ()>> {
        let mut expected_responses = self.expected_responses.lock();
        if expected_responses.remove(&message_id) {
            self.inner.message_response(message_id, response);
            Ok(())
        } else {
            Err(response)
        }
    }

    fn process_destroyed(&self, pid: Pid) {
        self.inner.process_destroyed(pid);
    }
}

impl<'a, T> AbstractMessageIdWrite for MessageIdWriteAdapter<'a, T>
where
    T: NativeProgramMessageIdWrite,
{
    fn acknowledge(&mut self, id: MessageId) {
        self.inner.take().unwrap().acknowledge(id);
        let _was_inserted = self.expected_responses.lock().insert(id);
        debug_assert!(_was_inserted);
    }
}

impl<'a> NativeProgramMessageIdWrite for NativeProgramsCollectionMessageIdWrite<'a> {
    fn acknowledge(mut self, message_id: MessageId) {
        self.write.acknowledge(message_id);
    }
}

// TODO: impl<'a> NativeProgram<'a> for NativeProgramsCollection<'a>
