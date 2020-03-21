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

//! Implementation of the [`Extrinsics`] trait that supports WASI.

// Reference for function signatures:
// https://github.com/WebAssembly/wasi-libc/blob/e1149ab0677317c6c981bcbb5e4c159e4d2b9669/libc-bottom-half/headers/public/wasi/api.h

use crate::extrinsics::{Extrinsics, ExtrinsicsAction, ExtrinsicsMemoryAccess, SupportedExtrinsic};
use crate::{sig, Encode as _, EncodedMessage, ThreadId};

use alloc::{
    borrow::Cow,
    string::String,
    vec,
    vec::{IntoIter, Vec},
};
use core::convert::TryFrom as _;
use wasmi::RuntimeValue;

/// Dummy implementation of the [`Extrinsics`] trait.
#[derive(Debug, Default)]
pub struct WasiExtrinsics;

/// Identifier of a WASI extrinsic.
#[derive(Debug, Clone)]
pub struct ExtrinsicId(ExtrinsicIdInner);

#[derive(Debug, Clone)]
enum ExtrinsicIdInner {
    ArgsGet,
    ArgsSizesGet,
    ClockTimeGet,
    EnvironGet,
    EnvironSizesGet,
    FdClose,
    FdFdstatGet,
    FdFdstatSetFlags,
    FdFileStatGet,
    FdPrestatDirName,
    FdPrestatGet,
    FdRead,
    FdSeek,
    FdWrite,
    PathCreateDirectory,
    PathFilestatGet,
    PathOpen,
    PollOneOff,
    ProcExit,
    RandomGet,
    SchedYield,
}

/// Context for a call to a WASI external function.
pub struct Context(ContextInner);

enum ContextInner {
    WaitRandom { out_ptr: u32, remaining_len: u32 },
    Resume(Option<RuntimeValue>),
    Finished,
}

impl Extrinsics for WasiExtrinsics {
    type ExtrinsicId = ExtrinsicId;
    type Context = Context;
    type Iterator = IntoIter<SupportedExtrinsic<Self::ExtrinsicId>>;

    fn supported_extrinsics() -> Self::Iterator {
        vec![
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::ArgsGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("args_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::ArgsSizesGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("args_sizes_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::ClockTimeGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("clock_time_get"),
                signature: sig!((I32, I64, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::EnvironGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("environ_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::EnvironSizesGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("environ_sizes_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdClose),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_close"),
                signature: sig!((I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdFdstatGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_fdstat_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdFdstatSetFlags),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_fdstat_set_flags"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdFileStatGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_filestat_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdPrestatDirName),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_prestat_dir_name"),
                signature: sig!((I32, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdPrestatGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_prestat_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdRead),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_read"),
                signature: sig!((I32, I32, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdSeek),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_seek"),
                signature: sig!((I32, I64, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::FdWrite),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("fd_write"),
                signature: sig!((I32, I32, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::PathCreateDirectory),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("path_create_directory"),
                signature: sig!((I32, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::PathFilestatGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("path_filestat_get"),
                signature: sig!((I32, I32, I32, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::PathOpen),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("path_open"),
                signature: sig!((I32, I32, I32, I32, I32, I64, I64, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::PollOneOff),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("poll_oneoff"),
                signature: sig!((I32, I32, I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::ProcExit),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("proc_exit"),
                signature: sig!((I32)),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::RandomGet),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("random_get"),
                signature: sig!((I32, I32) -> I32),
            },
            SupportedExtrinsic {
                id: ExtrinsicId(ExtrinsicIdInner::SchedYield),
                wasm_interface: Cow::Borrowed("wasi_snapshot_preview1"),
                function_name: Cow::Borrowed("sched_yield"),
                signature: sig!(() -> I32),
            },
        ]
        .into_iter()
    }

    fn new_context(
        &self,
        _: ThreadId,
        id: &Self::ExtrinsicId,
        mut params: impl ExactSizeIterator<Item = RuntimeValue>,
        mem_access: &mut impl ExtrinsicsMemoryAccess,
    ) -> (Self::Context, ExtrinsicsAction) {
        match id.0 {
            ExtrinsicIdInner::ArgsGet => unimplemented!(),
            ExtrinsicIdInner::ArgsSizesGet => unimplemented!(),
            ExtrinsicIdInner::ClockTimeGet => unimplemented!(),
            ExtrinsicIdInner::EnvironGet => unimplemented!(),
            ExtrinsicIdInner::EnvironSizesGet => {
                let num_ptr = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let buf_size_ptr = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                mem_access
                    .write_memory(num_ptr, &0u64.to_le_bytes())
                    .unwrap(); // TODO: don't unwrap
                mem_access
                    .write_memory(buf_size_ptr, &0u64.to_le_bytes())
                    .unwrap(); // TODO: don't unwrap

                let context = ContextInner::Finished;
                let action = ExtrinsicsAction::Resume(Some(RuntimeValue::I32(0)));
                (Context(context), action)
            }
            ExtrinsicIdInner::FdClose => unimplemented!(),
            ExtrinsicIdInner::FdFdstatGet => {
                let _fd = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _stat_out_buf = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                // TODO: stdin/stdout/stderr
                if _fd <= 2 {
                    unimplemented!()
                }

                // Returning `__WASI_ERRNO_NOENT` all the time.
                let context = ContextInner::Finished;
                let action = ExtrinsicsAction::Resume(Some(RuntimeValue::I32(44)));
                (Context(context), action)
            }
            ExtrinsicIdInner::FdFdstatSetFlags => unimplemented!(),
            ExtrinsicIdInner::FdFileStatGet => {
                let _fd = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _stat_out_buf = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                // TODO: stdin/stdout/stderr
                if _fd <= 2 {
                    unimplemented!()
                }

                // Returning `__WASI_ERRNO_NOENT` all the time.
                let context = ContextInner::Finished;
                let action = ExtrinsicsAction::Resume(Some(RuntimeValue::I32(44)));
                (Context(context), action)
            }
            ExtrinsicIdInner::FdPrestatDirName => unimplemented!(),
            ExtrinsicIdInner::FdPrestatGet => {
                let _fd = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _out_ptr = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                // TODO: stdin/stdout/stderr
                if _fd <= 2 {
                    unimplemented!()
                }

                // Returning `__WASI_ERRNO_NOENT` all the time.
                let context = ContextInner::Finished;
                let action = ExtrinsicsAction::Resume(Some(RuntimeValue::I32(44)));
                (Context(context), action)
            }
            ExtrinsicIdInner::FdRead => unimplemented!(),
            ExtrinsicIdInner::FdSeek => unimplemented!(),
            ExtrinsicIdInner::FdWrite => {
                let action = fd_write(params, mem_access);
                let context = ContextInner::Resume(Some(RuntimeValue::I32(0)));
                (Context(context), action)
            }
            ExtrinsicIdInner::PathCreateDirectory => unimplemented!(),
            ExtrinsicIdInner::PathFilestatGet => unimplemented!(),
            ExtrinsicIdInner::PathOpen => {
                let _fd = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _dirflags = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _path = {
                    let path_buf = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                    let path_buf_len = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                    // TODO: don't unwrap below
                    let path_utf8 = mem_access
                        .read_memory(path_buf..path_buf + path_buf_len)
                        .unwrap();
                    String::from_utf8(path_utf8).unwrap()
                };
                let _open_flags = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _fs_rights_base = params.next().unwrap().try_into::<i64>().unwrap() as u64;
                let _fs_rights_inherting = params.next().unwrap().try_into::<i64>().unwrap() as u64;
                let _fd_flags = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let opened_fd_ptr = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                // TODO: is it necessary to write the fd out buffer when error happens?
                mem_access
                    .write_memory(opened_fd_ptr, &0u32.to_le_bytes())
                    .unwrap(); // TODO: don't unwrap

                // Returning `__WASI_ERRNO_NOENT` all the time.
                let context = ContextInner::Finished;
                let action = ExtrinsicsAction::Resume(Some(RuntimeValue::I32(44)));
                (Context(context), action)
            }
            ExtrinsicIdInner::PollOneOff => {
                let _subscriptions_buf = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _events_out_buf = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _buf_size = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let _num_events_out = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                unimplemented!()
            }
            ExtrinsicIdInner::ProcExit => {
                let _ret_val = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                // TODO: returning `ProgramCrash` leads to `unimplemented!()`, so we panic
                // beforehand for a more explicit message
                panic!("proc_exit called with {:?}", _ret_val);

                // TODO: implement in a better way?
                let context = ContextInner::Finished;
                (Context(context), ExtrinsicsAction::ProgramCrash)
            }
            ExtrinsicIdInner::RandomGet => {
                let buf = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                let len = params.next().unwrap().try_into::<i32>().unwrap() as u32;
                assert!(params.next().is_none());

                let len_to_request = u16::try_from(len).unwrap_or(u16::max_value());
                debug_assert!(u32::from(len_to_request) <= len);
                let action = ExtrinsicsAction::EmitMessage {
                    interface: redshirt_random_interface::ffi::INTERFACE,
                    message: redshirt_random_interface::ffi::RandomMessage::Generate {
                        len: len_to_request,
                    }
                    .encode(),
                    response_expected: true,
                };

                let context = ContextInner::WaitRandom {
                    out_ptr: buf,
                    remaining_len: len,
                };

                (Context(context), action)
            }
            ExtrinsicIdInner::SchedYield => {
                // TODO: implement in a better way?
                let context = ContextInner::Finished;
                let action = ExtrinsicsAction::Resume(Some(RuntimeValue::I32(0)));
                (Context(context), action)
            }
        }
    }

    fn inject_message_response(
        &self,
        ctxt: &mut Self::Context,
        response: Option<EncodedMessage>,
        mem_access: &mut impl ExtrinsicsMemoryAccess,
    ) -> ExtrinsicsAction {
        match ctxt.0 {
            ContextInner::WaitRandom {
                mut out_ptr,
                mut remaining_len,
            } => {
                let response = response.unwrap();
                let value: redshirt_random_interface::ffi::GenerateResponse =
                    match response.decode() {
                        Ok(v) => v,
                        Err(_) => return ExtrinsicsAction::ProgramCrash,
                    };

                assert!(
                    u32::try_from(value.result.len()).unwrap_or(u32::max_value())
                        <= u32::from(remaining_len)
                );
                mem_access.write_memory(out_ptr, &value.result).unwrap(); // TODO: don't unwrap

                assert_ne!(value.result.len(), 0); // TODO: don't unwrap
                out_ptr += u32::try_from(value.result.len()).unwrap();
                remaining_len -= u32::try_from(value.result.len()).unwrap();

                if remaining_len == 0 {
                    ctxt.0 = ContextInner::Finished;
                    ExtrinsicsAction::Resume(Some(RuntimeValue::I32(0)))
                } else {
                    let len_to_request = u16::try_from(remaining_len).unwrap_or(u16::max_value());
                    debug_assert!(u32::from(len_to_request) <= remaining_len);

                    ctxt.0 = ContextInner::WaitRandom {
                        out_ptr,
                        remaining_len,
                    };

                    ExtrinsicsAction::EmitMessage {
                        interface: redshirt_random_interface::ffi::INTERFACE,
                        message: redshirt_random_interface::ffi::RandomMessage::Generate {
                            len: len_to_request,
                        }
                        .encode(),
                        response_expected: true,
                    }
                }
            }
            ContextInner::Resume(value) => {
                ctxt.0 = ContextInner::Finished;
                ExtrinsicsAction::Resume(value)
            }
            ContextInner::Finished => unreachable!(),
        }
    }
}

fn fd_write(
    mut params: impl ExactSizeIterator<Item = RuntimeValue>,
    mem_access: &mut impl ExtrinsicsMemoryAccess,
) -> ExtrinsicsAction {
    let fd = params.next().unwrap();
    // TODO: return error if wrong fd
    assert!(fd == RuntimeValue::I32(1) || fd == RuntimeValue::I32(2)); // either stdout or stderr

    // Get a list of pointers and lengths to write.
    // Elements 0, 2, 4, 6, ... in that list are pointers, and elements 1, 3, 5, 7, ... are
    // lengths.
    let list_to_write = {
        let addr = params.next().unwrap().try_into::<i32>().unwrap() as u32;
        let num = params.next().unwrap().try_into::<i32>().unwrap() as u32;
        let list_buf = mem_access.read_memory(addr..addr + 4 * num * 2).unwrap();
        let mut list_out = Vec::with_capacity(usize::try_from(num).unwrap());
        for elem in list_buf.chunks(4) {
            list_out.push(u32::from_le_bytes(<[u8; 4]>::try_from(elem).unwrap()));
        }
        list_out
    };

    let mut total_written = 0;
    let mut encoded_message = Vec::new();
    if fd == RuntimeValue::I32(2) {
        // TODO: handle better?
        encoded_message.push(4); // ERROR log level.
    } else {
        encoded_message.push(2); // INFO log level.
    }

    for ptr_and_len in list_to_write.windows(2) {
        let ptr = ptr_and_len[0] as u32;
        let len = ptr_and_len[1] as u32;

        encoded_message.extend(mem_access.read_memory(ptr..ptr + len).unwrap());
        total_written += len as usize;
    }

    // Write to the fourth parameter the number of bytes written to the file descriptor.
    {
        let out_ptr = params.next().unwrap().try_into::<i32>().unwrap() as u32;
        let total_written = u32::try_from(total_written).unwrap();
        mem_access
            .write_memory(out_ptr, &total_written.to_le_bytes())
            .unwrap();
    }

    assert!(params.next().is_none());

    ExtrinsicsAction::EmitMessage {
        interface: redshirt_log_interface::ffi::INTERFACE,
        message: EncodedMessage(encoded_message),
        response_expected: false,
    }
}
