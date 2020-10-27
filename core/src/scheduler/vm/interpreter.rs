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

use super::{ExecOutcome, NewErr, RunErr, StartErr};
use crate::module::Module;

use alloc::{
    borrow::{Cow, ToOwned as _},
    boxed::Box,
    format,
    vec::Vec,
};
use core::{cell::RefCell, convert::TryInto, fmt};
use smallvec::SmallVec;

pub struct Interpreter<T> {
    /// Original module, with resolved imports.
    module: wasmi::ModuleRef,

    /// Memory of the module instantiation.
    ///
    /// Right now we only support one unique `Memory` object per process. This is it.
    /// Contains `None` if the process doesn't export any memory object, which means it doesn't use
    /// any memory.
    memory: Option<wasmi::MemoryRef>,

    /// Table of the indirect function calls.
    ///
    /// In WASM, function pointers are in reality indices in a table called
    /// `__indirect_function_table`. This is this table, if it exists.
    indirect_table: Option<wasmi::TableRef>,

    /// List of threads that this process is running.
    threads: SmallVec<[ThreadState<T>; 4]>,

    /// If true, the state machine is in a poisoned state and cannot run any code anymore.
    is_poisoned: bool,
}

/// State of a single thread within the VM.
struct ThreadState<T> {
    /// Execution context of this thread. This notably holds the program counter, state of the
    /// stack, and so on.
    ///
    /// This field is an `Option` because we need to be able to temporarily extract it. It must
    /// always be `Some`.
    execution: Option<wasmi::FuncInvocation<'static>>,

    /// If false, then one must call `execution.start_execution()` instead of `resume_execution()`.
    /// This is a particularity of the WASM interpreter that we don't want to expose in our API.
    interrupted: bool,

    /// Opaque user data associated with the thread.
    user_data: T,
}

/// Access to a thread within the virtual machine.
pub struct Thread<'a, T> {
    /// Reference to the parent object.
    vm: &'a mut Interpreter<T>,

    // Index within [`Interpreter::threads`] of the thread we are referencing.
    index: usize,
}

impl<T> Interpreter<T> {
    /// Creates a new process state machine from the given module.
    ///
    /// The closure is called for each import that the module has. It must assign a number to each
    /// import, or return an error if the import can't be resolved. When the VM calls one of these
    /// functions, this number will be returned back in order for the user to know how to handle
    /// the call.
    ///
    /// A single main thread (whose user data is passed by parameter) is automatically created and
    /// is paused at the start of the "_start" function of the module.
    pub fn new(
        module: &Module,
        main_thread_user_data: T,
        mut symbols: impl FnMut(&str, &str, &wasmi::Signature) -> Result<usize, ()>,
    ) -> Result<Self, NewErr> {
        struct ImportResolve<'a> {
            func: RefCell<&'a mut dyn FnMut(&str, &str, &Signature) -> Result<usize, ()>>,
            memory: RefCell<&'a mut Option<wasmi::MemoryRef>>,
        }

        impl<'a> wasmi::ImportResolver for ImportResolve<'a> {
            fn resolve_func(
                &self,
                module_name: &str,
                field_name: &str,
                signature: &wasmi::Signature,
            ) -> Result<wasmi::FuncRef, wasmi::Error> {
                let closure = &mut **self.func.borrow_mut();
                let index = match closure(module_name, field_name, &From::from(signature)) {
                    Ok(i) => i,
                    Err(_) => {
                        return Err(wasmi::Error::Instantiation(format!(
                            "Couldn't resolve `{}`:`{}`",
                            module_name, field_name
                        )))
                    }
                };

                Ok(wasmi::FuncInstance::alloc_host(signature.clone(), index))
            }

            fn resolve_global(
                &self,
                _module_name: &str,
                _field_name: &str,
                _global_type: &wasmi::GlobalDescriptor,
            ) -> Result<wasmi::GlobalRef, wasmi::Error> {
                Err(wasmi::Error::Instantiation(
                    "Importing globals is not supported yet".to_owned(),
                ))
            }

            fn resolve_memory(
                &self,
                _module_name: &str,
                _field_name: &str,
                memory_type: &wasmi::MemoryDescriptor,
            ) -> Result<wasmi::MemoryRef, wasmi::Error> {
                let mut mem = self.memory.borrow_mut();
                if mem.is_some() {
                    return Err(wasmi::Error::Instantiation(
                        "Only one memory object is supported yet".to_owned(),
                    ));
                }

                let new_mem = wasmi::MemoryInstance::alloc(
                    wasmi::memory_units::Pages(usize::try_from(memory_type.initial()).unwrap()),
                    memory_type
                        .maximum()
                        .map(|p| wasmi::memory_units::Pages(usize::try_from(p).unwrap())),
                )
                .unwrap();
                **mem = Some(new_mem.clone());
                Ok(new_mem)
            }

            fn resolve_table(
                &self,
                _module_name: &str,
                _field_name: &str,
                _table_type: &wasmi::TableDescriptor,
            ) -> Result<wasmi::TableRef, wasmi::Error> {
                Err(wasmi::Error::Instantiation(
                    "Importing tables is not supported yet".to_owned(),
                ))
            }
        }

        let (not_started, imported_memory) = {
            let mut imported_memory = None;
            let resolve = ImportResolve {
                func: RefCell::new(&mut symbols),
                memory: RefCell::new(&mut imported_memory),
            };
            let not_started = wasmi::ModuleInstance::new(module.as_ref(), &resolve)
                .map_err(NewErr::Interpreter)?;
            (not_started, imported_memory)
        };

        // TODO: WASM has a special "start" instruction that can be used to designate a function
        // that must be executed before the module is considered initialized. It is unclear whether
        // this is intended to be a function that for example initializes global variables, or if
        // this is an equivalent of "_start". In practice, Rust never seems to generate such as
        // "start" instruction, so for now we ignore it. The code below panics if there is such
        // a "start" item, so we will fortunately not blindly run into troubles.
        let module = not_started.assert_no_start();

        let memory = if let Some(imported_mem) = imported_memory {
            if module
                .export_by_name("memory")
                .map_or(false, |m| m.as_memory().is_some())
            {
                return Err(NewErr::MultipleMemoriesNotSupported);
            }
            Some(imported_mem)
        } else if let Some(mem) = module.export_by_name("memory") {
            if let Some(mem) = mem.as_memory() {
                Some(mem.clone())
            } else {
                return Err(NewErr::MemoryIsntMemory);
            }
        } else {
            None
        };

        let indirect_table = if let Some(tbl) = module.export_by_name("__indirect_function_table") {
            if let Some(tbl) = tbl.as_table() {
                Some(tbl.clone())
            } else {
                return Err(NewErr::IndirectTableIsntTable);
            }
        } else {
            None
        };

        let mut state_machine = Interpreter {
            module,
            memory,
            indirect_table,
            is_poisoned: false,
            threads: SmallVec::new(),
        };

        // Try to start executing `_start` or `main`.
        // TODO: executing `main` is a hack right now in order to support wasm32-unknown-unknown which doesn't have
        // a `_start` function
        match state_machine.start_thread_by_name("_start", &[][..], main_thread_user_data) {
            Ok(_) => {}
            Err((StartErr::FunctionNotFound, user_data)) => {
                static ARGC_ARGV: [wasmi::RuntimeValue; 2] =
                    [wasmi::RuntimeValue::I32(0), wasmi::RuntimeValue::I32(0)];
                match state_machine.start_thread_by_name("main", &ARGC_ARGV[..], user_data) {
                    Ok(_) => {}
                    Err((StartErr::FunctionNotFound, _)) => return Err(NewErr::StartNotFound),
                    Err((StartErr::Poisoned, _)) => unreachable!(),
                    Err((StartErr::NotAFunction, _)) => return Err(NewErr::StartIsntAFunction),
                }
            }
            Err((StartErr::Poisoned, _)) => unreachable!(),
            Err((StartErr::NotAFunction, _)) => return Err(NewErr::StartIsntAFunction),
        };

        Ok(state_machine)
    }

    /// Returns true if the state machine is in a poisoned state and cannot run anymore.
    pub fn is_poisoned(&self) -> bool {
        self.is_poisoned
    }

    /// Starts executing a function. Immediately pauses the execution and puts it in an
    /// interrupted state.
    ///
    /// You should call [`run`](Thread::run) afterwards with a value of `None`.
    ///
    /// > **Note**: The "function ID" is the index of the function in the WASM module. WASM
    /// >           doesn't have function pointers. Instead, all the functions are part of a single
    /// >           global array of functions.
    pub fn start_thread_by_id(
        &mut self,
        function_id: u32,
        params: impl Into<Cow<'static, [wasmi::RuntimeValue]>>,
        user_data: T,
    ) -> Result<Thread<T>, StartErr> {
        if self.is_poisoned {
            return Err(StartErr::Poisoned);
        }

        // Find the function within the process.
        let function = self
            .indirect_table
            .as_ref()
            .and_then(|t| t.get(function_id).ok())
            .and_then(|f| f)
            .ok_or(StartErr::FunctionNotFound)?;

        let execution = match wasmi::FuncInstance::invoke_resumable(&function, params) {
            Ok(e) => e,
            Err(err) => unreachable!("{:?}", err),
        };

        self.threads.push(ThreadState {
            execution: Some(execution),
            interrupted: false,
            user_data,
        });

        let thread_id = self.threads.len() - 1;
        Ok(Thread {
            vm: self,
            index: thread_id,
        })
    }

    /// Same as [`start_thread_by_id`](Interpreter::start_thread_by_id), but executes a
    /// symbol by name.
    fn start_thread_by_name(
        &mut self,
        symbol_name: &str,
        params: impl Into<Cow<'static, [wasmi::RuntimeValue]>>,
        user_data: T,
    ) -> Result<Thread<T>, (StartErr, T)> {
        if self.is_poisoned {
            return Err((StartErr::Poisoned, user_data));
        }

        match self.module.export_by_name(symbol_name) {
            Some(wasmi::ExternVal::Func(f)) => {
                let execution = match wasmi::FuncInstance::invoke_resumable(&f, params) {
                    Ok(e) => e,
                    Err(err) => unreachable!("{:?}", err),
                };
                self.threads.push(ThreadState {
                    execution: Some(execution),
                    interrupted: false,
                    user_data,
                });
            }
            None => return Err((StartErr::FunctionNotFound, user_data)),
            _ => return Err((StartErr::NotAFunction, user_data)),
        }

        let thread_id = self.threads.len() - 1;
        Ok(Thread {
            vm: self,
            index: thread_id,
        })
    }

    /// Returns the number of threads that are running.
    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }

    /// Returns the thread with the given index. The index is between `0` and
    /// [`num_threads`](Interpreter::num_threads).
    ///
    /// The main thread is always index `0`, unless it has terminated.
    ///
    /// Keep in mind that when a thread finishes, all the indices above its index shift by one.
    ///
    /// Returns `None` if the index is superior or equal to what
    /// [`num_threads`](Interpreter::num_threads) would return.
    pub fn thread(&mut self, index: usize) -> Option<Thread<T>> {
        if index < self.threads.len() {
            Some(Thread { vm: self, index })
        } else {
            None
        }
    }

    /// Consumes this VM and returns all the remaining threads' user datas.
    pub fn into_user_datas(self) -> impl ExactSizeIterator<Item = T> {
        self.threads.into_iter().map(|thread| thread.user_data)
    }

    /// Copies the given memory range into a `Vec<u8>`.
    ///
    /// Returns an error if the range is invalid or out of range.
    pub fn read_memory(&self, offset: u32, size: u32) -> Result<Vec<u8>, ()> {
        let mem = match self.memory.as_ref() {
            Some(m) => m,
            None => unreachable!(),
        };

        mem.get(offset, size.try_into().map_err(|_| ())?)
            .map_err(|_| ())
    }

    /// Write the data at the given memory location.
    ///
    /// Returns an error if the range is invalid or out of range.
    pub fn write_memory(&mut self, offset: u32, value: &[u8]) -> Result<(), ()> {
        let mem = match self.memory.as_ref() {
            Some(m) => m,
            None => unreachable!(),
        };

        mem.set(offset, value).map_err(|_| ())
    }
}

// The fields related to `wasmi` do not implement `Send` because they use `std::rc::Rc`. `Rc`
// does not implement `Send` because incrementing/decrementing the reference counter from
// multiple threads simultaneously would be racy. It is however perfectly sound to move all the
// instances of `Rc`s at once between threads, which is what we're doing here.
//
// This importantly means that we should never return a `Rc` (even by reference) across the API
// boundary.
// TODO: really annoying to have to use unsafe code
unsafe impl<T: Send> Send for Interpreter<T> {}

impl<T> fmt::Debug for Interpreter<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.threads.iter()).finish()
    }
}

impl<T> fmt::Debug for ThreadState<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Thread").field(&self.user_data).finish()
    }
}

impl<'a, T> Thread<'a, T> {
    /// Starts or continues execution of this thread.
    ///
    /// If this is the first call you call [`run`](Thread::run) for this thread, then you must pass
    /// a value of `None`.
    /// If, however, you call this function after a previous call to [`run`](Thread::run) that was
    /// interrupted by an external function call, then you must pass back the outcome of that call.
    pub fn run(mut self, value: Option<wasmi::RuntimeValue>) -> Result<ExecOutcome<'a, T>, RunErr> {
        struct DummyExternals;
        impl wasmi::Externals for DummyExternals {
            fn invoke_index(
                &mut self,
                index: usize,
                args: wasmi::RuntimeArgs,
            ) -> Result<Option<wasmi::RuntimeValue>, wasmi::Trap> {
                Err(wasmi::TrapKind::Host(Box::new(Interrupt {
                    index,
                    args: args.as_ref().to_vec(),
                }))
                .into())
            }
        }

        #[derive(Debug)]
        struct Interrupt {
            index: usize,
            args: Vec<wasmi::RuntimeValue>,
        }
        impl fmt::Display for Interrupt {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "Interrupt")
            }
        }
        impl wasmi::HostError for Interrupt {}

        if self.vm.is_poisoned {
            return Err(RunErr::Poisoned);
        }

        let thread_state = &mut self.vm.threads[self.index];

        let mut execution = match thread_state.execution.take() {
            Some(e) => e,
            None => unreachable!(),
        };
        let result = if thread_state.interrupted {
            let expected_ty = execution.resumable_value_type();
            let obtained_ty = value.as_ref().map(|v| v.value_type());
            if expected_ty != obtained_ty {
                return Err(RunErr::BadValueTy {
                    expected: expected_ty,
                    obtained: obtained_ty,
                });
            }
            execution.resume_execution(value, &mut DummyExternals)
        } else {
            if value.is_some() {
                return Err(RunErr::BadValueTy {
                    expected: None,
                    obtained: value.as_ref().map(|v| v.value_type()),
                });
            }
            thread_state.interrupted = true;
            execution.start_execution(&mut DummyExternals)
        };

        match result {
            Ok(return_value) => {
                let user_data = self.vm.threads.remove(self.index).user_data;
                // If this is the "main" function, the state machine is now poisoned.
                if self.index == 0 {
                    self.vm.is_poisoned = true;
                }
                Ok(ExecOutcome::ThreadFinished {
                    thread_index: self.index,
                    return_value,
                    user_data,
                })
            }
            Err(wasmi::ResumableError::AlreadyStarted) => unreachable!(),
            Err(wasmi::ResumableError::NotResumable) => unreachable!(),
            Err(wasmi::ResumableError::Trap(ref trap)) if trap.kind().is_host() => {
                let interrupt: &Interrupt = match trap.kind() {
                    wasmi::TrapKind::Host(err) => match err.downcast_ref() {
                        Some(e) => e,
                        None => unreachable!(),
                    },
                    _ => unreachable!(),
                };
                thread_state.execution = Some(execution);
                Ok(ExecOutcome::Interrupted {
                    thread: From::from(self),
                    id: interrupt.index,
                    params: interrupt.args.clone(),
                })
            }
            Err(wasmi::ResumableError::Trap(trap)) => {
                self.vm.is_poisoned = true;
                Ok(ExecOutcome::Errored {
                    thread: From::from(self),
                    error: trap,
                })
            }
        }
    }

    /// Returns the index of the thread, so that you can retreive the thread later by calling
    /// [`Interpreter::thread`].
    ///
    /// Keep in mind that when a thread finishes, all the indices above its index shift by one.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the user data associated to that thread.
    pub fn user_data(&mut self) -> &mut T {
        &mut self.vm.threads[self.index].user_data
    }

    /// Turns this thread into the user data associated to it.
    pub fn into_user_data(self) -> &'a mut T {
        &mut self.vm.threads[self.index].user_data
    }
}

impl<'a, T> fmt::Debug for Thread<'a, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.vm.threads[self.index], f)
    }
}
