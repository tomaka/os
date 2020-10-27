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

use crate::{module::Module, primitives::Signature, ValueType, WasmValue};

use alloc::vec::Vec;
use core::fmt;

#[cfg(target_arch = "x86_64")]
mod jit;
#[cfg(target_arch = "x86_64")]
use jit::{Jit as ImpStateMachine, Thread as ImpThread};

#[cfg(not(target_arch = "x86_64"))]
mod interpreter;
#[cfg(not(target_arch = "x86_64"))]
use interpreter::{Interpreter as ImpStateMachine, Thread as ImpThread};

/// WASMI state machine dedicated to a process.
///
/// # Initialization
///
/// Initializing a state machine is done by passing a [`Module`](crate::module::Module) object,
/// which holds a successfully-parsed WASM binary.
///
/// The module might contain a list of functions to import and that the initialization process
/// must resolve. When such an import is encountered, the closure passed to the
/// [`new`](ProcessStateMachine::new) function is invoked and must return an opaque integer decided
/// by the user. This integer is later passed back to the user of this struct in the situation when
/// the state machine invokes that external function.
///
/// # Threads
///
/// This struct is composed of one or multiple threads. When initialized, the VM starts with a
/// single thread at the start of the "main" function of the WASM module.
///
/// In order to run the VM, grab a thread by calling [`ProcessStateMachine::threads`], then call
/// [`Thread::run`]. The thread will then run until it either finishes (in which case the thread
/// is then destroyed), or attempts to call an imported function.
///
/// TODO: It is intended that in the future the `run` function stops after a certain period of
/// time has elapsed, in order to do preemptive multithreading. This requires a lot of changes in
/// the interpreter, and isn't going to happen any time soon.
///
/// The [`run`](Thread::run) method requires passing a value. The first time you call
/// [`run`](Thread::run) for any given thread, you must pass the value `None`. If that thread is
/// then interrupted by a call to an imported function, you must execute the imported function and
/// pass its return value the next time you call [`run`](Thread::run).
///
/// The generic parameter of this struct is some userdata that is associated with each thread.
/// You must pass a value when creating a thread, and can retreive it later by calling
/// [`user_data`](Thread::user_data) or [`into_user_data`](Thread::into_user_data).
///
/// # Poisoning
///
/// If the main thread stops, or if any thread encounters an error, then the VM moves into a
/// "poisoned" state. It is then no longer possible to run anything in it. Threads are kept alive
/// so that you can examine their state, but attempting to call [`run`](Thread::run) will return
/// an error.
///
/// # Single-threaded-ness
///
/// The [`ProcessStateMachine`] is single-threaded. In other words, the VM can only ever run one
/// thread simultaneously. This might change in the future.
///
pub struct ProcessStateMachine<T>(ImpStateMachine<T>);

/// Access to a thread within the virtual machine.
pub struct Thread<'a, T>(ImpThread<'a, T>);

/// Outcome of the [`run`](Thread::run) function.
#[derive(Debug)]
pub enum ExecOutcome<'a, T> {
    /// A thread has finished. The thread no longer exists in the list.
    ///
    /// If this was the main thread (i.e. `thread_index` is 0), then the state machine is now in
    /// a poisoned state, and calling [`is_poisoned`](ProcessStateMachine::is_poisoned) will
    /// return true.
    ///
    /// > **Note**: Keep in mind that how you want to react to this event is probably very
    /// >           different depending on whether `thread_index` is 0. If this is the main thread,
    /// >           then the entire process is now stopped.
    ///
    ThreadFinished {
        /// Index of the thread that has finished.
        thread_index: usize,

        /// Return value of the thread function.
        return_value: Option<WasmValue>,

        /// User data that was stored within the thread.
        user_data: T,
    },

    /// The currently-executed thread has been paused due to a call to an external function.
    ///
    /// This variant contains the identifier of the external function that is expected to be
    /// called, and its parameters. When you call [`run`](Thread::run) again, you must pass back
    /// the outcome of calling that function.
    ///
    /// > **Note**: The type of the return value of the function is called is not specified, as the
    /// >           user is supposed to know it based on the identifier. It is an error to call
    /// >           [`run`](Thread::run) with a value of the wrong type.
    Interrupted {
        /// Thread that was interrupted.
        thread: Thread<'a, T>,

        /// Identifier of the function to call. Corresponds to the value provided at
        /// initialization when resolving imports.
        id: usize,

        /// Parameters of the function call.
        params: Vec<WasmValue>,
    },

    /// The currently-executed function has finished with an error. The state machine is now in a
    /// poisoned state.
    ///
    /// Calling [`is_poisoned`](ProcessStateMachine::is_poisoned) will return true.
    Errored {
        /// Thread that error'd.
        thread: Thread<'a, T>,

        /// Error that happened.
        // TODO: error type should change here
        error: wasmi::Trap,
    },
}

/// Error that can happen when initializing a VM.
#[derive(Debug)]
pub enum NewErr {
    /// Error in the interpreter.
    Interpreter(wasmi::Error),
    /// The "start" symbol doesn't exist.
    StartNotFound,
    /// The "start" symbol must be a function.
    StartIsntAFunction,
    /// If a "memory" symbol is provided, it must be a memory.
    MemoryIsntMemory,
    /// A memory object has both been imported and exported.
    MultipleMemoriesNotSupported,
    /// If a "__indirect_function_table" symbol is provided, it must be a table.
    IndirectTableIsntTable,
}

/// Error that can happen when starting a new thread.
#[derive(Debug)]
pub enum StartErr {
    /// The state machine is poisoned and cannot run anymore.
    Poisoned,
    /// Couldn't find the requested function.
    FunctionNotFound,
    /// The requested function has been found in the list of exports, but it is not a function.
    NotAFunction,
}

/// Error that can happen when resuming the execution of a function.
#[derive(Debug)]
pub enum RunErr {
    /// The state machine is poisoned.
    Poisoned,
    /// Passed a wrong value back.
    BadValueTy {
        /// Type of the value that was expected.
        expected: Option<ValueType>,
        /// Type of the value that was actually passed.
        obtained: Option<ValueType>,
    },
}

impl<T> ProcessStateMachine<T> {
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
        symbols: impl FnMut(&str, &str, &Signature) -> Result<usize, ()>,
    ) -> Result<Self, NewErr> {
        Ok(ProcessStateMachine(ImpStateMachine::new(
            module,
            main_thread_user_data,
            symbols,
        )?))
    }

    /// Returns true if the state machine is in a poisoned state and cannot run anymore.
    pub fn is_poisoned(&self) -> bool {
        self.0.is_poisoned()
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
        params: impl IntoIterator<Item = WasmValue>,
        user_data: T,
    ) -> Result<Thread<T>, StartErr> {
        Ok(Thread(self.0.start_thread_by_id(
            function_id,
            params,
            user_data,
        )?))
    }

    /// Returns the number of threads that are running.
    pub fn num_threads(&self) -> usize {
        self.0.num_threads()
    }

    /// Returns the thread with the given index. The index is between `0` and
    /// [`num_threads`](ProcessStateMachine::num_threads).
    ///
    /// The main thread is always index `0`, unless it has terminated.
    ///
    /// Keep in mind that when a thread finishes, all the indices above its index shift by one.
    ///
    /// Returns `None` if the index is superior or equal to what
    /// [`num_threads`](ProcessStateMachine::num_threads) would return.
    pub fn thread(&mut self, index: usize) -> Option<Thread<T>> {
        Some(Thread(self.0.thread(index)?))
    }

    /// Consumes this VM and returns all the remaining threads' user datas.
    pub fn into_user_datas(self) -> impl ExactSizeIterator<Item = T> {
        self.0.into_user_datas()
    }

    /// Copies the given memory range into a `Vec<u8>`.
    ///
    /// Returns an error if the range is invalid or out of range.
    pub fn read_memory(&self, offset: u32, size: u32) -> Result<Vec<u8>, ()> {
        self.0.read_memory(offset, size)
    }

    /// Write the data at the given memory location.
    ///
    /// Returns an error if the range is invalid or out of range.
    pub fn write_memory(&mut self, offset: u32, value: &[u8]) -> Result<(), ()> {
        self.0.write_memory(offset, value)
    }
}

impl<T> From<ImpStateMachine<T>> for ProcessStateMachine<T> {
    fn from(t: ImpStateMachine<T>) -> Self {
        Self(t)
    }
}

impl<T> fmt::Debug for ProcessStateMachine<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<'a, T> Thread<'a, T> {
    /// Starts or continues execution of this thread.
    ///
    /// If this is the first call you call [`run`](Thread::run) for this thread, then you must pass
    /// a value of `None`.
    /// If, however, you call this function after a previous call to [`run`](Thread::run) that was
    /// interrupted by an external function call, then you must pass back the outcome of that call.
    pub fn run(self, value: Option<WasmValue>) -> Result<ExecOutcome<'a, T>, RunErr> {
        self.0.run(value)
    }

    /// Returns the index of the thread, so that you can retreive the thread later by calling
    /// [`ProcessStateMachine::thread`].
    ///
    /// Keep in mind that when a thread finishes, all the indices above its index shift by one.
    pub fn index(&self) -> usize {
        self.0.index()
    }

    /// Returns the user data associated to that thread.
    pub fn user_data(&mut self) -> &mut T {
        self.0.user_data()
    }

    /// Turns this thread into the user data associated to it.
    pub fn into_user_data(self) -> &'a mut T {
        self.0.into_user_data()
    }
}

impl<'a, T> From<ImpThread<'a, T>> for Thread<'a, T> {
    fn from(t: ImpThread<'a, T>) -> Self {
        Self(t)
    }
}

impl<'a, T> fmt::Debug for Thread<'a, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for NewErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NewErr::Interpreter(err) => write!(f, "Error in the interpreter: {}", err),
            NewErr::StartNotFound => write!(f, "The \"start\" symbol doesn't exist"),
            NewErr::StartIsntAFunction => write!(f, "The \"start\" symbol must be a function"),
            NewErr::MemoryIsntMemory => {
                write!(f, "If a \"memory\" symbol is provided, it must be a memory")
            }
            NewErr::MultipleMemoriesNotSupported => {
                write!(f, "A memory object has both been imported and exported")
            }
            NewErr::IndirectTableIsntTable => write!(
                f,
                "If a \"__indirect_function_table\" symbol is provided, it must be a table"
            ),
        }
    }
}

impl fmt::Display for StartErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StartErr::Poisoned => write!(f, "State machine is in a poisoned state"),
            StartErr::FunctionNotFound => write!(f, "Function to start was not found"),
            StartErr::NotAFunction => write!(f, "Symbol to start is not a function"),
        }
    }
}

impl fmt::Display for RunErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RunErr::Poisoned => write!(f, "State machine is poisoned"),
            RunErr::BadValueTy { expected, obtained } => write!(
                f,
                "Expected value of type {:?} but got {:?} instead",
                expected, obtained
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecOutcome, NewErr, ProcessStateMachine};
    use crate::primitives::WasmValue;

    #[test]
    fn starts_if_main() {
        let module = from_wat!(
            local,
            r#"(module
            (func $_start (result i32)
                i32.const 5)
            (export "_start" (func $_start)))
        "#
        );

        let _state_machine =
            ProcessStateMachine::new(&module, (), |_, _, _| unreachable!()).unwrap();
    }

    #[test]
    fn error_if_no_main() {
        let module = from_wat!(
            local,
            r#"(module
            (func $_start (result i32)
                i32.const 5)
            (export "foo" (func $_start)))
        "#
        );

        match ProcessStateMachine::new(&module, (), |_, _, _| unreachable!()) {
            Err(NewErr::StartNotFound) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn main_executes() {
        let module = from_wat!(
            local,
            r#"(module
            (func $_start (result i32)
                i32.const 5)
            (export "_start" (func $_start)))
        "#
        );

        let mut state_machine =
            ProcessStateMachine::new(&module, (), |_, _, _| unreachable!()).unwrap();
        match state_machine.thread(0).unwrap().run(None) {
            Ok(ExecOutcome::ThreadFinished {
                return_value: Some(WasmValue::I32(5)),
                ..
            }) => {}
            _ => panic!(),
        }
        assert!(state_machine.thread(0).is_none());
    }

    #[test]
    fn external_call_then_resume() {
        let module = from_wat!(
            local,
            r#"(module
            (import "" "test" (func $test (result i32)))
            (func $_start (result i32)
                call $test)
            (export "_start" (func $_start)))
        "#
        );

        let mut state_machine = ProcessStateMachine::new(&module, (), |_, _, _| Ok(9876)).unwrap();
        match state_machine.thread(0).unwrap().run(None) {
            Ok(ExecOutcome::Interrupted {
                id: 9876,
                ref params,
                ..
            }) if params.is_empty() => {}
            _ => panic!(),
        }

        match state_machine
            .thread(0)
            .unwrap()
            .run(Some(WasmValue::I32(2227)))
        {
            Ok(ExecOutcome::ThreadFinished {
                return_value: Some(WasmValue::I32(2227)),
                ..
            }) => {}
            _ => panic!(),
        }
        assert!(state_machine.thread(0).is_none());
    }

    #[test]
    fn poisoning_works() {
        let module = from_wat!(
            local,
            r#"(module
            (func $_start
                unreachable)
            (export "_start" (func $_start)))
        "#
        );

        let mut state_machine =
            ProcessStateMachine::new(&module, (), |_, _, _| unreachable!()).unwrap();
        match state_machine.thread(0).unwrap().run(None) {
            Ok(ExecOutcome::Errored { .. }) => {}
            _ => panic!(),
        }

        assert!(state_machine.is_poisoned());

        // TODO: start running another function and check that `Poisoned` error is returned
    }

    // TODO: start mutiple threads
}
