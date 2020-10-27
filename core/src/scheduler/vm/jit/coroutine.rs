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

use alloc::rc::Rc;
use core::{cell::Cell, marker::PhantomData, pin::Pin};

// Documentation about the x86_64 ABI here: https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI

// TODO: make it works on 32bits
// TODO: the UnwindSafe trait should be enforced but isn't available in core
// TODO: require Send trait for the closure? it's a bit tricky, need to look into details
// TODO: will leak heap-allocated stuff if Dropped before it's finished

/// Prototype for a [`Coroutine`].
pub struct CoroutineBuilder<TInt, TRes> {
    stack_size: usize,
    state: Rc<Shared<TInt, TRes>>,
}

/// Runnable coroutine.
pub struct Coroutine<TExec, TInt, TRes> {
    /// We use a `u128` because the stack must be 16-bytes-aligned.
    stack: Pin<Box<[u128]>>,
    /// State shared between the coroutine and the interrupters.
    state: Rc<Shared<TInt, TRes>>,
    /// True if the coroutine has been run at least once.
    has_run_once: bool,
    /// True if the coroutine has finished. Trying to resume running will panic.
    has_finished: bool,
    marker: PhantomData<TExec>,
}

/// Object whose intent is to be stored in the closure and that is capable of interrupting
/// execution.
pub struct Interrupter<TInt, TRes> {
    state: Rc<Shared<TInt, TRes>>,
}

struct Shared<TInt, TRes> {
    /// Value to put in `rsp` in order to resume the coroutine.
    ///
    /// Must point somewhere within [`Coroutine::stack`]. Will contain null before initialization.
    ///
    /// Must point to a memory location that contains the values of the following registers in
    /// order: r15, r14, r13, r12, rbp, rbx, rsi, rip
    ///
    /// In order to resume the coroutine, set `rsp` to this value then pop all the registers.
    coroutine_stack_pointer: Cell<usize>,

    /// Stack pointer of the caller. Only valid if we are within the coroutine.
    caller_stack_pointer: Cell<usize>,

    /// Where to write the return value.
    potential_return_value_ptr: Cell<usize>,
    /// Storage where to write the value yielded to outside the coroutine before jumping out,
    /// or left to `None` if the coroutine has terminated.
    interrupt_val: Cell<Option<TInt>>,

    /// Storage where to write the value yielded back *to* the coroutine before resuming
    /// execution.
    resume_value: Cell<Option<TRes>>,
}

impl<TInt, TRes> Default for CoroutineBuilder<TInt, TRes> {
    fn default() -> Self {
        Self::new()
    }
}

impl<TInt, TRes> CoroutineBuilder<TInt, TRes> {
    /// Starts a builder.
    pub fn new() -> Self {
        CoroutineBuilder {
            // TODO: no stack protection :(
            stack_size: 4 * 1024 * 1024,
            state: Rc::new(Shared {
                coroutine_stack_pointer: Cell::new(0),
                caller_stack_pointer: Cell::new(0),
                potential_return_value_ptr: Cell::new(0),
                interrupt_val: Cell::new(None),
                resume_value: Cell::new(None),
            }),
        }
    }

    /// Creates a new [`Interrupter`] to store within the closure1.
    // TODO: it's super unsafe to use an interrupter with a different closure than the one passed
    // to build
    pub fn interrupter(&self) -> Interrupter<TInt, TRes> {
        Interrupter {
            state: self.state.clone(),
        }
    }

    /// Builds the coroutine.
    pub fn build<TRet, TExec: FnOnce() -> TRet>(
        self,
        to_exec: TExec,
    ) -> Coroutine<TExec, TInt, TRes> {
        let to_actually_exec = Box::into_raw({
            let state = self.state.clone();
            // TODO: panic safety handling here
            let closure = Box::new(move |caller_stack_pointer| {
                state.caller_stack_pointer.set(caller_stack_pointer);
                let ret_value = to_exec();
                let ptr =
                    unsafe { &mut *(state.potential_return_value_ptr.get() as *mut Option<TRet>) };
                assert!(ptr.is_none());
                *ptr = Some(ret_value);
                unsafe {
                    coroutine_switch_stack(state.caller_stack_pointer.get());
                    core::hint::unreachable_unchecked()
                }
            }) as Box<dyn FnOnce(usize)>;
            Box::new(closure) as Box<Box<dyn FnOnce(usize)>>
        });

        let mut stack = Pin::new(unsafe {
            let stack = Box::new_uninit_slice(1 + (self.stack_size.checked_sub(1).unwrap() / 16));
            stack.assume_init()
        });

        unsafe {
            // Starting the stack of top, we virtually push a fake return address, plus our 8 saved
            // registers.
            // The fake return address is important is order to guarantee the proper alignment of
            // the stack.
            let stack_top = (stack.as_mut_ptr() as *mut u64)
                .add(self.stack_size / 8)
                .sub(9);

            stack_top.add(6).write(to_actually_exec as usize as u64); // RSI
            let rip = start_call as extern "C" fn(usize, usize) as usize as u64;
            stack_top.add(7).write(rip);
            stack_top.add(8).write(0);

            self.state.coroutine_stack_pointer.set(stack_top as usize);
        }

        Coroutine {
            stack,
            state: self.state.clone(),
            has_run_once: false,
            has_finished: false,
            marker: PhantomData,
        }
    }
}

/// Return value of [`Coroutine::run`].
pub enum RunOut<TRet, TInt> {
    /// The coroutine has finished. Contains the return value of the closure.
    Finished(TRet),
    /// The coroutine has called [`Interrupter::interrupt`]
    Interrupted(TInt),
}

impl<TExec: FnOnce() -> TRet, TRet, TInt, TRes> Coroutine<TExec, TInt, TRes> {
    /// Returns true if running the closure has produced a [`RunOut::Finished`] earlier.
    pub fn is_finished(&self) -> bool {
        self.has_finished
    }

    /// Runs the coroutine until it finishes or is interrupted.
    ///
    /// `resume` must be `None` the first time a coroutine is run, then must be `Some` with the
    /// value to reinject back as the return value of [`Interrupter::interrupt`].
    ///
    /// # Panic
    ///
    /// Panics if [`RunOut::Finished`] has been returned earlier.
    /// Panics if `None` is passed and it is not the first time the coroutine is being run.
    /// Panics if `Some` is passed and it is the first time the coroutine is being run.
    ///
    pub fn run(&mut self, resume: Option<TRes>) -> RunOut<TRet, TInt> {
        assert!(!self.has_finished);

        if !self.has_run_once {
            assert!(resume.is_none());
            self.has_run_once = true;
        } else {
            assert!(resume.is_some());
        }

        // Store the resume value for the coroutine to pick up.
        debug_assert!(self.state.resume_value.take().is_none());
        self.state.resume_value.set(resume);

        // We allocate some space where the coroutine is allowed to put its return value, and put
        // a pointer to this space in `self.state`.
        let mut potential_return_value = None::<TRet>;
        self.state
            .potential_return_value_ptr
            .set(&mut potential_return_value as *mut _ as usize);

        // Doing a jump to the coroutine, which will then jump back here once it interrupts or
        // finishes.
        let new_stack_ptr =
            unsafe { coroutine_switch_stack(self.state.coroutine_stack_pointer.get()) };
        self.state.coroutine_stack_pointer.set(new_stack_ptr);
        debug_assert!(self.state.resume_value.take().is_none());

        // We determine whether the function has ended or is simply interrupted based on the
        // content of `self.state`.
        if let Some(interrupted) = self.state.interrupt_val.take() {
            debug_assert!(potential_return_value.take().is_none());
            RunOut::Interrupted(interrupted)
        } else {
            self.has_finished = true;
            RunOut::Finished(potential_return_value.take().unwrap())
        }
    }
}

impl<TInt, TRes> Interrupter<TInt, TRes> {
    /// Interrupts the current execution flow and jumps back to the [`Coroutine::run`] function,
    /// which will then return a [`RunOut::Interrupted`] containing the value passed as parameter.
    pub fn interrupt(&self, val: TInt) -> TRes {
        debug_assert!(self.state.interrupt_val.take().is_none());
        self.state.interrupt_val.set(Some(val));

        let new_caller_ptr =
            unsafe { coroutine_switch_stack(self.state.caller_stack_pointer.get()) };
        self.state.caller_stack_pointer.set(new_caller_ptr);

        self.state.resume_value.take().unwrap()
    }
}

impl<TInt, TRes> Clone for Interrupter<TInt, TRes> {
    fn clone(&self) -> Self {
        Interrupter {
            state: self.state.clone(),
        }
    }
}

/// Function whose role is to bootstrap the coroutine.
///
/// `caller_stack_pointer` is the value produced by [`coroutine_switch_stack`], and [`to_exec`]
/// is a pointer to the closure to execute. The closure must never return.
// TODO: turn `Box<dyn FnOnce(usize)>` into `Box<dyn FnOnce(usize) -> !>` when `!` is stable
extern "C" fn start_call(caller_stack_pointer: usize, to_exec: usize) {
    unsafe {
        let to_exec: Box<Box<dyn FnOnce(usize)>> =
            Box::from_raw(to_exec as *mut Box<dyn FnOnce(usize)>);
        (*to_exec)(caller_stack_pointer);
        core::hint::unreachable_unchecked()
    }
}

extern "C" {
    /// Pushes the current processor's state on the stack, then sets the stack pointer to the
    /// given value and pops the processor's state back. The "processor's state" includes the
    /// %rip register, which means that we're essentially returning somewhere else than where this
    /// function has been called.
    ///
    /// Before returning, this function sets the `%rax` and `%rdi` registers to the stack pointer
    /// that contains the state of the caller. This is shown here as a return value, but it also
    /// means that the poped `%rip` can be the entry point of a function that will thus accept
    /// as first parameter the value of this stack pointer.
    ///
    /// The state of the caller can later be restored by calling this function again with the
    /// value that it produced.
    ///
    /// Contrary to pre-emptive multitasking systems, we don't need to save the entire state. We
    /// only need to save the registers that the caller expects to not change, as defined by the
    /// ABI.
    fn coroutine_switch_stack(stack: usize) -> usize;
}
// TODO: we would like to use a naked function in order to have mangled function name and possibly
// avoid collisions, but if we do so the compiler still inserts a `mov %rdi, ...` at the top.
global_asm! {r#"
.global coroutine_switch_stack
coroutine_switch_stack:
    push %rsi
    push %rbx
    push %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    mov %rsp, %rax
    mov %rdi, %rsp
    mov %rax, %rdi
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    pop %rbx
    pop %rsi
    ret
"#}

#[cfg(test)]
mod tests {
    use super::{CoroutineBuilder, RunOut};

    #[test]
    fn basic_works() {
        let mut coroutine = CoroutineBuilder::<(), ()>::new().build(|| 12);
        match coroutine.run(None) {
            RunOut::Finished(12) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn basic_interrupter() {
        let builder = CoroutineBuilder::new();
        let interrupter = builder.interrupter();
        let mut coroutine = builder.build(|| {
            let val = interrupter.interrupt(format!("hello world {}", 53));
            val + 12
        });

        match coroutine.run(None) {
            RunOut::Interrupted(val) => assert_eq!(val, "hello world 53"),
            _ => panic!(),
        }

        match coroutine.run(Some(5)) {
            RunOut::Finished(17) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn many_interruptions() {
        let builder = CoroutineBuilder::new();
        let interrupter = builder.interrupter();
        let mut coroutine = builder.build(|| {
            let mut val = 0;
            for _ in 0..1000 {
                val = interrupter.interrupt(format!("hello! {}", val));
            }
            val
        });

        let mut val = None;
        loop {
            match coroutine.run(val) {
                RunOut::Interrupted(v) => {
                    assert_eq!(v, format!("hello! {}", val.unwrap_or(0)));
                    val = Some(val.unwrap_or(0) + 1);
                }
                RunOut::Finished(v) => {
                    assert_eq!(v, val.unwrap());
                    break;
                }
            }
        }
    }
}
