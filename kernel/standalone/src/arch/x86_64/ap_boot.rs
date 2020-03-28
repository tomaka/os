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

//! Bootstrapping associated processors.
//!
//! On x86 and x86_64 platforms, processors are divided in two categories: one BSP (bootstrap
//! processor) and zero or more APs (associated processors). Only the BSP initially starts,
//! and the APs have to be manually started either from the BSP or an AP that has previously
//! been started. This what this module is responsible for doing.
//!
//! # Usage
//!
//! - Create an [`ApBootAlloc`] using the [`filter_build_ap_boot_alloc`] function.
//! - Determine the [`ApicId`] of the processors we want to wake up.
//! - Call [`boot_associated_processor`] for each processor one by one.
//!

use crate::arch::x86_64::apic::{local::LocalApicsControl, timers::Timers, ApicId};
use crate::arch::x86_64::{executor, interrupts};

use alloc::{alloc::Layout, boxed::Box};
use core::{convert::TryFrom as _, ops::Range, ptr, slice, time::Duration};
use futures::channel::oneshot;

/// Allocator required by the [`boot_associated_processor`] function.
pub struct ApBootAlloc {
    inner: linked_list_allocator::Heap,
}

/// Accepts as input an iterator to a list of free memory ranges. If the `Option` is `None`,
/// filters out a range of memory, builds an [`ApBootAlloc`] out of it and puts it in the
/// `Option`.
///
/// # Usage
///
/// ```norun
/// use core::iter;
/// let mut alloc = None;
/// let remaining_ranges = filter_build_ap_boot_alloc(iter::once(0 .. 0x1000000), &mut alloc);
/// let alloc = alloc.expect("Couldn't find free memory range");
/// ```
///
/// # Safety
///
/// The memory ranges have to be RAM or behave like RAM (i.e. both readable and writable,
/// consistent, and so on). The memory ranges must not be touched by anything (other than the
/// allocator) afterwards.
///
pub unsafe fn filter_build_ap_boot_alloc<'a>(
    ranges: impl Iterator<Item = Range<usize>> + 'a,
    alloc: &'a mut Option<ApBootAlloc>,
) -> impl Iterator<Item = Range<usize>> + 'a {
    // Size that we grab from the ranges.
    // TODO: This value is kind of arbitrary for now. Once
    // https://github.com/rust-lang/rust/issues/51910 is stabilized, we can instead compute this
    // value from `_ap_boot_end - _ap_boot_start`.
    const WANTED: usize = 0x4000;

    ranges.filter_map(move |mut range| {
        if alloc.is_none() {
            let range_size = range.end.checked_sub(range.start).unwrap();
            if range.start.saturating_add(WANTED) <= 0x100000 && range_size >= WANTED {
                *alloc = Some(ApBootAlloc {
                    inner: linked_list_allocator::Heap::new(range.start, WANTED),
                });
                if range_size > WANTED {
                    return Some(range.start + WANTED..range.end);
                }
                return None;
            }
        }

        Some(range)
    })
}

/// Bootstraps the given processor, making it execute `boot_code`.
///
/// This function returns once the target processor has been successfully initialized and has
/// started (or will soon start) to execute `boot_code`.
///
/// This function also takes care to enable the local APIC and interrupts on the newly-started
/// processor.
///
/// > **Note**: It is safe to call this function multiple times simultaneously with multiple
/// >           different targets. For example, processor 0 can boot up processor 2 while
/// >           processor 1 simultaneously boots up processor 3.
///
/// # Safety
///
/// `target` must not be the local processor.
///
/// This function must only be called once per `target`, and no other code has sent or attempted
/// to send an INIT or SIPI to the target processor.
///
// TODO: it kind of sucks that an important detail such as "we also initialize the local APIC and
// interrupts" is just part of the documentation, but we need them for the implementation to work
// TODO: replace `Infallible` with `!` when stable
pub unsafe fn boot_associated_processor(
    alloc: &mut ApBootAlloc,
    executor: &executor::Executor,
    local_apics: &'static LocalApicsControl,
    timers: &Timers,
    target: ApicId,
    boot_code: impl FnOnce() -> core::convert::Infallible + Send + 'static,
) {
    // In order to boot an associated processor, we must send to it an inter-processor interrupt
    // (IPI) containing the offset of a 4kiB memory page containing the code that it must start
    // executing. The CS register of the target processor will take the value that we send, the
    // IP register will be set to 0, and the processor will start running in 16 bits mode.
    //
    // Since this is 16 bits mode, the processor cannot execute any code (or access any data)
    // above one megabyte of physical memory. Most, if not all, of the kernel is loaded above
    // that limit, and therefore we cannot simply ask the processor to start executing a certain
    // function as we would like to.
    //
    // Instead, what we must do is allocate a buffer below that one megabyte limit, write some
    // x86 machine code in that buffer, and then we can ask the processor to run it. This is
    // implemented by copying a template code into that buffer and tweaking the constants.

    // We start by allocating the buffer where to write the bootstrap code.
    let mut bootstrap_code_buf = {
        let size = (_ap_boot_end as *const u8 as usize)
            .checked_sub(_ap_boot_start as *const u8 as usize)
            .unwrap();
        // Basic sanity check to make sure that nothing's fundamentally wrong.
        assert!(size <= 0x1000);
        let layout = Layout::from_size_align(size, 0x1000).unwrap();
        Allocation::new(&mut alloc.inner, layout)
    };

    // Start by sending an INIT IPI to the target so that it reboots.
    local_apics.send_interprocessor_init(target);

    let rdtsc = core::arch::x86_64::_rdtsc(); // TODO: crappy code

    // Write the template code to the buffer.
    ptr::copy_nonoverlapping(
        _ap_boot_start as *const u8,
        bootstrap_code_buf.as_mut_ptr(),
        bootstrap_code_buf.size(),
    );

    // Later, we will want to wait until the AP has finished initializing. To do so, we create
    // a channel and modify `boot_code` to signal that channel before doing anything more.
    let (boot_code, init_finished_future) = {
        let (tx, rx) = oneshot::channel();
        let boot_code = move || {
            local_apics.init_local();
            interrupts::load_idt();
            let _ = tx.send(());
            boot_code()
        };
        (boot_code, rx)
    };

    // We want the processor we bootstrap to call the `ap_after_boot` function defined below.
    // `ap_after_boot` will cast its first parameter into a `Box<Box<dyn FnOnce()>>` and call it.
    // We therefore cast `boot_code` into the proper format, then leak it with the intent to pass
    // this value to `ap_after_boot`, which will then "unleak" it and call it.
    let ap_after_boot_param = {
        let boxed = Box::new(Box::new(boot_code) as Box<_>);
        let param_value: ApAfterBootParam = Box::into_raw(boxed);
        u64::try_from(param_value as usize).unwrap()
    };

    // Allocate a stack for the processor. This is the one and unique stack that will be used for
    // everything by this processor.
    let stack_size = 10 * 1024 * 1024usize;
    let stack_top = {
        let layout = Layout::from_size_align(stack_size, 0x1000).unwrap();
        let ptr = alloc::alloc::alloc(layout);
        assert!(!ptr.is_null());
        u64::try_from(ptr as usize + stack_size).unwrap()
    };

    // There exists several placeholders within the template code that we must adjust before it
    // can be executed.
    //
    // The code at symbol `_ap_boot_marker1` starts with the following instruction:
    //
    // ```
    // 66 ea ad de ad de 08    ljmpl  $0x8, $0xdeaddead
    // ```
    //
    // The code at symbol `_ap_boot_marker3` starts with the following instruction:
    //
    // ```
    // 66 ba dd ba 00 ff    mov $0xff00badd, %edx
    // ```
    //
    // The code at symbol `_ap_boot_marker2` starts with the following instructions:
    //
    // ```
    // 48 bc ef cd ab 90 78 56 34 12    movabs $0x1234567890abcdef, %rsp
    // 48 b8 ff ff 22 22 cc cc 99 99    movabs $0x9999cccc2222ffff, %rax
    // ```
    //
    // The values `0xdeaddead`, `0xff00badd`, `0x1234567890abcdef`, and `0x9999cccc2222ffff` are
    // placeholders that we overwrite in the block below.
    {
        let ap_boot_marker1_loc: *mut u8 = {
            let offset = (_ap_boot_marker1 as usize)
                .checked_sub(_ap_boot_start as usize)
                .unwrap();
            bootstrap_code_buf.as_mut_ptr().add(offset)
        };
        let ap_boot_marker2_loc: *mut u8 = {
            let offset = (_ap_boot_marker2 as usize)
                .checked_sub(_ap_boot_start as usize)
                .unwrap();
            bootstrap_code_buf.as_mut_ptr().add(offset)
        };
        let ap_boot_marker3_loc: *mut u8 = {
            let offset = (_ap_boot_marker3 as usize)
                .checked_sub(_ap_boot_start as usize)
                .unwrap();
            bootstrap_code_buf.as_mut_ptr().add(offset)
        };

        // Perform some sanity check. Since we're doing dark magic, we really want to be sure
        // that we're overwriting the correct code, or we will run into issues that are very hard
        // to debug.
        assert_eq!(
            slice::from_raw_parts(ap_boot_marker1_loc as *const u8, 7),
            &[0x66, 0xea, 0xad, 0xde, 0xad, 0xde, 0x08]
        );
        assert_eq!(
            slice::from_raw_parts(ap_boot_marker2_loc as *const u8, 20),
            &[
                0x48, 0xbc, 0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0x48, 0xb8, 0xff, 0xff,
                0x22, 0x22, 0xcc, 0xcc, 0x99, 0x99
            ]
        );
        assert_eq!(
            slice::from_raw_parts(ap_boot_marker3_loc as *const u8, 6),
            &[0x66, 0xba, 0xdd, 0xba, 0x00, 0xff]
        );

        // Write first constant at marker 2.
        let stack_ptr_ptr = (ap_boot_marker2_loc.add(2)) as *mut u64;
        assert_eq!(stack_ptr_ptr.read_unaligned(), 0x1234567890abcdef);
        stack_ptr_ptr.write_unaligned(stack_top);

        // Write second constant at marker 2.
        let param_ptr = (ap_boot_marker2_loc.add(12)) as *mut u64;
        assert_eq!(param_ptr.read_unaligned(), 0x9999cccc2222ffff);
        param_ptr.write_unaligned(ap_after_boot_param);

        // Write the location of marker 2 into the constant at marker 1.
        let ljmp_target_ptr = (ap_boot_marker1_loc.add(2)) as *mut u32;
        assert_eq!(ljmp_target_ptr.read_unaligned(), 0xdeaddead);
        ljmp_target_ptr.write_unaligned({ u32::try_from(ap_boot_marker2_loc as usize).unwrap() });

        // Write the value of our `cr3` register to the constant at marker 3.
        let pml_addr_ptr = (ap_boot_marker3_loc.add(2)) as *mut u32;
        assert_eq!(pml_addr_ptr.read_unaligned(), 0xff00badd);
        pml_addr_ptr.write_unaligned({
            let pml_addr = x86_64::registers::control::Cr3::read()
                .0
                .start_address()
                .as_u64();
            u32::try_from(pml_addr).unwrap()
        });
    }

    // Wait for 10ms to have elapsed since we sent the INIT IPI.
    executor.block_on(timers.register_tsc_timer(Duration::from_millis(10)));

    // Send the SINIT IPI, pointing to the bootstrap code that we have carefully crafted.
    local_apics.send_interprocessor_sipi(target, bootstrap_code_buf.as_mut_ptr() as *const _);

    // TODO: the APIC doesn't automatically try resubmitting the SIPI in case the target CPU was
    //       busy, so we should send a second SIPI if the first one timed out
    //       (the Intel manual also recommends doing so)
    //       this is however tricky, as we have to make sure we're not sending the second SIPI
    //       if the first one succeeded
    /*let rdtsc = unsafe { core::arch::x86_64::_rdtsc() };
    executor::block_on(local_apics, timers.register_tsc_timer(rdtsc + 1_000_000_000));
    local_apics.send_interprocessor_sipi(target, bootstrap_code_buf.as_mut_ptr() as *const _);*/

    // Wait for CPU initialization to finish.
    executor.block_on(init_finished_future).unwrap();

    // Make sure the buffer is dropped at the end.
    drop(bootstrap_code_buf);
}

// The code here is the template in question. Just like any code, is included in the kernel and
// will be loaded in memory. However, it is not actually meant be executed. Instead it is meant
// to be used as a template.
// Because the associated processor (AP) boot code must be in the first megabyte of memory, we
// first copy this code somewhere in this first megabyte and adjust it.
//
// The `_ap_boot_start` and `_ap_boot_end` symbols encompass the template, so that `ap_boot.rs`
// can copy it. There exist three other symbols `_ap_boot_marker1`, `_ap_boot_marker2` and
// `_ap_boot_marker3` that point to instructions that must be adjusted before execution.
//
// Within this module, we must be careful to not use any absolute address referring to anything
// between `_ap_boot_start` and `_ap_boot_end`, and to not use any relative address referring to
// anything outside of this range, as the addresses will then be wrong when the code gets copied.
global_asm! {r#"
.code16
.align 0x1000
.global _ap_boot_start
.type _ap_boot_start, @function
_ap_boot_start:
    // When we enter here, the CS register is set to the value that we passed through the SIPI,
    // and the IP register is set to `0`.

    movw %cs, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %fs
    movw %ax, %gs
    movw %ax, %ss

    // TODO: properly set up and document the flags here. There's some 1G pages thing that we may not want
    movl $((1 << 10) | (1 << 9) | (1 << 5)), %eax
    movl %eax, %cr4

.global _ap_boot_marker3
_ap_boot_marker3:
    // The `0xff00badd` constant below is replaced with the address of a PML4 table when the
    // template gets adjusted.
    mov $0xff00badd, %edx
    mov %edx, %cr3

    // Enable the EFER.LMA bit, which enables compatibility mode and will make us switch to long
    // mode when we update the CS register.
    mov $0xc0000080, %ecx
    rdmsr
    or $(1 << 8), %eax
    wrmsr

    // Set the appropriate CR0 flags: Paging, Extension Type (math co-processor), and
    // Protected Mode.
    movl $((1 << 31) | (1 << 4) | (1 << 0)), %eax
    movl %eax, %cr0

    // Set up the GDT. Since the absolute address of `_ap_boot_start` is effectively 0 according
    // to the CPU in this 16 bits context, we pass an "absolute" address to `_ap_gdt_ptr` by
    // substracting `_ap_boot_start` from its 32 bits address.
    lgdtl (_ap_gdt_ptr - _ap_boot_start)

.global _ap_boot_marker1
_ap_boot_marker1:
    // A long jump is necessary in order to update the CS registry and properly switch to
    // long mode.
    // The `0xdeaddead` constant below is replaced with the location of `_ap_boot_marker2` when
    // the template gets adjusted.
    ljmpl $8, $0xdeaddead

.code64
.global _ap_boot_marker2
.type _ap_boot_marker2, @function
_ap_boot_marker2:
    // The constants below are replaced with an actual stack location when the template gets
    // adjusted.
    // Set up the stack.
    movq $0x1234567890abcdef, %rsp
    // This is an opaque value for the purpose of this assembly code. It is the parameter that we
    // pass to `ap_after_boot`
    movq $0x9999cccc2222ffff, %rax

    movw $0, %bx
    movw %bx, %ds
    movw %bx, %es
    movw %bx, %fs
    movw %bx, %gs
    movw %bx, %ss

    // In the x86-64 calling convention, the RDI register is used to store the value of the first
    // parameter to pass to a function.
    movq %rax, %rdi

    // We do an indirect call in order to force the assembler to use the absolute address rather
    // than a relative call.
    mov $ap_after_boot, %rdx
    call *%rdx

    cli
    hlt

// Small structure whose location is passed to the CPU in order to load the GDT.
.align 8
_ap_gdt_ptr:
    .short 15
    .long gdt_table

.global _ap_boot_end
.type _ap_boot_end, @function
_ap_boot_end:
    nop
"#}

extern "C" {
    fn _ap_boot_start();
    fn _ap_boot_marker1();
    fn _ap_boot_marker2();
    fn _ap_boot_marker3();
    fn _ap_boot_end();
}

/// Holds an allocation with the given layout.
///
/// There is surprisingly no type in the Rust standard library that keeps track of an allocation.
// TODO: use a `Box` or something once it's possible to pass a custom allocator
struct Allocation<'a, T: alloc::alloc::AllocRef> {
    alloc: &'a mut T,
    inner: ptr::NonNull<u8>,
    layout: Layout,
}

impl<'a, T: alloc::alloc::AllocRef> Allocation<'a, T> {
    fn new(alloc: &'a mut T, layout: Layout) -> Self {
        unsafe {
            let (buf, _) = alloc.alloc(layout).unwrap();
            Allocation {
                alloc,
                inner: buf,
                layout,
            }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_ptr()
    }

    fn size(&self) -> usize {
        self.layout.size()
    }
}

impl<'a, T: alloc::alloc::AllocRef> Drop for Allocation<'a, T> {
    fn drop(&mut self) {
        unsafe { self.alloc.dealloc(self.inner, self.layout) }
    }
}

/// Actual type of the parameter passed to `ap_after_boot`.
type ApAfterBootParam = *mut Box<dyn FnOnce() -> core::convert::Infallible + Send + 'static>;

/// Called by the template code after setup.
///
/// When this function is called, the stack and paging have already been properly set up. The
/// first parameter is gathered from the `rdi` register according to the x86_64 calling
/// convention.
#[no_mangle]
extern "C" fn ap_after_boot(to_exec: usize) -> ! {
    unsafe {
        let to_exec = to_exec as ApAfterBootParam;
        let to_exec = Box::from_raw(to_exec);
        let ret = (*to_exec)();
        match ret {} // TODO: remove this `ret` thingy once `!` is stable
    }
}
