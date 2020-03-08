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

//! x86_64 interrupts handling.
//!
//! This module provides handling of interrupts on x86_64. It sets up the interrupts table (IDT)
//! and allows registers a [`Waker`](core::task::Waker) that is waken up when an interrupt happens.
//! This is done by calling [`set_interrupt_waker`].
//!
//! Note that this API is racy. Once a `Waker` has been woken up, it gets discarded and needs to
//! be registered again. It is possible that an interrupt gets triggered between the discard and
//! the re-registration.
//!
//! This is not considered to be a problem, as hardware normally lets you know why an interrupt
//! has happened and/or requires you to notify the hardware when you processed an interrupt before
//! the next one can be issued. By re-registering a `Waker` before looking for the interrupt
//! reason, there is no risk of losing information.
//!

// TODO: init() has to be called; this isn't great
// TODO: while it's not a problem that the API is racy, it is a problem if multiple different
//       pieces of code inadvertently try to share an interrupt vector
// TODO: handle end-of-interrupt of the APIC

use core::task::Waker;
use futures::task::AtomicWaker;
use x86_64::structures::idt;

/// Registers a `Waker` to wake up when an interrupt happens.
///
/// For each value of `interrupt`, only the latest registered `Waker` will be waken up.
///
/// # Panic
///
/// You can only set a waker for interrupts 32 to 255. Any value inferior to 32 will
/// trigger a panic.
///
pub fn set_interrupt_waker(interrupt: u8, waker: &Waker) {
    assert!(interrupt >= 32);
    WAKERS[usize::from(interrupt - 32)].register(waker);
}

/// Initializes the interrupts system.
///
/// Before this is called, the waker passed to [`set_interrupt_waker`] will never work.
pub unsafe fn init() {
    IDT.load();
    x86_64::instructions::interrupts::enable();
}

lazy_static::lazy_static! {
    /// Table read by the hardware in order to determine what to do when an interrupt happens.
    static ref IDT: idt::InterruptDescriptorTable = {
        let mut idt = idt::InterruptDescriptorTable::new();
        macro_rules! set_entry {
            ($idt:ident[$n:expr]) => {{
                set_entry!($idt[$n], $n);
            }};
            ($entry:expr, $n:expr) => {{
                extern "x86-interrupt" fn handler(_: &mut idt::InterruptStackFrame) {
                    WAKERS[$n - 32].wake();
                }
                $entry.set_handler_fn(handler)
                    .disable_interrupts(false);
            }};
        }

        idt[0].set_handler_fn(int0).disable_interrupts(false);
        idt[1].set_handler_fn(int1).disable_interrupts(false);
        idt[2].set_handler_fn(int2).disable_interrupts(false);
        idt[3].set_handler_fn(int3).disable_interrupts(false);
        idt[4].set_handler_fn(int4).disable_interrupts(false);
        idt[5].set_handler_fn(int5).disable_interrupts(false);
        idt[6].set_handler_fn(int6).disable_interrupts(false);
        idt[7].set_handler_fn(int7).disable_interrupts(false);
        idt.double_fault.set_handler_fn(int8).disable_interrupts(false);
        idt[9].set_handler_fn(int9).disable_interrupts(false);
        idt.invalid_tss.set_handler_fn(int10).disable_interrupts(false);
        idt.segment_not_present.set_handler_fn(int11).disable_interrupts(false);
        idt.stack_segment_fault.set_handler_fn(int12).disable_interrupts(false);
        idt.general_protection_fault.set_handler_fn(int13).disable_interrupts(false);
        idt.page_fault.set_handler_fn(int14).disable_interrupts(false);
        // 15 is reserved
        idt[16].set_handler_fn(int16).disable_interrupts(false);
        idt.alignment_check.set_handler_fn(int17).disable_interrupts(false);
        idt.machine_check.set_handler_fn(int18).disable_interrupts(false);
        idt[19].set_handler_fn(int19).disable_interrupts(false);
        idt[20].set_handler_fn(int20).disable_interrupts(false);
        // 21 is reserved
        // 22 is reserved
        // 23 is reserved
        // 24 is reserved
        // 25 is reserved
        // 26 is reserved
        // 27 is reserved
        // 28 is reserved
        // 29 is reserved
        idt.security_exception.set_handler_fn(int30).disable_interrupts(false);
        // 31 is reserved
        set_entry!(idt[32]);
        set_entry!(idt[33]);
        set_entry!(idt[34]);
        set_entry!(idt[35]);
        set_entry!(idt[36]);
        set_entry!(idt[37]);
        set_entry!(idt[38]);
        set_entry!(idt[39]);
        set_entry!(idt[40]);
        set_entry!(idt[41]);
        set_entry!(idt[42]);
        set_entry!(idt[43]);
        set_entry!(idt[44]);
        set_entry!(idt[45]);
        set_entry!(idt[46]);
        set_entry!(idt[47]);
        set_entry!(idt[48]);
        set_entry!(idt[49]);
        set_entry!(idt[50]);
        set_entry!(idt[51]);
        set_entry!(idt[52]);
        set_entry!(idt[53]);
        set_entry!(idt[54]);
        set_entry!(idt[55]);
        set_entry!(idt[56]);
        set_entry!(idt[57]);
        set_entry!(idt[58]);
        set_entry!(idt[59]);
        set_entry!(idt[60]);
        set_entry!(idt[61]);
        set_entry!(idt[62]);
        set_entry!(idt[63]);
        set_entry!(idt[64]);
        set_entry!(idt[65]);
        set_entry!(idt[66]);
        set_entry!(idt[67]);
        set_entry!(idt[68]);
        set_entry!(idt[69]);
        set_entry!(idt[70]);
        set_entry!(idt[71]);
        set_entry!(idt[72]);
        set_entry!(idt[73]);
        set_entry!(idt[74]);
        set_entry!(idt[75]);
        set_entry!(idt[76]);
        set_entry!(idt[77]);
        set_entry!(idt[78]);
        set_entry!(idt[79]);
        set_entry!(idt[80]);
        set_entry!(idt[81]);
        set_entry!(idt[82]);
        set_entry!(idt[83]);
        set_entry!(idt[84]);
        set_entry!(idt[85]);
        set_entry!(idt[86]);
        set_entry!(idt[87]);
        set_entry!(idt[88]);
        set_entry!(idt[89]);
        set_entry!(idt[90]);
        set_entry!(idt[91]);
        set_entry!(idt[92]);
        set_entry!(idt[93]);
        set_entry!(idt[94]);
        set_entry!(idt[95]);
        set_entry!(idt[96]);
        set_entry!(idt[97]);
        set_entry!(idt[98]);
        set_entry!(idt[99]);
        set_entry!(idt[100]);
        set_entry!(idt[101]);
        set_entry!(idt[102]);
        set_entry!(idt[103]);
        set_entry!(idt[104]);
        set_entry!(idt[105]);
        set_entry!(idt[106]);
        set_entry!(idt[107]);
        set_entry!(idt[108]);
        set_entry!(idt[109]);
        set_entry!(idt[110]);
        set_entry!(idt[111]);
        set_entry!(idt[112]);
        set_entry!(idt[113]);
        set_entry!(idt[114]);
        set_entry!(idt[115]);
        set_entry!(idt[116]);
        set_entry!(idt[117]);
        set_entry!(idt[118]);
        set_entry!(idt[119]);
        set_entry!(idt[120]);
        set_entry!(idt[121]);
        set_entry!(idt[122]);
        set_entry!(idt[123]);
        set_entry!(idt[124]);
        set_entry!(idt[125]);
        set_entry!(idt[126]);
        set_entry!(idt[127]);
        set_entry!(idt[128]);
        set_entry!(idt[129]);
        set_entry!(idt[130]);
        set_entry!(idt[131]);
        set_entry!(idt[132]);
        set_entry!(idt[133]);
        set_entry!(idt[134]);
        set_entry!(idt[135]);
        set_entry!(idt[136]);
        set_entry!(idt[137]);
        set_entry!(idt[138]);
        set_entry!(idt[139]);
        set_entry!(idt[140]);
        set_entry!(idt[141]);
        set_entry!(idt[142]);
        set_entry!(idt[143]);
        set_entry!(idt[144]);
        set_entry!(idt[145]);
        set_entry!(idt[146]);
        set_entry!(idt[147]);
        set_entry!(idt[148]);
        set_entry!(idt[149]);
        set_entry!(idt[150]);
        set_entry!(idt[151]);
        set_entry!(idt[152]);
        set_entry!(idt[153]);
        set_entry!(idt[154]);
        set_entry!(idt[155]);
        set_entry!(idt[156]);
        set_entry!(idt[157]);
        set_entry!(idt[158]);
        set_entry!(idt[159]);
        set_entry!(idt[160]);
        set_entry!(idt[161]);
        set_entry!(idt[162]);
        set_entry!(idt[163]);
        set_entry!(idt[164]);
        set_entry!(idt[165]);
        set_entry!(idt[166]);
        set_entry!(idt[167]);
        set_entry!(idt[168]);
        set_entry!(idt[169]);
        set_entry!(idt[170]);
        set_entry!(idt[171]);
        set_entry!(idt[172]);
        set_entry!(idt[173]);
        set_entry!(idt[174]);
        set_entry!(idt[175]);
        set_entry!(idt[176]);
        set_entry!(idt[177]);
        set_entry!(idt[178]);
        set_entry!(idt[179]);
        set_entry!(idt[180]);
        set_entry!(idt[181]);
        set_entry!(idt[182]);
        set_entry!(idt[183]);
        set_entry!(idt[184]);
        set_entry!(idt[185]);
        set_entry!(idt[186]);
        set_entry!(idt[187]);
        set_entry!(idt[188]);
        set_entry!(idt[189]);
        set_entry!(idt[190]);
        set_entry!(idt[191]);
        set_entry!(idt[192]);
        set_entry!(idt[193]);
        set_entry!(idt[194]);
        set_entry!(idt[195]);
        set_entry!(idt[196]);
        set_entry!(idt[197]);
        set_entry!(idt[198]);
        set_entry!(idt[199]);
        set_entry!(idt[200]);
        set_entry!(idt[201]);
        set_entry!(idt[202]);
        set_entry!(idt[203]);
        set_entry!(idt[204]);
        set_entry!(idt[205]);
        set_entry!(idt[206]);
        set_entry!(idt[207]);
        set_entry!(idt[208]);
        set_entry!(idt[209]);
        set_entry!(idt[210]);
        set_entry!(idt[211]);
        set_entry!(idt[212]);
        set_entry!(idt[213]);
        set_entry!(idt[214]);
        set_entry!(idt[215]);
        set_entry!(idt[216]);
        set_entry!(idt[217]);
        set_entry!(idt[218]);
        set_entry!(idt[219]);
        set_entry!(idt[220]);
        set_entry!(idt[221]);
        set_entry!(idt[222]);
        set_entry!(idt[223]);
        set_entry!(idt[224]);
        set_entry!(idt[225]);
        set_entry!(idt[226]);
        set_entry!(idt[227]);
        set_entry!(idt[228]);
        set_entry!(idt[229]);
        set_entry!(idt[230]);
        set_entry!(idt[231]);
        set_entry!(idt[232]);
        set_entry!(idt[233]);
        set_entry!(idt[234]);
        set_entry!(idt[235]);
        set_entry!(idt[236]);
        set_entry!(idt[237]);
        set_entry!(idt[238]);
        set_entry!(idt[239]);
        set_entry!(idt[240]);
        set_entry!(idt[241]);
        set_entry!(idt[242]);
        set_entry!(idt[243]);
        set_entry!(idt[244]);
        set_entry!(idt[245]);
        set_entry!(idt[246]);
        set_entry!(idt[247]);
        set_entry!(idt[248]);
        set_entry!(idt[249]);
        set_entry!(idt[250]);
        set_entry!(idt[251]);
        set_entry!(idt[252]);
        set_entry!(idt[253]);
        set_entry!(idt[254]);
        set_entry!(idt[255]);

        idt
    };
}

// TODO: properly document all interrupts

macro_rules! interrupt_panic {
    ($msg:expr, $frame:expr) => {
        panic!(
            "Exception: {} at 0x{:x}",
            $msg,
            $frame.instruction_pointer.as_u64()
        )
    }
}

extern "x86-interrupt" fn int0(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Division by zero", frame);
}

extern "x86-interrupt" fn int1(frame: &mut idt::InterruptStackFrame) {
    let dr0: u64;
    let dr1: u64;
    let dr2: u64;
    let dr3: u64;
    let dr6: u64;
    let dr7: u64;

    unsafe {
        asm!("mov %dr0, $0" : "=r"(dr0));
        asm!("mov %dr1, $0" : "=r"(dr1));
        asm!("mov %dr2, $0" : "=r"(dr2));
        asm!("mov %dr3, $0" : "=r"(dr3));
        asm!("mov %dr6, $0" : "=r"(dr6));
        asm!("mov %dr7, $0" : "=r"(dr7));
    }

    panic!(r#"Debug interrupt
DR0 = 0x{:016x} ; DR1 = 0x{:016x}
DR2 = 0x{:016x} ; DR3 = 0x{:016x}
DR6 = 0x{:016x}
DR7 = 0x{:016x}
"#, dr0, dr1, dr2, dr3, dr6, dr7)
}

extern "x86-interrupt" fn int2(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("NMI", frame); // TODO: there might be additional trickery here
}

extern "x86-interrupt" fn int3(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Breakpoint", frame);
}

extern "x86-interrupt" fn int4(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Overflow", frame);
}

extern "x86-interrupt" fn int5(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Bounds", frame);
}

extern "x86-interrupt" fn int6(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Invalid opcode", frame);
}

extern "x86-interrupt" fn int7(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Coprocessor not available", frame);
}

extern "x86-interrupt" fn int8(frame: &mut idt::InterruptStackFrame, _: u64) -> ! {
    // A double fault happens when an exception happens while an exception was already
    // being handled.
    //
    // We don't panic, as it's likely that it's the panic handler that trigger this.
    x86_64::instructions::interrupts::disable();
    x86_64::instructions::bochs_breakpoint();
    loop {
        x86_64::instructions::hlt();
    }
}

extern "x86-interrupt" fn int9(frame: &mut idt::InterruptStackFrame) {
    // Since the 486, this exception is instead a GPF.
    interrupt_panic!("Coprocessor segment overrun", frame);
}

extern "x86-interrupt" fn int10(frame: &mut idt::InterruptStackFrame, _: u64) {
    interrupt_panic!("Invalid TSS", frame);
}

extern "x86-interrupt" fn int11(frame: &mut idt::InterruptStackFrame, _: u64) {
    interrupt_panic!("Segment not present", frame);
}

extern "x86-interrupt" fn int12(frame: &mut idt::InterruptStackFrame, _: u64) {
    interrupt_panic!("Stack segment fault", frame);
}

extern "x86-interrupt" fn int13(frame: &mut idt::InterruptStackFrame, _: u64) {
    interrupt_panic!("General protection fault", frame);
}

extern "x86-interrupt" fn int14(frame: &mut idt::InterruptStackFrame, _: idt::PageFaultErrorCode) {
    interrupt_panic!("Page fault", frame);
}

extern "x86-interrupt" fn int16(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("x87 exception", frame);
}

extern "x86-interrupt" fn int17(frame: &mut idt::InterruptStackFrame, _: u64) {
    interrupt_panic!("Alignment error", frame);
}

extern "x86-interrupt" fn int18(frame: &mut idt::InterruptStackFrame) -> ! {
    interrupt_panic!("Machine check", frame);
}

extern "x86-interrupt" fn int19(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("SIMD error", frame);
}

extern "x86-interrupt" fn int20(frame: &mut idt::InterruptStackFrame) {
    interrupt_panic!("Virtualization exception", frame);
}

extern "x86-interrupt" fn int30(frame: &mut idt::InterruptStackFrame, _: u64) {
    interrupt_panic!("Security exception", frame);
}

/// For each interrupt vector, a [`Waker`](core::task::Waker) that must be waken up when that
/// interrupt happens.
static WAKERS: [AtomicWaker; 256 - 32] = [
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
    AtomicWaker::new(),
];
