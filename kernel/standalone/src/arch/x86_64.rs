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

#![cfg(target_arch = "x86_64")]

use alloc::sync::Arc;
use core::{convert::TryFrom as _, future::Future, ops::Range};
use x86_64::registers::model_specific::Msr;
use x86_64::structures::port::{PortRead as _, PortWrite as _};

mod acpi;
mod apic;
mod boot_link;
mod interrupts;
mod panic;

/// Called by `boot.S` after basic set up has been performed.
///
/// When this function is called, a stack has been set up and as much memory space as possible has
/// been identity-mapped (i.e. the virtual memory is equal to the physical memory).
///
/// Since the kernel was loaded by a multiboot2 bootloader, the first parameter is the memory
/// address of the multiboot header.
#[no_mangle]
extern "C" fn after_boot(multiboot_header: usize) -> ! {
    unsafe {
        let multiboot_info = multiboot2::load(multiboot_header);

        crate::mem_alloc::initialize(find_free_memory_ranges(&multiboot_info));

        // TODO: panics in BOCHS
        //let acpi = acpi::load_acpi_tables(&multiboot_info);

        unsafe {
            APIC = Some(apic::init());
        }
        interrupts::init();

        let kernel = crate::kernel::Kernel::init(crate::kernel::KernelConfig {
            num_cpus: 1,
            ..Default::default()
        });

        kernel.run()
    }
}

// TODO: safisize
static mut APIC: Option<Arc<apic::ApicControl>> = None;

// TODO: define the semantics of that
pub fn halt() -> ! {
    loop {
        x86_64::instructions::hlt()
    }
}

/// Returns the number of nanoseconds that happened since an undeterminate moment in time.
pub fn monotonic_clock() -> u128 {
    // TODO: wrong unit; these are not nanoseconds
    // TODO: maybe TSC not supported? move method to ApicControl instead?
    u128::from(unsafe { core::arch::x86_64::_rdtsc() })
}

/// Returns a `Future` that fires when the monotonic clock reaches the given value.
pub fn timer(clock_value: u128) -> impl Future<Output = ()> {
    let clock_value = u64::try_from(clock_value).unwrap_or(u64::max_value());
    unsafe { APIC.as_ref().unwrap().register_tsc_timer(clock_value) }
}

/// Reads the boot information and find the memory ranges that can be used as a heap.
///
/// # Panic
///
/// Panics if the information is wrong or if there isn't enough information available.
///
fn find_free_memory_ranges<'a>(
    multiboot_info: &'a multiboot2::BootInformation,
) -> impl Iterator<Item = Range<usize>> + 'a {
    let mem_map = multiboot_info.memory_map_tag().unwrap();
    let elf_sections = multiboot_info.elf_sections_tag().unwrap();

    mem_map.memory_areas().filter_map(move |area| {
        let mut area_start = area.start_address();
        let mut area_end = area.end_address();
        debug_assert!(area_start <= area_end);

        // The kernel has probably been loaded into RAM, so we have to remove ELF sections
        // from the portion of memory that we use.
        for section in elf_sections.sections() {
            if section.start_address() >= area_start && section.end_address() <= area_end {
                /*         ↓ section_start    section_end ↓
                ==================================================
                    ↑ area_start                      area_end ↑
                */
                let off_bef = section.start_address() - area_start;
                let off_aft = area_end - section.end_address();
                if off_bef > off_aft {
                    area_end = section.start_address();
                } else {
                    area_start = section.end_address();
                }
            } else if section.start_address() < area_start && section.end_address() > area_end {
                /*    ↓ section_start             section_end ↓
                ==================================================
                        ↑ area_start         area_end ↑
                */
                // We have no memory available!
                return None;
            } else if section.start_address() <= area_start && section.end_address() > area_start {
                /*    ↓ section_start     section_end ↓
                ==================================================
                        ↑ area_start                 area_end ↑
                */
                area_start = section.end_address();
            } else if section.start_address() < area_end && section.end_address() >= area_end {
                /*         ↓ section_start      section_end ↓
                ==================================================
                    ↑ area_start         area_end ↑
                */
                area_end = section.start_address();
            }
        }

        let area_start = usize::try_from(area_start).unwrap();
        let area_end = usize::try_from(area_end).unwrap();
        Some(area_start..area_end)
    })
}

pub unsafe fn write_port_u8(port: u32, data: u8) {
    if let Ok(port) = u16::try_from(port) {
        u8::write_to_port(port, data);
    }
}

pub unsafe fn write_port_u16(port: u32, data: u16) {
    if let Ok(port) = u16::try_from(port) {
        u16::write_to_port(port, data);
    }
}

pub unsafe fn write_port_u32(port: u32, data: u32) {
    if let Ok(port) = u16::try_from(port) {
        u32::write_to_port(port, data);
    }
}

pub unsafe fn read_port_u8(port: u32) -> u8 {
    if let Ok(port) = u16::try_from(port) {
        u8::read_from_port(port)
    } else {
        0
    }
}

pub unsafe fn read_port_u16(port: u32) -> u16 {
    if let Ok(port) = u16::try_from(port) {
        u16::read_from_port(port)
    } else {
        0
    }
}

pub unsafe fn read_port_u32(port: u32) -> u32 {
    if let Ok(port) = u16::try_from(port) {
        u32::read_from_port(port)
    } else {
        0
    }
}
