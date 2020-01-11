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

//! Main kernel module.
//!
//! # Usage
//!
//! - Create a [`KernelConfig`] struct indicating the configuration.
//! - From one CPU, create a [`Kernel`] with [`Kernel::init`].
//! - Share the newly-created [`Kernel`] between CPUs, and call [`Kernel::run`] once for each CPU.
//!

use core::sync::atomic::{AtomicBool, Ordering};

/// Main struct of this crate. Runs everything.
pub struct Kernel {
    /// If true, the kernel has started running from a different thread already.
    running: AtomicBool,
}

/// Configuration for creating a [`Kernel`].
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct KernelConfig {
    /// Number of times the [`Kernel::run`] function might be called.
    pub num_cpus: u32,
}

impl Kernel {
    /// Initializes a new `Kernel`.
    pub fn init(_cfg: KernelConfig) -> Self {
        Kernel {
            running: AtomicBool::new(false),
        }
    }

    /// Run the kernel. Must be called once per CPU.
    pub fn run(&self) -> ! {
        // We only want a single CPU to run for now.
        if self.running.swap(true, Ordering::SeqCst) {
            crate::arch::halt();
        }

        let hello_module = redshirt_core::module::Module::from_bytes(
            &include_bytes!(
                "../../../modules/target/wasm32-unknown-unknown/release/hello-world.wasm"
            )[..],
        )
        .unwrap();

        // TODO: use a better system than cfgs
        #[cfg(target_arch = "x86_64")]
        let log_module = redshirt_core::module::Module::from_bytes(
            &include_bytes!("../../../modules/target/wasm32-unknown-unknown/release/x86-log.wasm")
                [..],
        )
        .unwrap();
        #[cfg(target_arch = "arm")]
        let log_module = redshirt_core::module::Module::from_bytes(
            &include_bytes!("../../../modules/target/wasm32-unknown-unknown/release/arm-log.wasm")
                [..],
        )
        .unwrap();

        // TODO: use a better system than cfgs
        #[cfg(target_arch = "x86_64")]
        let pci_module = redshirt_core::module::Module::from_bytes(
            &include_bytes!("../../../modules/target/wasm32-unknown-unknown/release/x86-pci.wasm")
                [..],
        )
        .unwrap();

        // TODO: use a better system than cfgs
        #[cfg(target_arch = "x86_64")]
        let ne2000_module = redshirt_core::module::Module::from_bytes(
            &include_bytes!("../../../modules/target/wasm32-unknown-unknown/debug/ne2000.wasm")[..],
        )
        .unwrap();

        let mut system_builder = redshirt_core::system::SystemBuilder::new()
            .with_native_program(crate::hardware::HardwareHandler::new())
            .with_native_program(crate::random::native::RandomNativeProgram::new())
            .with_startup_process(log_module)
            .with_startup_process(hello_module);

        // TODO: use a better system than cfgs
        #[cfg(target_arch = "x86_64")]
        {
            system_builder = system_builder
                .with_startup_process(pci_module)
                .with_startup_process(ne2000_module)
        }

        let mut system = system_builder
            .with_main_program([0; 32]) // TODO: just a test
            .build();

        loop {
            // TODO: ideally the entire function would be async, and this would be an `await`,
            // but async functions don't work on no_std yet
            match crate::executor::block_on(system.run()) {
                redshirt_core::system::SystemRunOutcome::ProgramFinished { pid, outcome } => {
                    //console.write(&format!("Program finished {:?} => {:?}\n", pid, outcome));
                }
                _ => panic!(),
            }
        }
    }
}
