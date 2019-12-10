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

use alloc::vec::Vec;
use parity_scale_codec::{Decode, Encode};

// TODO: this has been randomly generated; instead should be a hash or something
pub const INTERFACE: [u8; 32] = [
    0x24, 0x5d, 0x25, 0x5e, 0x37, 0xf1, 0x8a, 0xce, 0x23, 0xd6, 0x68, 0xe9, 0xe2, 0xd8, 0xd1, 0xbc,
    0x37, 0xf3, 0xd3, 0x3c, 0xad, 0x55, 0xf8, 0xd9, 0x22, 0x3a, 0x57, 0xd1, 0x54, 0x46, 0x7b, 0x78,
];

/// Message in destination to the hardware interface handler.
#[derive(Debug, Encode, Decode)]
pub enum HardwareMessage {
    /// Request to perform some access on the physical memory or ports.
    ///
    /// All operations must be performed in order.
    ///
    /// If there is at least one memory or port read, the response must be a
    /// `Vec<HardwareAccessResponse>` where each element corresponds to a read. No response is
    /// expected if there are only writes.
    HardwareAccess(Vec<Operation>),

    /// Ask the handler to send back a response when the interrupt with the given number is
    /// triggered.
    ///
    /// > **Note**: If called with a non-hardware interrupt, no response will ever come back.
    // TODO: how to not miss any interrupt? we instead need some registration system or something
    InterruptWait(u32),
}

/// Request to perform accesses to physical memory or to ports.
#[derive(Debug, Encode, Decode)]
pub enum Operation {
    PhysicalMemoryWrite {
        address: u64,
        data: Vec<u8>,
    },
    PhysicalMemoryRead {
        address: u64,
        len: u32,
    },
    /// Write data to a port.
    ///
    /// If the hardware doesn't support this operation, then nothing happens.
    PortWriteU8 {
        port: u32,
        data: u8,
    },
    /// Write data to a port.
    ///
    /// If the hardware doesn't support this operation, then nothing happens.
    PortWriteU16 {
        port: u32,
        data: u16,
    },
    /// Write data to a port.
    ///
    /// If the hardware doesn't support this operation, then nothing happens.
    PortWriteU32 {
        port: u32,
        data: u32,
    },
    /// Reads data from a port.
    ///
    /// If the hardware doesn't support this operation, then `0` is produced.
    PortReadU8 {
        port: u32,
    },
    /// Reads data from a port.
    ///
    /// If the hardware doesn't support this operation, then `0` is produced.
    PortReadU16 {
        port: u32,
    },
    /// Reads data from a port.
    ///
    /// If the hardware doesn't support this operation, then `0` is produced.
    PortReadU32 {
        port: u32,
    },
}

#[derive(Debug, Encode, Decode)]
pub enum HardwareAccessResponse {
    PhysicalMemoryRead(Vec<u8>),
    PortReadU8(u8),
    PortReadU16(u16),
    PortReadU32(u32),
}
