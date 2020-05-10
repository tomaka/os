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

use crate::{Buffer32, HwAccessRef};

use alloc::alloc::handle_alloc_error;
use core::{alloc::Layout, marker::PhantomData, num::NonZeroU32};

/// A single endpoint descriptor.
///
/// This structure can be seen as a list of transfers that the USB controller must perform with
/// a specific endpoint. The endpoint descriptor has to be put in an appropriate list for any work
/// to be done.
///
/// Since this list might be accessed by the controller, appropriate thread-safety measures have
/// to be taken.
pub struct EndpointDescriptor<TAcc>
where
    for<'r> &'r TAcc: HwAccessRef<'r>,
{
    /// Hardware abstraction layer.
    hardware_access: TAcc,
    /// Physical memory buffer containing the endpoint descriptor.
    buffer: Buffer32<TAcc>,
}

/// Configuration when initialization an [`EndpointDescriptor`].
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum number of bytes that can be sent or received in a single data packet. Only used
    /// when the direction is `OUT` or `SETUP`. Must be inferior or equal to 4095.
    pub maximum_packet_size: u16,
    /// Value between 0 and 128. The USB address of the function containing the endpoint.
    pub function_address: u8,
    /// Value between 0 and 16. The USB address of the endpoint within the function.
    pub endpoint_number: u8,
    /// If true, isochronous TD format. If false, general TD format.
    pub isochronous: bool,
    /// If false, full speed. If true, low speed.
    pub low_speed: bool,
    /// Direction of the data flow.
    pub direction: Direction,
}

#[derive(Debug, Clone)]
pub enum Direction {
    In,
    Out,
    FromTd,
}

impl<TAcc> EndpointDescriptor<TAcc>
where
    TAcc: Clone,
    for<'r> &'r TAcc: HwAccessRef<'r>,
{
    /// Allocates a new endpoint descriptor buffer in physical memory.
    pub async fn new(hardware_access: TAcc, config: Config) -> EndpointDescriptor<TAcc> {
        let buffer = {
            const ENDPOINT_DESCRIPTOR_LAYOUT: Layout =
                unsafe { Layout::from_size_align_unchecked(16, 16) };
            Buffer32::new(hardware_access.clone(), ENDPOINT_DESCRIPTOR_LAYOUT).await
        };

        let header = EndpointControlDecoded {
            maximum_packet_size: config.maximum_packet_size,
            format: config.isochronous,
            skip: true,
            low_speed: config.low_speed,
            direction: config.direction,
            endpoint_number: config.endpoint_number,
            function_address: config.function_address,
        };

        unsafe {
            hardware_access
                .write_memory_u32_be(u64::from(buffer.pointer().get()), &[
                    header.encode(),    // Header
                    0x0,    // Transfer descriptor tail
                    0x0,    // Transfer descriptor head
                    0x0,    // Next endpoint descriptor
                ])
                .await;
        }

        EndpointDescriptor {
            hardware_access,
            buffer,
        }
    }

    /// Returns the physical memory address of the descriptor.
    ///
    /// This value never changes and is valid until the [`EndpointDescriptor`] is destroyed.
    pub fn pointer(&self) -> NonZeroU32 {
        self.buffer.pointer()
    }

    /// Returns the value of the next endpoint descriptor in the linked list.
    ///
    /// If [`EndpointDescriptor::set_next`] or [`EndpointDescriptor::set_next_raw`] was previously
    /// called, returns the corresponding physical memory pointer. If
    /// [`EndpointDescriptor::clear_next`]
    pub async fn get_next_raw(&self) -> u32 {
        unsafe {
            let mut out = [0];
            self.hardware_access
                .read_memory_u32_be(u64::from(self.buffer.pointer().get() + 12), &mut out)
                .await;
            out[0]
        }
    }

    /// Sets the next endpoint descriptor in the linked list.
    ///
    /// Endpoint descriptors are always part of a linked list, where each descriptor points to the
    /// next one, or to nothing.
    ///
    /// # Safety
    ///
    /// `next` must remain valid until the next time [`EndpointDescriptor::clear_next`],
    /// [`EndpointDescriptor::set_next`] or [`EndpointDescriptor::set_next_raw`] is called, or
    /// until this [`EndpointDescriptor`] is destroyed.
    pub async unsafe fn set_next<UAcc>(&mut self, next: &EndpointDescriptor<UAcc>)
    where
        UAcc: Clone,
        for<'r> &'r UAcc: HwAccessRef<'r>,
    {
        self.set_next_raw(next.pointer().get()).await;
    }

    /// Sets the next endpoint descriptor in the linked list.
    ///
    /// If 0 is passed, has the same effect as [`EndpointDescriptor::clear_next`].
    ///
    /// # Safety
    ///
    /// If not 0, `next` must be the physical memory address of an endpoint descriptor. It must
    /// remain valid until the next time [`EndpointDescriptor::clear_next`],
    /// [`EndpointDescriptor::set_next`] or [`EndpointDescriptor::set_next_raw`] is called, or
    /// until this [`EndpointDescriptor`] is destroyed.
    pub async unsafe fn set_next_raw(&mut self, next: u32) {
        self.hardware_access
            .write_memory_u32_be(u64::from(self.buffer.pointer().get() + 12), &[next])
            .await;
    }

    /// Sets the next endpoint descriptor in the linked list to nothing.
    pub async fn clear_next(&mut self) {
        unsafe {
            self.set_next_raw(0).await;
        }
    }
}

#[derive(Debug)]
struct EndpointControlDecoded {
    /// Maximum number of bytes that can be sent or received in a single data packet. Only used
    /// when the direction is `OUT` or `SETUP`. Must be inferior or equal to 4095.
    maximum_packet_size: u16,
    /// If true, isochronous TD format. If false, general TD format.
    format: bool,
    /// When set, the HC continues on the next ED off the list without accessing this one.
    skip: bool,
    /// If false, full speed. If true, low speed.
    low_speed: bool,
    /// Direction of the data flow.
    direction: Direction,
    /// Value between 0 and 16. The USB address of the endpoint within the function.
    endpoint_number: u8,
    /// Value between 0 and 128. The USB address of the function containing the endpoint.
    function_address: u8,
}

impl EndpointControlDecoded {
    fn encode(&self) -> u32 {
        assert!(self.maximum_packet_size < (1 << 12));
        assert!(self.endpoint_number < (1 << 5));
        assert!(self.function_address < (1 << 7));

        let direction = match self.direction {
            Direction::In => 0b10,
            Direction::Out => 0b01,
            Direction::FromTd => 0b00,
        };

        u32::from(self.maximum_packet_size) << 16
            | if self.format { 1 } else { 0 } << 15
            | if self.skip { 1 } else { 0 } << 14
            | if self.low_speed { 1 } else { 0 } << 13
            | direction << 11
            | u32::from(self.endpoint_number) << 7
            | u32::from(self.function_address)
    }
}
