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

//! Manages a collection of network interfaces.
//!
//! This module manages the state of all the network interfaces together. Most of the
//! implementation is delegated to the [`interface`] module, and the primary role of this code
//! is to aggregate interfaces and assign new sockets to the correct interface based on the
//! available routes.

use fnv::FnvBuildHasher;
use futures::prelude::*;
use hashbrown::{hash_map::Entry, HashMap};
use std::{
    fmt, hash::Hash, iter, marker::PhantomData, net::SocketAddr, pin::Pin, sync::MutexGuard,
};

mod interface;
mod port_assign;

/// State machine managing all the network interfaces and sockets.
///
/// The `TIfId` generic parameter is an identifier for network interfaces.
// TODO: Debug
pub struct NetworkManager<TIfId, TIfUser> {
    /// List of devices that have been registered.
    devices: HashMap<TIfId, Device<TIfUser>, FnvBuildHasher>,
    /// Id to assign to the next socket.
    next_socket_id: u64,
    /// List of sockets open in the manager.
    sockets: HashMap<u64, SocketState<TIfId>, FnvBuildHasher>,
}

/// State of a socket.
#[derive(Debug)]
enum SocketState<TIfId> {
    /// Socket is waiting to be assigned to an interface.
    Pending {
        /// `listen` parameter passed to the socket constructor.
        listen: bool,
        /// Socket address parameter passed to the socket constructor.
        addr: SocketAddr,
    },
    /// Socket has been assigned to a specific interface.
    Assigned {
        /// Interface it's been assigned to.
        interface: TIfId,
        /// Id of the socket within the interface.
        inner_id: interface::SocketId,
    },
}

/// State of a device.
struct Device<TIfUser> {
    /// Inner state.
    inner: interface::NetInterfaceState<u64>,
    /// Additional user data.
    user_data: TIfUser,
}

/// Event generated by the [`NetworkManagerEvent::next_event`] function.
#[derive(Debug)]
pub enum NetworkManagerEvent<'a, TIfId, TIfUser> {
    /// Data to be sent out by the Ethernet cable is available.
    ///
    /// Contains a mutable reference of the data buffer. Data can be left in the buffer if
    /// desired.
    EthernetCableOut(TIfId, &'a mut TIfUser, Vec<u8>),
    /// A TCP/IP socket has connected to its target.
    TcpConnected(TcpSocket<'a, TIfId>),
    /// A TCP/IP socket has been closed by the remote.
    TcpClosed(TcpSocket<'a, TIfId>),
    /// A TCP/IP socket has data ready to be read.
    TcpReadReady(TcpSocket<'a, TIfId>),
    /// A TCP/IP socket has finished writing the data that we passed to it, and is now ready to
    /// accept more.
    TcpWriteFinished(TcpSocket<'a, TIfId>),
}

/// Internal enum similar to [`NetworkManagerEvent`], except that it is `'static`.
///
/// Necessary because of borrow checker issue.
// TODO: remove this once Polonius lands in Rust
#[derive(Debug)]
enum NetworkManagerEventStatic {
    EthernetCableOut,
    TcpConnected(interface::SocketId),
    TcpClosed(interface::SocketId),
    TcpReadReady(interface::SocketId),
    TcpWriteFinished(interface::SocketId),
    DhcpDiscovery,
}

/// Access to a socket within the manager.
pub struct TcpSocket<'a, TIfId> {
    id: u64,
    inner: TcpSocketInner<'a, TIfId>,
}

enum TcpSocketInner<'a, TIfId> {
    Pending,
    Assigned {
        inner: interface::TcpSocket<'a, u64>,
        device_id: TIfId,
    },
}

/// Identifier of a socket within the [`NetworkManager`]. Common between all types of sockets.
#[derive(Debug, Copy, Clone, PartialEq, Eq)] // TODO: Hash
pub struct SocketId<TIfId> {
    id: u64,
    marker: PhantomData<TIfId>,
}

impl<TIfId, TIfUser> NetworkManager<TIfId, TIfUser>
where
    TIfId: Clone + Hash + PartialEq + Eq,
{
    /// Initializes a new `NetworkManager`.
    pub fn new() -> Self {
        NetworkManager {
            devices: HashMap::default(),
            next_socket_id: 1,
            sockets: HashMap::default(),
        }
    }

    /// Adds a new TCP socket to the state of the network manager.
    ///
    /// If `listen` is `true`, then `addr` is a local address that the socket will listen on.
    pub fn build_tcp_socket(&mut self, listen: bool, addr: &SocketAddr) -> TcpSocket<TIfId> {
        let socket_id = self.next_socket_id;
        self.next_socket_id += 1;

        for (device_id, device) in self.devices.iter_mut() {
            if let Ok(socket) = device.inner.build_tcp_socket(listen, addr, socket_id) {
                self.sockets.insert(
                    socket_id,
                    SocketState::Assigned {
                        interface: device_id.clone(),
                        inner_id: socket.id(),
                    },
                );

                return TcpSocket {
                    id: socket_id,
                    inner: TcpSocketInner::Assigned {
                        inner: socket,
                        device_id: device_id.clone(),
                    },
                };
            }
        }

        self.sockets.insert(
            socket_id,
            SocketState::Pending {
                listen,
                addr: addr.clone(),
            },
        );

        TcpSocket {
            id: socket_id,
            inner: TcpSocketInner::Pending,
        }
    }

    ///
    pub fn tcp_socket_by_id(&mut self, id: &SocketId<TIfId>) -> Option<TcpSocket<TIfId>> {
        match self.sockets.get_mut(&id.id)? {
            SocketState::Pending { listen, addr } => Some(TcpSocket {
                id: id.id,
                inner: TcpSocketInner::Pending,
            }),
            SocketState::Assigned {
                interface,
                inner_id,
            } => {
                let int_ref = &mut self.devices.get_mut(&interface)?.inner;
                let inner = int_ref.tcp_socket_by_id(*inner_id)?;
                Some(TcpSocket {
                    id: id.id,
                    inner: TcpSocketInner::Assigned {
                        device_id: interface.clone(),
                        inner,
                    },
                })
            }
        }
    }

    /// Registers an interface with the given ID. Returns an error if an interface with that ID
    /// already exists.
    pub async fn register_interface(
        &mut self,
        id: TIfId,
        mac_address: [u8; 6],
        user_data: TIfUser,
    ) -> Result<(), ()> {
        let entry = match self.devices.entry(id.clone()) {
            Entry::Occupied(_) => return Err(()),
            Entry::Vacant(e) => e,
        };

        log::debug!("Registering interface with MAC {:?}", mac_address);

        let interface = interface::NetInterfaceState::new(interface::Config {
            ip_address: interface::ConfigIpAddr::DHCPv4,
            mac_address,
        })
        .await;

        entry.insert(Device {
            inner: interface,
            user_data,
        });

        Ok(())
    }

    // TODO: better API?
    pub fn unregister_interface(&mut self, id: &TIfId) {
        let device = self.devices.remove(id);
        // TODO:
    }

    /// Extract the data to transmit out of the Ethernet cable.
    ///
    /// Returns an empty buffer if nothing is ready.
    // TODO: better API?
    pub fn interface_user_data(&mut self, id: &TIfId) -> &mut TIfUser {
        &mut self
            .devices
            .get_mut(id)
            .unwrap() // TODO: don't unwrap
            .user_data
    }

    /// Extract the data to transmit out of the Ethernet cable.
    ///
    /// Returns an empty buffer if nothing is ready.
    // TODO: better API?
    pub fn read_ethernet_cable_out(&mut self, id: &TIfId) -> Vec<u8> {
        self.devices
            .get_mut(id)
            .unwrap() // TODO: don't unwrap
            .inner
            .read_ethernet_cable_out()
    }

    /// Injects some data coming from the Ethernet cable.
    // TODO: better API?
    pub fn inject_interface_data(&mut self, id: &TIfId, data: impl AsRef<[u8]>) {
        self.devices
            .get_mut(id)
            .unwrap() // TODO: don't unwrap
            .inner
            .inject_interface_data(data)
    }

    /// Returns the next event generated by the [`NetworkManager`].
    pub async fn next_event<'a>(&'a mut self) -> NetworkManagerEvent<'a, TIfId, TIfUser> {
        loop {
            let (device_id, event) = loop {
                break match self.next_event_inner().await {
                    Some(ev) => ev,
                    None => continue,
                };
            };

            match event {
                NetworkManagerEventStatic::EthernetCableOut => {
                    let device = self.devices.get_mut(&device_id).unwrap();
                    let data = device.inner.read_ethernet_cable_out();
                    debug_assert!(!data.is_empty());
                    return NetworkManagerEvent::EthernetCableOut(
                        device_id,
                        &mut device.user_data,
                        data,
                    );
                }
                NetworkManagerEventStatic::TcpConnected(socket) => {
                    let device = self.devices.get_mut(&device_id).unwrap();
                    let inner = device.inner.tcp_socket_by_id(socket).unwrap();
                    return NetworkManagerEvent::TcpConnected(TcpSocket {
                        id: *inner.user_data(),
                        inner: TcpSocketInner::Assigned { inner, device_id },
                    });
                }
                NetworkManagerEventStatic::TcpClosed(socket) => {
                    let device = self.devices.get_mut(&device_id).unwrap();
                    let inner = device.inner.tcp_socket_by_id(socket).unwrap();
                    return NetworkManagerEvent::TcpClosed(TcpSocket {
                        id: *inner.user_data(),
                        inner: TcpSocketInner::Assigned { inner, device_id },
                    });
                }
                NetworkManagerEventStatic::TcpReadReady(socket) => {
                    let device = self.devices.get_mut(&device_id).unwrap();
                    let inner = device.inner.tcp_socket_by_id(socket).unwrap();
                    return NetworkManagerEvent::TcpReadReady(TcpSocket {
                        id: *inner.user_data(),
                        inner: TcpSocketInner::Assigned { inner, device_id },
                    });
                }
                NetworkManagerEventStatic::TcpWriteFinished(socket) => {
                    let device = self.devices.get_mut(&device_id).unwrap();
                    let inner = device.inner.tcp_socket_by_id(socket).unwrap();
                    return NetworkManagerEvent::TcpWriteFinished(TcpSocket {
                        id: *inner.user_data(),
                        inner: TcpSocketInner::Assigned { inner, device_id },
                    });
                }
                NetworkManagerEventStatic::DhcpDiscovery => {
                    let interface = self.devices.get_mut(&device_id).unwrap();

                    // Take all the pending sockets and try to assign them to that new interface.
                    // TODO: that's O(n)
                    for (socket_id, socket) in self.sockets.iter_mut() {
                        let (listen, addr) = match &socket {
                            SocketState::Pending { listen, addr } => (*listen, addr),
                            SocketState::Assigned { .. } => continue,
                        };

                        // TODO: naive
                        if let Ok(inner_socket) =
                            interface.inner.build_tcp_socket(listen, addr, *socket_id)
                        {
                            log::debug!(
                                "Assigned TCP socket ({}) to newly-registered interface",
                                addr
                            );
                            *socket = SocketState::Assigned {
                                interface: device_id.clone(),
                                inner_id: inner_socket.id(),
                            };
                        }
                    }
                }
            }
        }
    }

    // TODO: don't return an Option
    async fn next_event_inner<'a>(&'a mut self) -> Option<(TIfId, NetworkManagerEventStatic)> {
        // TODO: optimize?
        let next_event = future::select_all(
            self.devices
                .iter_mut()
                .map(move |(n, d)| {
                    let user_data = &mut d.user_data;
                    Box::pin(
                        d.inner
                            .next_event()
                            .map(move |ev| (n.clone(), user_data, ev)),
                    ) as Pin<Box<dyn Future<Output = _>>>
                })
                .chain(iter::once(Box::pin(future::pending()) as Pin<Box<_>>)),
        );

        match next_event.await.0 {
            (device_id, _, interface::NetInterfaceEvent::EthernetCableOut) => {
                Some((device_id, NetworkManagerEventStatic::EthernetCableOut))
            }
            (device_id, _, interface::NetInterfaceEvent::TcpConnected(inner)) => Some((
                device_id,
                NetworkManagerEventStatic::TcpConnected(inner.id()),
            )),
            (device_id, _, interface::NetInterfaceEvent::TcpClosed(inner)) => {
                Some((device_id, NetworkManagerEventStatic::TcpClosed(inner.id())))
            }
            (device_id, _, interface::NetInterfaceEvent::TcpReadReady(inner)) => Some((
                device_id,
                NetworkManagerEventStatic::TcpReadReady(inner.id()),
            )),
            (device_id, _, interface::NetInterfaceEvent::TcpWriteFinished(inner)) => Some((
                device_id,
                NetworkManagerEventStatic::TcpWriteFinished(inner.id()),
            )),
            (device_id, _, interface::NetInterfaceEvent::DhcpDiscovery { .. }) => {
                Some((device_id, NetworkManagerEventStatic::DhcpDiscovery))
            }
        }
    }
}

impl<'a, TIfId: Clone> TcpSocket<'a, TIfId> {
    /// Returns the identifier of the socket, for later retrieval.
    pub fn id(&self) -> SocketId<TIfId> {
        SocketId {
            id: self.id,
            marker: PhantomData,
        }
    }

    /// Closes the socket.
    pub fn close(self) {
        //self.device.
    }
}

impl<'a, TIfId> fmt::Debug for TcpSocket<'a, TIfId>
where
    TIfId: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: better impl
        f.debug_tuple("TcpSocket").finish()
    }
}
