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

//! TCP/IP.
//!
//! Allows opening asynchronous TCP sockets and listeners, similar to what the `tokio` or
//! `async-std` libraries do.

use futures::{lock::Mutex, prelude::*, ready};
use redshirt_syscalls::{Encode as _, MessageResponseFuture};
use std::{
    cmp, io, mem,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    pin::Pin,
    task::{Context, Poll},
};

pub mod ffi;

/// Active TCP connection to a remote.
///
/// This type is similar to [`std::net::TcpStream`].
pub struct TcpStream {
    handle: u32,
    /// Buffer of data that has been read from the socket but not transmitted to the user yet.
    read_buffer: Vec<u8>,
    /// If Some, we have sent out a "read" message and are waiting for a response.
    pending_read: Option<MessageResponseFuture<ffi::TcpReadResponse>>,
    /// If Some, we have sent out a "write" message and are waiting for a response.
    pending_write: Option<MessageResponseFuture<ffi::TcpWriteResponse>>,
}

/// Active TCP listening socket.
///
/// This type is similar to [`std::net::TcpListener`].
pub struct TcpListener {
    local_addr: SocketAddr,
    next_incoming: Mutex<
        stream::FuturesUnordered<
            Pin<Box<dyn Future<Output = Result<(TcpStream, SocketAddr), ()>> + Send>>,
        >,
    >,
}

impl TcpStream {
    /// Start connecting to the given address. Returns a `TcpStream` if the connection is
    /// successful.
    pub fn connect(socket_addr: &SocketAddr) -> impl Future<Output = Result<TcpStream, ()>> {
        let fut = TcpStream::new(socket_addr, false);
        async move { Ok(fut.await?.0) }
    }

    /// Dialing and listening use the same underlying messages. The only different being a boolean
    /// indicating whether the address is a binding point or a destination.
    fn new(
        socket_addr: &SocketAddr,
        listen: bool,
    ) -> impl Future<Output = Result<(TcpStream, SocketAddr), ()>> {
        let tcp_open = ffi::TcpMessage::Open(match socket_addr {
            SocketAddr::V4(addr) => ffi::TcpOpen {
                ip: addr.ip().to_ipv6_mapped().segments(),
                port: addr.port(),
                listen,
            },
            SocketAddr::V6(addr) => ffi::TcpOpen {
                ip: addr.ip().segments(),
                port: addr.port(),
                listen,
            },
        });

        async move {
            let message: ffi::TcpOpenResponse = unsafe {
                let msg = tcp_open.encode();
                redshirt_syscalls::MessageBuilder::new()
                    .add_data(&msg)
                    .emit_with_response(&ffi::INTERFACE)
                    .unwrap()
                    .await
            };

            let socket_open_info = message.result?;
            let remote_addr = {
                let ip = Ipv6Addr::from(socket_open_info.remote_ip);
                SocketAddr::new(IpAddr::from(ip), socket_open_info.remote_port)
            };

            let stream = TcpStream {
                handle: socket_open_info.socket_id,
                read_buffer: Vec::new(),
                pending_read: None,
                pending_write: None,
            };

            Ok((stream, remote_addr))
        }
    }
}

impl AsyncRead for TcpStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &mut [u8],
    ) -> Poll<Result<usize, io::Error>> {
        loop {
            if let Some(pending_read) = self.pending_read.as_mut() {
                self.read_buffer = match ready!(Future::poll(Pin::new(pending_read), cx)).result {
                    Ok(d) => d,
                    Err(_) => return Poll::Ready(Err(io::ErrorKind::Other.into())), // TODO:
                };
                self.pending_read = None;
            }

            debug_assert!(self.pending_read.is_none());

            if !self.read_buffer.is_empty() {
                let to_copy = cmp::min(self.read_buffer.len(), buf.len());
                let mut tmp = mem::replace(&mut self.read_buffer, Vec::new());
                self.read_buffer = tmp.split_off(to_copy);
                buf[..to_copy].copy_from_slice(&tmp);
                return Poll::Ready(Ok(to_copy));
            }

            self.pending_read = {
                let tcp_read = ffi::TcpMessage::Read(ffi::TcpRead {
                    socket_id: self.handle,
                });

                let msg_id = unsafe {
                    let msg = tcp_read.encode();
                    redshirt_syscalls::MessageBuilder::new()
                        .add_data(&msg)
                        .emit_with_response_raw(&ffi::INTERFACE)
                        .unwrap()
                };

                Some(redshirt_syscalls::message_response(msg_id))
            };
        }
    }

    // TODO: implement poll_read_vectored
    // TODO: unsafe fn initializer(&self) -> Initializer { ... }
}

impl AsyncWrite for TcpStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        // Try to finish the previous write, if any is in progress.
        if let Some(pending_write) = self.pending_write.as_mut() {
            match ready!(Future::poll(Pin::new(pending_write), cx)).result {
                Ok(()) => self.pending_write = None,
                Err(_) => return Poll::Ready(Err(io::ErrorKind::Other.into())), // TODO:
            }
        }

        debug_assert!(self.pending_write.is_none());

        // Perform the write, and store into `self.pending_write` a future to when we can start
        // the next write.
        self.pending_write = {
            let tcp_write = ffi::TcpMessage::Write(ffi::TcpWrite {
                socket_id: self.handle,
                data: buf.to_vec(), // TODO: meh for cloning
            });

            let msg_id = unsafe {
                let msg = tcp_write.encode(); // TODO: meh because we clone data a second time here
                redshirt_syscalls::MessageBuilder::new()
                    .add_data(&msg)
                    .emit_with_response_raw(&ffi::INTERFACE)
                    .unwrap()
            };

            Some(redshirt_syscalls::message_response(msg_id))
        };

        Poll::Ready(Ok(buf.len()))
    }

    // TODO: implement poll_write_vectored

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _: &mut Context) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }
}

impl tokio::io::AsyncRead for TcpStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &mut [u8],
    ) -> Poll<Result<usize, io::Error>> {
        AsyncRead::poll_read(self, cx, buf)
    }

    unsafe fn prepare_uninitialized_buffer(&self, _: &mut [mem::MaybeUninit<u8>]) -> bool {
        false
    }
}

impl tokio::io::AsyncWrite for TcpStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        AsyncWrite::poll_write(self, cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Result<(), io::Error>> {
        AsyncWrite::poll_flush(self, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Result<(), io::Error>> {
        AsyncWrite::poll_close(self, cx)
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        unsafe {
            let tcp_close = ffi::TcpMessage::Close(ffi::TcpClose {
                socket_id: self.handle,
            });

            let _ = redshirt_syscalls::emit_message_without_response(&ffi::INTERFACE, &tcp_close);
        }
    }
}

impl TcpListener {
    /// Create a new [`TcpListener`] listening on the given address and port.
    pub fn bind(socket_addr: &SocketAddr) -> impl Future<Output = Result<TcpListener, ()>> {
        let next_incoming = Mutex::new(
            (0..10)
                .map(|_| Box::pin(TcpStream::new(socket_addr, true)) as Pin<Box<_>>)
                .collect(),
        );

        let socket_addr = socket_addr.clone();
        async move {
            Ok(TcpListener {
                local_addr: socket_addr,
                next_incoming,
            })
        }
    }

    /// Returns the local address of the listener. Useful to determine the port.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Waits for a new incoming connection and returns it.
    pub async fn accept(&self) -> (TcpStream, SocketAddr) {
        let mut next_incoming = self.next_incoming.lock().await;

        let (tcp_stream, remote_addr) = loop {
            match next_incoming.next().await {
                Some(Ok(v)) => break v,
                Some(Err(_)) => continue,
                None => unreachable!(),
            }
        };

        next_incoming.push(Box::pin(TcpStream::new(&self.local_addr, true)));
        (tcp_stream, remote_addr)
    }
}
