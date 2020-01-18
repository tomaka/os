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

//! TCP/IP.

#![deny(intra_doc_link_resolution_failure)]

// TODO: everything here is a draft

use futures::{prelude::*, ready};
use parity_scale_codec::DecodeAll;
use redshirt_syscalls::{Encode as _, MessageId};
use std::{
    cmp, io, mem, net::Ipv6Addr, net::SocketAddr, pin::Pin, sync::Arc, task::Context, task::Poll,
    task::Waker,
};

pub mod ffi;

pub struct TcpStream {
    handle: u32,
    /// Buffer of data that has been read from the socket but not transmitted to the user yet.
    read_buffer: Vec<u8>,
    /// If Some, we have sent out a "read" message and are waiting for a response.
    // TODO: use strongly typed Future here
    pending_read: Option<Pin<Box<dyn Future<Output = ffi::TcpReadResponse> + Send>>>,
    /// If Some, we have sent out a "write" message and are waiting for a response.
    // TODO: use strongly typed Future here
    pending_write: Option<Pin<Box<dyn Future<Output = ffi::TcpWriteResponse> + Send>>>,
}

impl TcpStream {
    pub fn connect(socket_addr: &SocketAddr) -> impl Future<Output = Result<TcpStream, ()>> {
        let tcp_open = ffi::TcpMessage::Open(match socket_addr {
            SocketAddr::V4(addr) => ffi::TcpOpen {
                ip: addr.ip().to_ipv6_mapped().segments(),
                port: addr.port(),
            },
            SocketAddr::V6(addr) => ffi::TcpOpen {
                ip: addr.ip().segments(),
                port: addr.port(),
            },
        });

        let msg_id = unsafe {
            let msg = tcp_open.encode();
            redshirt_syscalls::MessageBuilder::new()
                .add_data(&msg)
                .emit_with_response_raw(&ffi::INTERFACE)
                .unwrap()
        };

        async move {
            let message: ffi::TcpOpenResponse = redshirt_syscalls::message_response(msg_id).await;
            let handle = message.result?;

            Ok(TcpStream {
                handle,
                read_buffer: Vec::new(),
                pending_read: None,
                pending_write: None,
            })
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

            if !self.read_buffer.is_empty() {
                let to_copy = cmp::min(self.read_buffer.len(), buf.len());
                let mut tmp = mem::replace(&mut self.read_buffer, Vec::new());
                self.read_buffer = tmp.split_off(to_copy);
                buf[..to_copy].copy_from_slice(&tmp);
                return Poll::Ready(Ok(to_copy));
            }

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
            self.pending_read = Some(Box::pin(redshirt_syscalls::message_response(msg_id)));
        }
    }

    // TODO: unsafe fn initializer(&self) -> Initializer { ... }
}

impl AsyncWrite for TcpStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        if let Some(pending_write) = self.pending_write.as_mut() {
            match ready!(Future::poll(Pin::new(pending_write), cx)).result {
                Ok(()) => self.pending_write = None,
                Err(_) => return Poll::Ready(Err(io::ErrorKind::Other.into())), // TODO:
            }
        }

        let tcp_write = ffi::TcpMessage::Write(ffi::TcpWrite {
            socket_id: self.handle,
            data: buf.to_vec(),
        });
        let msg_id = unsafe {
            let msg = tcp_write.encode();
            redshirt_syscalls::MessageBuilder::new()
                .add_data(&msg)
                .emit_with_response_raw(&ffi::INTERFACE)
                .unwrap()
        };
        self.pending_write = Some(Box::pin(redshirt_syscalls::message_response(msg_id)));
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _: &mut Context) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }
}

impl tokio_io::AsyncRead for TcpStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &mut [u8],
    ) -> Poll<Result<usize, io::Error>> {
        AsyncRead::poll_read(self, cx, buf)
    }
}

impl tokio_io::AsyncWrite for TcpStream {
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

            redshirt_syscalls::emit_message_without_response(&ffi::INTERFACE, &tcp_close);
        }
    }
}

pub struct TcpListener {
    handle: u32,
    local_addr: SocketAddr,
    /// If Some, we have sent out an "accept" message and are waiting for a response.
    // TODO: use strongly typed Future here
    pending_accept: Option<Pin<Box<dyn Future<Output = ffi::TcpAcceptResponse> + Send>>>,
}

impl TcpListener {
    pub fn bind(socket_addr: &SocketAddr) -> impl Future<Output = Result<TcpListener, ()>> {
        let tcp_listen = ffi::TcpMessage::Listen(match socket_addr {
            SocketAddr::V4(addr) => ffi::TcpListen {
                local_ip: addr.ip().to_ipv6_mapped().segments(),
                port: addr.port(),
            },
            SocketAddr::V6(addr) => ffi::TcpListen {
                local_ip: addr.ip().segments(),
                port: addr.port(),
            },
        });

        let msg_id = unsafe {
            let msg = tcp_listen.encode();
            redshirt_syscalls::MessageBuilder::new()
                .add_data(&msg)
                .emit_with_response_raw(&ffi::INTERFACE)
                .unwrap()
        };

        let mut local_addr = socket_addr.clone();

        async move {
            let message: ffi::TcpListenResponse = redshirt_syscalls::message_response(msg_id).await;
            let (handle, local_port) = message.result?;
            local_addr.set_port(local_port);

            Ok(TcpListener {
                handle,
                local_addr,
                pending_accept: None,
            })
        }
    }

    /// Returns the local address of the listener. Useful to determine the port.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    // TODO: make `&self` instead
    pub async fn accept(&mut self) -> (TcpStream, SocketAddr) {
        loop {
            if let Some(pending_accept) = self.pending_accept.as_mut() {
                let new_stream = pending_accept.await;
                self.pending_accept = None;
                let stream = TcpStream {
                    handle: new_stream.accepted_socket_id,
                    read_buffer: Vec::new(),
                    pending_read: None,
                    pending_write: None,
                };
                let remote_ip = Ipv6Addr::from(new_stream.remote_ip);
                let remote_addr = SocketAddr::from((remote_ip, new_stream.remote_port));
                return (stream, remote_addr);
            }

            let tcp_accept = ffi::TcpMessage::Accept(ffi::TcpAccept {
                socket_id: self.handle,
            });
            let msg_id = unsafe {
                let msg = tcp_accept.encode();
                redshirt_syscalls::MessageBuilder::new()
                    .add_data(&msg)
                    .emit_with_response_raw(&ffi::INTERFACE)
                    .unwrap()
            };
            self.pending_accept = Some(Box::pin(redshirt_syscalls::message_response(msg_id)));
        }
    }
}

impl Drop for TcpListener {
    fn drop(&mut self) {
        unsafe {
            let tcp_close = ffi::TcpMessage::Close(ffi::TcpClose {
                socket_id: self.handle,
            });

            redshirt_syscalls::emit_message_without_response(&ffi::INTERFACE, &tcp_close);
        }
    }
}
