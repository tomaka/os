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

use futures::{channel::mpsc, prelude::*};
use std::{pin::Pin, task::Context, task::Poll};

fn main() {
    std::panic::set_hook(Box::new(|info| {
        redshirt_log_interface::log(format!("Panic: {}\n", info));
    }));
    redshirt_log_interface::init();

    redshirt_syscalls::block_on(async move {
        // TODO: hack to leave time for interface registration
        redshirt_time_interface::Delay::new(std::time::Duration::from_secs(5)).await;

        // Note: IPv6 = 2a00:1450:4007:80f::200e
        let out = redshirt_network_interface::TcpStream::connect(&From::from((
            "fe80::844b:2aff:fea4:513"
                .parse::<std::net::Ipv6Addr>()
                .unwrap(),
            8000,
        )))
        .await
        .unwrap();

        let listener = redshirt_network_interface::TcpListener::bind(&"[::]:8000".parse().unwrap())
            .await
            .unwrap();

        log::info!("Now listening on 0.0.0.0:8000");

        let stream = stream::unfold(listener, |mut l| {
            async move {
                let connec = l.accept().await.0;
                Some((connec, l))
            }
        });

        let mut active_conncs =
            stream::FuturesUnordered::<Pin<Box<dyn Future<Output = ()>>>>::new();
        active_conncs.push(Box::pin(future::pending()));
        let (tx, mut rx) = mpsc::unbounded();

        let http = hyper::server::conn::Http::new().with_executor(Executor { pusher: tx });

        let mut server = hyper::server::Builder::new(
            Accept {
                next_connec: Box::pin(stream),
            },
            http,
        )
        .serve(hyper::service::make_service_fn(|_| {
            async {
                Ok::<_, std::io::Error>(hyper::service::service_fn(|_req| {
                    async {
                        Ok::<_, std::io::Error>(hyper::Response::new(hyper::Body::from(
                            "Hello World",
                        )))
                    }
                }))
            }
        }));

        loop {
            let new_connec =
                match future::select(future::select(&mut server, rx.next()), active_conncs.next())
                    .await
                {
                    future::Either::Left((future::Either::Left((_, _)), _)) => {
                        println!("server finished");
                        break;
                    }
                    future::Either::Left((future::Either::Right((new_connec, _)), _)) => {
                        new_connec.unwrap()
                    }
                    future::Either::Right((_, _)) => continue,
                };

            active_conncs.push(new_connec);
        }
    });
}

struct Accept {
    next_connec: Pin<Box<dyn Stream<Item = redshirt_network_interface::TcpStream>>>,
}

impl hyper::server::accept::Accept for Accept {
    type Conn = redshirt_network_interface::TcpStream;
    type Error = std::io::Error;

    fn poll_accept(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<Self::Conn, Self::Error>>> {
        match Stream::poll_next(Pin::new(&mut self.next_connec), cx) {
            Poll::Ready(Some(c)) => Poll::Ready(Some(Ok(c))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[derive(Clone)]
struct Executor {
    pusher: mpsc::UnboundedSender<Pin<Box<dyn Future<Output = ()>>>>,
}

impl<T: Future<Output = ()> + 'static> hyper::rt::Executor<T> for Executor {
    fn execute(&self, future: T) {
        self.pusher
            .unbounded_send(Box::pin(future))
            .unwrap()
    }
}
