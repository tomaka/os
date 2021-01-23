// Copyright (C) 2019-2021  Pierre Krieger
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

//! Futures executor that works on bare metal.

use alloc::sync::Arc;
use core::future::Future;
use core::sync::atomic;
use core::task::{Context, Poll};
use futures::task::{waker, ArcWake};

/// Waits for the `Future` to resolve to a value.
///
/// This function is similar to [`futures::executor::block_on`].
pub fn block_on<R>(future: impl Future<Output = R>) -> R {
    futures::pin_mut!(future);

    let local_wake = Arc::new(LocalWake {
        woken_up: atomic::AtomicBool::new(false),
    });

    let waker = waker(local_wake.clone());
    let mut context = Context::from_waker(&waker);

    loop {
        if let Poll::Ready(val) = Future::poll(future.as_mut(), &mut context) {
            return val;
        }

        // Loop until `woken_up` is true.
        loop {
            if local_wake
                .woken_up
                .compare_exchange(
                    true,
                    false,
                    atomic::Ordering::Acquire,
                    atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                break;
            }

            // Enter a low-power state and wait for an event to happen.
            // TODO: can an interrupt happen between `local_wake` and here?
            // contrary to other platforms, the manual doesn't mention anything about enabling
            // instructions having a delay
            unsafe { asm!("wfi", options(nomem, nostack, preserves_flags)) }
        }
    }
}

struct LocalWake {
    woken_up: atomic::AtomicBool,
}

impl ArcWake for LocalWake {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.woken_up.store(true, atomic::Ordering::Release);
        // TODO: there doesn't seem to be a standard way to wake up another processor
        // For now we are single-CPUed.
    }
}
