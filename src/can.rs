//! A raw SocketCAN socket: no allocation and no blocking on the send path.

use std::ffi::CString;
use std::io::{Error, Result};
use std::mem::{size_of, zeroed};

pub struct Frame {
    pub id: u32,
    pub len: u8,
    pub data: [u8; 8],
}

pub struct Bus {
    fd: libc::c_int,
}

fn check(result: libc::c_int) -> Result<libc::c_int> {
    if result < 0 {
        Err(Error::last_os_error())
    } else {
        Ok(result)
    }
}

impl Bus {
    pub fn open(channel: &str) -> Result<Bus> {
        let name = CString::new(channel)?;
        unsafe {
            let fd = check(libc::socket(
                libc::PF_CAN,
                libc::SOCK_RAW | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                libc::CAN_RAW,
            ))?;
            let bus = Bus { fd };
            let index = libc::if_nametoindex(name.as_ptr());
            if index == 0 {
                return Err(Error::last_os_error());
            }
            let mut address: libc::sockaddr_can = zeroed();
            address.can_family = libc::AF_CAN as u16;
            address.can_ifindex = index as i32;
            check(libc::bind(
                fd,
                &address as *const _ as *const libc::sockaddr,
                size_of::<libc::sockaddr_can>() as u32,
            ))?;
            Ok(bus)
        }
    }

    /// False when the queue is full: the frame is dropped, not waited for.
    pub fn send(&self, frame: &Frame) -> Result<bool> {
        unsafe {
            let mut raw: libc::can_frame = zeroed();
            raw.can_id = frame.id;
            raw.can_dlc = frame.len;
            raw.data = frame.data;
            let written =
                libc::write(self.fd, &raw as *const _ as *const libc::c_void, size_of::<libc::can_frame>());
            if written < 0 {
                let error = Error::last_os_error();
                return match error.raw_os_error() {
                    Some(libc::EAGAIN) | Some(libc::ENOBUFS) => Ok(false),
                    _ => Err(error),
                };
            }
        }
        Ok(true)
    }

    /// The next frame, waiting until `deadline` on CLOCK_MONOTONIC at most.
    pub fn receive_until(&self, deadline: &libc::timespec) -> Result<Option<Frame>> {
        unsafe {
            loop {
                let mut raw: libc::can_frame = zeroed();
                let read =
                    libc::read(self.fd, &mut raw as *mut _ as *mut libc::c_void, size_of::<libc::can_frame>());
                if read == size_of::<libc::can_frame>() as isize {
                    return Ok(Some(Frame { id: raw.can_id, len: raw.can_dlc, data: raw.data }));
                }
                let error = Error::last_os_error();
                if read >= 0 || error.raw_os_error() != Some(libc::EAGAIN) {
                    return Err(error);
                }
                let now = crate::can::monotonic();
                let left = diff_ns(deadline, &now);
                if left <= 0 {
                    return Ok(None);
                }
                let timeout = from_ns(left);
                let mut poll = libc::pollfd { fd: self.fd, events: libc::POLLIN, revents: 0 };
                let ready = libc::ppoll(&mut poll, 1, &timeout, std::ptr::null());
                if ready < 0 && Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                    return Err(Error::last_os_error());
                }
            }
        }
    }
}

impl Drop for Bus {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

pub fn monotonic() -> libc::timespec {
    unsafe {
        let mut now: libc::timespec = zeroed();
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now);
        now
    }
}

pub fn diff_ns(a: &libc::timespec, b: &libc::timespec) -> i64 {
    (a.tv_sec - b.tv_sec) * 1_000_000_000 + (a.tv_nsec - b.tv_nsec)
}

pub fn from_ns(ns: i64) -> libc::timespec {
    libc::timespec { tv_sec: ns / 1_000_000_000, tv_nsec: ns % 1_000_000_000 }
}

pub fn add_ns(t: &libc::timespec, ns: i64) -> libc::timespec {
    let total = t.tv_sec * 1_000_000_000 + t.tv_nsec + ns;
    from_ns(total)
}
