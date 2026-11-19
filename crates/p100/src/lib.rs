//! Tesla P100 (GP100, sm_60) device layer: driver plumbing plus the kernels this
//! engine needs, compiled once for `sm_60` and loaded as a cubin at start-up.
//!
//! Design notes that come from measurements on this hardware, not from taste:
//!
//! * sm_60 has no DP4A, so int8 dot products cost four scalar multiplies. GP100
//!   has full-rate `HFMA2` instead, and every weight kernel here dequantises into
//!   subnormal half bit patterns (`q * 2^-24`, exact) and accumulates with
//!   `HFMA2`. That is roughly twice the tokens/s of an int8 path on this chip.
//! * The decode kernels are load-issue bound, not bandwidth bound, so the layout
//!   is chosen so each lane fetches a single 16-byte global load per 32 weight
//!   values, with activation rows staged through shared memory.
//! * There is no NVLink between PCIe P100s, so multiple GPUs are used by
//!   **layer split** with a small hand-off through pinned host memory; a per-layer
//!   all-reduce over PCIe would cost more than the extra parallelism buys.

pub mod driver;
pub mod kernels;

use std::ffi::{c_int, c_void, CStr, CString};
use std::fmt;
use std::slice;

use driver::*;

pub type Error = CuError;

/// A CUDA device with its primary context retained.
pub struct Device {
    pub id: usize,
    pub name: String,
    pub cc: (i32, i32),
    pub sm_count: i32,
    pub total_mem: usize,
    pub clock_khz: i32,
    pub mem_clock_khz: i32,
    pub bus_width: i32,
    pub l2_bytes: i32,
    pub max_threads_per_sm: i32,
    pub max_shared_optin: i32,
    ctx: CUcontext,
    drv: &'static Driver,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Device({} {} cc={}.{} {}MiB)",
            self.id, self.name, self.cc.0, self.cc.1, self.total_mem >> 20
        )
    }
}

impl Device {
    pub fn count() -> Result<usize> {
        let drv = Driver::get()?;
        let mut n = 0;
        drv.check(unsafe { (drv.device_get_count)(&mut n) })?;
        Ok(n as usize)
    }

    pub fn open(id: usize) -> Result<Device> {
        let drv = Driver::get()?;
        let mut count = 0;
        drv.check(unsafe { (drv.device_get_count)(&mut count) })?;
        if id >= count as usize {
            return Err(CuError {
                code: CUDA_ERROR_INVALID_VALUE,
                name: "CUDA_ERROR_INVALID_VALUE".into(),
                message: format!("device {id} does not exist (have {count})"),
            });
        }
        let mut dev: CUdevice = 0;
        drv.check(unsafe { (drv.device_get)(&mut dev, id as c_int) })?;
        let mut name_buf = [0i8; 128];
        drv.check(unsafe { (drv.device_get_name)(name_buf.as_mut_ptr(), 128, dev) })?;
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let attr = |a: c_int| -> Result<i32> {
            let mut v = 0;
            drv.check(unsafe { (drv.device_get_attribute)(&mut v, a, dev) })?;
            Ok(v)
        };
        let mut ctx: CUcontext = std::ptr::null_mut();
        drv.check(unsafe { (drv.primary_ctx_retain)(&mut ctx, dev) })?;
        let mut total_mem = 0usize;
        drv.check(unsafe { (drv.device_total_mem)(&mut total_mem, dev) })?;

        Ok(Device {
            id,
            name,
            cc: (
                attr(ATTR_MAX_COMPUTE_CAPABILITY_MAJOR)?,
                attr(ATTR_MAX_COMPUTE_CAPABILITY_MINOR)?,
            ),
            sm_count: attr(ATTR_MULTIPROCESSOR_COUNT)?,
            total_mem,
            clock_khz: attr(ATTR_CLOCK_RATE)?,
            mem_clock_khz: attr(ATTR_MEMORY_CLOCK_RATE)?,
            bus_width: attr(ATTR_GLOBAL_MEMORY_BUS_WIDTH)?,
            l2_bytes: attr(ATTR_L2_CACHE_SIZE)?,
            max_threads_per_sm: attr(ATTR_MAX_THREADS_PER_MULTIPROCESSOR)?,
            max_shared_optin: attr(ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
            ctx,
            drv,
        })
    }

    #[inline]
    pub fn set_current(&self) {
        let rc = unsafe { (self.drv.ctx_set_current)(self.ctx) };
        debug_assert_eq!(rc, CUDA_SUCCESS, "cuCtxSetCurrent failed");
    }

    pub fn sync(&self) -> Result<()> {
        self.set_current();
        self.drv.check(unsafe { (self.drv.ctx_synchronize)() })
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.set_current();
        let (mut free, mut total) = (0usize, 0usize);
        self.drv
            .check(unsafe { (self.drv.mem_get_info)(&mut free, &mut total) })?;
        Ok((free, total))
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        self.set_current();
        let mut ptr: CUdeviceptr = 0;
        if bytes == 0 {
            return Ok(Buf {
                ptr: 0,
                bytes: 0,
                ctx: self.ctx,
                drv: self.drv,
                id: self.id,
            });
        }
        self.drv
            .check(unsafe { (self.drv.mem_alloc)(&mut ptr, bytes) })?;
        Ok(Buf {
            ptr,
            bytes,
            ctx: self.ctx,
            drv: self.drv,
            id: self.id,
        })
    }

    /// Allocate and zero.
    pub fn alloc_zeroed(&self, bytes: usize) -> Result<Buf> {
        let b = self.alloc(bytes)?;
        if bytes > 0 {
            self.set_current();
            self.drv
                .check(unsafe { (self.drv.memset_d8)(b.ptr, 0, bytes) })?;
        }
        Ok(b)
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<Buf> {
        let b = self.alloc(bytes.len())?;
        if !bytes.is_empty() {
            self.set_current();
            self.drv.check(unsafe {
                (self.drv.memcpy_htod)(b.ptr, bytes.as_ptr() as *const c_void, bytes.len())
            })?;
        }
        Ok(b)
    }

    pub fn stream(&self) -> Result<Stream> {
        self.set_current();
        let mut ptr: CUstream = std::ptr::null_mut();
        self.drv
            .check(unsafe { (self.drv.stream_create)(&mut ptr, 0) })?;
        Ok(Stream {
            ptr,
            ctx: self.ctx,
            drv: self.drv,
        })
    }

    pub fn pinned(&self, bytes: usize) -> Result<PinnedBuf> {
        self.set_current();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        self.drv.check(unsafe {
            (self.drv.mem_alloc_host)(
                &mut ptr,
                bytes,
            )
        })?;
        // cuMemAllocHost has no flags; use cuMemHostAlloc semantics via portable
        // allocation is unnecessary for a single process.
        let _ = CU_MEMHOSTALLOC_PORTABLE;
        Ok(PinnedBuf {
            ptr: ptr as *mut u8,
            bytes,
            drv: self.drv,
        })
    }

    pub fn can_access_peer(&self, peer: &Device) -> bool {
        let mut ok = 0;
        let rc = unsafe { (self.drv.device_can_access_peer)(&mut ok, self.id as CUdevice, peer.id as CUdevice) };
        rc == CUDA_SUCCESS && ok == 1
    }

    pub fn enable_peer(&self, peer: &Device) -> Result<()> {
        self.set_current();
        let rc = unsafe { (self.drv.ctx_enable_peer_access)(peer.ctx, 0) };
        // Already enabled (or unsupported by the topology) is not fatal.
        if rc == CUDA_SUCCESS || rc == 704 {
            Ok(())
        } else {
            Err(self.drv.error(rc))
        }
    }

    pub fn module(&self, cubin: &[u8]) -> Result<Module> {
        self.set_current();
        let mut ptr: CUmodule = std::ptr::null_mut();
        self.drv
            .check(unsafe { (self.drv.module_load_data)(&mut ptr, cubin.as_ptr() as *const c_void) })?;
        Ok(Module {
            ptr,
            ctx: self.ctx,
            drv: self.drv,
        })
    }

    /// Effective HBM bandwidth in GB/s, from the memory clock and bus width. The
    /// measurable ceiling on an ECC P100 is lower than this number; `bench
    /// --mem` reports the real one.
    pub fn theoretical_bw_gbs(&self) -> f64 {
        2.0 * self.mem_clock_khz as f64 * 1e3 * (self.bus_width as f64 / 8.0) / 1e9
    }
}

pub struct Buf {
    pub ptr: CUdeviceptr,
    pub bytes: usize,
    id: usize,
    ctx: CUcontext,
    drv: &'static Driver,
}

unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn device(&self) -> usize {
        self.id
    }

    /// Copy into this buffer synchronously (host must have the context current).
    pub fn upload_from(&self, src: &[u8]) -> Result<()> {
        assert!(src.len() <= self.bytes);
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        self.drv.check(unsafe {
            (self.drv.memcpy_htod)(self.ptr, src.as_ptr() as *const c_void, src.len())
        })
    }

    pub fn download_to(&self, dst: &mut [u8]) -> Result<()> {
        assert!(dst.len() <= self.bytes);
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        self.drv.check(unsafe {
            (self.drv.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len())
        })
    }

    pub fn copy_from(&self, src: &Buf) -> Result<()> {
        assert!(src.bytes <= self.bytes);
        if self.ptr == 0 || src.ptr == 0 {
            return Ok(());
        }
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        if self.id == src.id {
            self.drv
                .check(unsafe { (self.drv.memcpy_dtod)(self.ptr, src.ptr, src.bytes) })
        } else {
            // Cross-device: peer copy where the topology allows it, which cuMemcpy
            // performs as a staged copy otherwise.
            unsafe { (self.drv.ctx_set_current)(src.ctx) };
            self.drv.check(unsafe {
                (self.drv.memcpy_peer_async)(
                    self.ptr,
                    self.ctx,
                    src.ptr,
                    src.ctx,
                    src.bytes,
                    std::ptr::null_mut(),
                )
            })
        }
    }

    pub fn zero(&self) -> Result<()> {
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        self.drv
            .check(unsafe { (self.drv.memset_d8)(self.ptr, 0, self.bytes) })
    }

    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { (self.drv.ctx_set_current)(self.ctx) };
            let _ = unsafe { (self.drv.mem_free)(self.ptr) };
        }
    }
}

pub struct Stream {
    pub ptr: CUstream,
    ctx: CUcontext,
    drv: &'static Driver,
}

impl Stream {
    pub fn sync(&self) -> Result<()> {
        self.drv
            .check(unsafe { (self.drv.stream_synchronize)(self.ptr) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        let _ = unsafe { (self.drv.stream_destroy)(self.ptr) };
    }
}

pub struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
    drv: &'static Driver,
}

unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    /// Interpret the first `n` bytes as `f32` (host copy only).
    pub fn as_f32(&self, n: usize) -> &[f32] {
        assert!(n * 4 <= self.bytes);
        unsafe { slice::from_raw_parts(self.ptr as *const f32, n) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        let _ = unsafe { (self.drv.mem_free_host)(self.ptr as *mut c_void) };
    }
}

pub struct Module {
    ptr: CUmodule,
    ctx: CUcontext,
    drv: &'static Driver,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub fn kernel(&self, name: &str) -> Result<Function> {
        let cname = CString::new(name).unwrap();
        let mut ptr: CUfunction = std::ptr::null_mut();
        let rc = unsafe { (self.drv.module_get_function)(&mut ptr, self.ptr, cname.as_ptr()) };
        if rc != CUDA_SUCCESS {
            return Err(self.drv.error(rc));
        }
        Ok(Function {
            ptr,
            name: name.to_string(),
            ctx: self.ctx,
            drv: self.drv,
        })
    }

    /// Ask for more than 48 KiB of shared memory (needed by the prefill GEMM).
    pub fn set_max_smem(&self, f: &Function, bytes: i32) -> Result<()> {
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        self.drv.check(unsafe {
            (self.drv.func_set_attribute)(
                f.ptr,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        let _ = unsafe { (self.drv.module_unload)(self.ptr) };
    }
}

#[derive(Clone)]
pub struct Function {
    pub ptr: CUfunction,
    pub name: String,
    ctx: CUcontext,
    drv: &'static Driver,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Number of registers per thread, for occupancy accounting (`cudaFuncGetAttributes`).
    pub fn num_regs(&self) -> Option<i32> {
        let mut v = 0;
        let rc = unsafe { (self.drv.func_get_attribute)(&mut v, 4, self.ptr) };
        (rc == CUDA_SUCCESS).then_some(v)
    }

    pub fn max_threads(&self) -> Option<i32> {
        let mut v = 0;
        let rc = unsafe { (self.drv.func_get_attribute)(&mut v, 0, self.ptr) };
        (rc == CUDA_SUCCESS).then_some(v)
    }
}

pub struct LaunchArgs {
    storage: Vec<*mut c_void>,
}

impl LaunchArgs {
    pub fn new() -> Self {
        LaunchArgs { storage: Vec::new() }
    }

    pub fn push<T: Copy>(&mut self, v: &mut T) -> &mut Self {
        self.storage.push(v as *mut T as *mut c_void);
        self
    }
}

impl Default for LaunchArgs {
    fn default() -> Self {
        Self::new()
    }
}

/// Launch a kernel. `args` are pointers to the argument values, in order.
pub fn launch(
    f: &Function,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    stream: Option<&Stream>,
    args: &mut LaunchArgs,
) -> Result<()> {
    unsafe { (f.drv.ctx_set_current)(f.ctx) };
    let rc = unsafe {
        (f.drv.launch_kernel)(
            f.ptr,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            smem,
            stream.map(|s| s.ptr).unwrap_or(std::ptr::null_mut()),
            args.storage.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if rc != CUDA_SUCCESS {
        return Err(f.drv.error(rc));
    }
    Ok(())
}

/// GPU timer pair.
pub struct Event {
    ptr: CUevent,
    ctx: CUcontext,
    drv: &'static Driver,
}

impl Event {
    pub fn new(dev: &Device) -> Result<Event> {
        dev.set_current();
        let mut ptr: CUevent = std::ptr::null_mut();
        dev.drv
            .check(unsafe { (dev.drv.event_create)(&mut ptr, 0) })?;
        Ok(Event {
            ptr,
            ctx: dev.ctx,
            drv: dev.drv,
        })
    }

    pub fn record(&self, s: &Stream) -> Result<()> {
        self.drv
            .check(unsafe { (self.drv.event_record)(self.ptr, s.ptr) })
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        let _ = unsafe { (self.drv.event_destroy)(self.ptr) };
    }
}

pub fn elapsed_ms(a: &Event, b: &Event) -> Result<f32> {
    let mut ms = 0f32;
    b.drv
        .check(unsafe { (b.drv.event_elapsed_time)(&mut ms, a.ptr, b.ptr) })?;
    Ok(ms)
}

/// Environment knobs shared by the CLI and the engine.
pub fn arch_string() -> &'static str {
    env!("P100_KERNEL_ARCH")
}

/// The cubin embedded at build time.
pub fn cubin() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"))
}

impl Buf {
    /// Synchronous partial download starting at `offset` bytes.
    pub fn download_at(&self, offset: usize, dst: &mut [u8]) -> Result<()> {
        assert!(offset + dst.len() <= self.bytes, "download_at out of range");
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        self.drv.check(unsafe {
            (self.drv.memcpy_dtoh)(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr + offset as u64,
                dst.len(),
            )
        })
    }

    /// Synchronous partial upload starting at `offset` bytes.
    pub fn upload_at(&self, offset: usize, src: &[u8]) -> Result<()> {
        assert!(offset + src.len() <= self.bytes, "upload_at out of range");
        unsafe { (self.drv.ctx_set_current)(self.ctx) };
        self.drv.check(unsafe {
            (self.drv.memcpy_htod)(
                self.ptr + offset as u64,
                src.as_ptr() as *const c_void,
                src.len(),
            )
        })
    }

    /// Raw device pointer arithmetic, for kernels that need a window.
    pub fn ptr_at(&self, offset: usize) -> CUdeviceptr {
        assert!(offset <= self.bytes);
        self.ptr + offset as u64
    }
}
