//! Minimal, hand-written bindings to the CUDA *driver* API.
//!
//! The driver API is loaded with `dlopen("libcuda.so.1")` at runtime, so building
//! this crate does not require the CUDA development package, and a machine with
//! several driver versions keeps using the one that owns the device nodes.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::fmt;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_LAUNCH_FAILED: CUresult = 719;

// Device attributes used by the engine.
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub const ATTR_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const ATTR_MAX_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const ATTR_MAX_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const ATTR_CLOCK_RATE: c_int = 13;
pub const ATTR_MEMORY_CLOCK_RATE: c_int = 36;
pub const ATTR_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const ATTR_L2_CACHE_SIZE: c_int = 38;

// Context creation / kernel function attributes.
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

// cuMemHostAlloc flags.
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 4;

#[derive(Clone, PartialEq, Eq)]
pub struct CuError {
    pub code: CUresult,
    pub name: String,
    pub message: String,
}

impl fmt::Debug for CuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) {}", self.name, self.code, self.message)
    }
}

impl fmt::Display for CuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}): {}", self.name, self.code, self.message)
    }
}

impl std::error::Error for CuError {}

pub type Result<T, E = CuError> = std::result::Result<T, E>;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let cname = concat!($name, "\0");
        let p = unsafe { dlsym($lib, cname.as_ptr() as *const c_char) };
        if p.is_null() {
            return Err(CuError {
                code: -1,
                name: "dlsym".into(),
                message: format!("symbol {} not found in libcuda.so.1", $name),
            });
        }
        unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
    }};
}

type FnInit = unsafe extern "C" fn(u32) -> CUresult;
type FnErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type FnErrorName = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type FnDriverVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type FnDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type FnDeviceTotalMem = unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult;
type FnPrimaryCtxRetain = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type FnPrimaryCtxRelease = unsafe extern "C" fn(CUdevice) -> CUresult;
type FnCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCtxGetCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
type FnCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type FnMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type FnMemAllocHost = unsafe extern "C" fn(*mut *mut c_void, usize) -> CUresult;
type FnMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
type FnMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnMemcpyDtoD = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
type FnMemcpyHtoDAsync = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type FnMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type FnMemcpyDtoDAsync = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
type FnMemcpyPeerAsync =
    unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize, CUstream) -> CUresult;
type FnMemsetD32 = unsafe extern "C" fn(CUdeviceptr, u32, usize) -> CUresult;
type FnMemsetD8 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
type FnModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type FnModuleGetFunction = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type FnFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
type FnFuncGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUfunction) -> CUresult;
type FnLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type FnStreamCreate = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
type FnStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
type FnStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
type FnEventCreate = unsafe extern "C" fn(*mut CUevent, u32) -> CUresult;
type FnEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type FnEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
type FnEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type FnEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
type FnDeviceCanAccessPeer = unsafe extern "C" fn(*mut c_int, CUdevice, CUdevice) -> CUresult;
type FnCtxEnablePeerAccess = unsafe extern "C" fn(CUcontext, u32) -> CUresult;

pub struct Driver {
    #[allow(dead_code)]
    lib: *mut c_void,
    pub version: c_int,
    pub init: FnInit,
    pub get_error_string: FnErrorString,
    pub get_error_name: FnErrorName,
    pub driver_get_version: FnDriverVersion,
    pub device_get_count: FnDeviceGetCount,
    pub device_get: FnDeviceGet,
    pub device_get_name: FnDeviceGetName,
    pub device_get_attribute: FnDeviceGetAttribute,
    pub device_total_mem: FnDeviceTotalMem,
    pub primary_ctx_retain: FnPrimaryCtxRetain,
    pub primary_ctx_release: FnPrimaryCtxRelease,
    pub ctx_set_current: FnCtxSetCurrent,
    pub ctx_get_current: FnCtxGetCurrent,
    pub ctx_synchronize: FnCtxSynchronize,
    pub mem_alloc: FnMemAlloc,
    pub mem_free: FnMemFree,
    pub mem_alloc_host: FnMemAllocHost,
    pub mem_free_host: FnMemFreeHost,
    pub mem_get_info: FnMemGetInfo,
    pub memcpy_htod: FnMemcpyHtoD,
    pub memcpy_dtoh: FnMemcpyDtoH,
    pub memcpy_dtod: FnMemcpyDtoD,
    pub memcpy_htod_async: FnMemcpyHtoDAsync,
    pub memcpy_dtoh_async: FnMemcpyDtoHAsync,
    pub memcpy_dtod_async: FnMemcpyDtoDAsync,
    pub memcpy_peer_async: FnMemcpyPeerAsync,
    pub memset_d32: FnMemsetD32,
    pub memset_d8: FnMemsetD8,
    pub module_load_data: FnModuleLoadData,
    pub module_unload: FnModuleUnload,
    pub module_get_function: FnModuleGetFunction,
    pub func_set_attribute: FnFuncSetAttribute,
    pub func_get_attribute: FnFuncGetAttribute,
    pub launch_kernel: FnLaunchKernel,
    pub stream_create: FnStreamCreate,
    pub stream_destroy: FnStreamDestroy,
    pub stream_synchronize: FnStreamSynchronize,
    pub event_create: FnEventCreate,
    pub event_record: FnEventRecord,
    pub event_synchronize: FnEventSynchronize,
    pub event_elapsed_time: FnEventElapsedTime,
    pub event_destroy: FnEventDestroy,
    pub device_can_access_peer: FnDeviceCanAccessPeer,
    pub ctx_enable_peer_access: FnCtxEnablePeerAccess,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: std::sync::LazyLock<std::result::Result<Driver, CuError>> =
    std::sync::LazyLock::new(Driver::load);

impl Driver {
    /// Load libcuda.so.1 and resolve every entry point.
    pub fn get() -> Result<&'static Driver> {
        match &*DRIVER {
            Ok(d) => Ok(d),
            Err(e) => Err(e.clone()),
        }
    }

    fn load() -> std::result::Result<Driver, CuError> {
        let lib = unsafe { dlopen(b"libcuda.so.1\0".as_ptr() as *const c_char, RTLD_NOW) };
        if lib.is_null() {
            let msg = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "libcuda.so.1 not found (is the NVIDIA driver installed?)".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            return Err(CuError {
                code: -1,
                name: "dlopen".into(),
                message: msg,
            });
        }

        let mut d = Driver {
            lib,
            version: 0,
            init: sym!(lib, "cuInit", FnInit),
            get_error_string: sym!(lib, "cuGetErrorString", FnErrorString),
            get_error_name: sym!(lib, "cuGetErrorName", FnErrorName),
            driver_get_version: sym!(lib, "cuDriverGetVersion", FnDriverVersion),
            device_get_count: sym!(lib, "cuDeviceGetCount", FnDeviceGetCount),
            device_get: sym!(lib, "cuDeviceGet", FnDeviceGet),
            device_get_name: sym!(lib, "cuDeviceGetName", FnDeviceGetName),
            device_get_attribute: sym!(lib, "cuDeviceGetAttribute", FnDeviceGetAttribute),
            device_total_mem: sym!(lib, "cuDeviceTotalMem_v2", FnDeviceTotalMem),
            primary_ctx_retain: sym!(lib, "cuDevicePrimaryCtxRetain", FnPrimaryCtxRetain),
            primary_ctx_release: sym!(lib, "cuDevicePrimaryCtxRelease_v2", FnPrimaryCtxRelease),
            ctx_set_current: sym!(lib, "cuCtxSetCurrent", FnCtxSetCurrent),
            ctx_get_current: sym!(lib, "cuCtxGetCurrent", FnCtxGetCurrent),
            ctx_synchronize: sym!(lib, "cuCtxSynchronize", FnCtxSynchronize),
            mem_alloc: sym!(lib, "cuMemAlloc_v2", FnMemAlloc),
            mem_free: sym!(lib, "cuMemFree_v2", FnMemFree),
            mem_alloc_host: sym!(lib, "cuMemAllocHost_v2", FnMemAllocHost),
            mem_free_host: sym!(lib, "cuMemFreeHost", FnMemFreeHost),
            mem_get_info: sym!(lib, "cuMemGetInfo_v2", FnMemGetInfo),
            memcpy_htod: sym!(lib, "cuMemcpyHtoD_v2", FnMemcpyHtoD),
            memcpy_dtoh: sym!(lib, "cuMemcpyDtoH_v2", FnMemcpyDtoH),
            memcpy_dtod: sym!(lib, "cuMemcpyDtoD_v2", FnMemcpyDtoD),
            memcpy_htod_async: sym!(lib, "cuMemcpyHtoDAsync_v2", FnMemcpyHtoDAsync),
            memcpy_dtoh_async: sym!(lib, "cuMemcpyDtoHAsync_v2", FnMemcpyDtoHAsync),
            memcpy_dtod_async: sym!(lib, "cuMemcpyDtoDAsync_v2", FnMemcpyDtoDAsync),
            memcpy_peer_async: sym!(lib, "cuMemcpyPeerAsync", FnMemcpyPeerAsync),
            memset_d32: sym!(lib, "cuMemsetD32_v2", FnMemsetD32),
            memset_d8: sym!(lib, "cuMemsetD8_v2", FnMemsetD8),
            module_load_data: sym!(lib, "cuModuleLoadData", FnModuleLoadData),
            module_unload: sym!(lib, "cuModuleUnload", FnModuleUnload),
            module_get_function: sym!(lib, "cuModuleGetFunction", FnModuleGetFunction),
            func_set_attribute: sym!(lib, "cuFuncSetAttribute", FnFuncSetAttribute),
            func_get_attribute: sym!(lib, "cuFuncGetAttribute", FnFuncGetAttribute),
            launch_kernel: sym!(lib, "cuLaunchKernel", FnLaunchKernel),
            stream_create: sym!(lib, "cuStreamCreate", FnStreamCreate),
            stream_destroy: sym!(lib, "cuStreamDestroy_v2", FnStreamDestroy),
            stream_synchronize: sym!(lib, "cuStreamSynchronize", FnStreamSynchronize),
            event_create: sym!(lib, "cuEventCreate", FnEventCreate),
            event_record: sym!(lib, "cuEventRecord", FnEventRecord),
            event_synchronize: sym!(lib, "cuEventSynchronize", FnEventSynchronize),
            event_elapsed_time: sym!(lib, "cuEventElapsedTime", FnEventElapsedTime),
            event_destroy: sym!(lib, "cuEventDestroy_v2", FnEventDestroy),
            device_can_access_peer: sym!(lib, "cuDeviceCanAccessPeer", FnDeviceCanAccessPeer),
            ctx_enable_peer_access: sym!(lib, "cuCtxEnablePeerAccess", FnCtxEnablePeerAccess),
        };
        let rc = unsafe { (d.init)(0) };
        if rc != CUDA_SUCCESS {
            return Err(d.error(rc));
        }
        let mut v = 0;
        unsafe { (d.driver_get_version)(&mut v) };
        d.version = v;
        Ok(d)
    }

    /// Turn a `CUresult` into a descriptive error.
    pub fn error(&self, code: CUresult) -> CuError {
        let mut name = std::ptr::null();
        let mut msg = std::ptr::null();
        unsafe {
            (self.get_error_name)(code, &mut name);
            (self.get_error_string)(code, &mut msg);
        }
        CuError {
            code,
            name: if name.is_null() {
                format!("CUDA_ERROR_{code}")
            } else {
                unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
            },
            message: if msg.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned()
            },
        }
    }

    #[inline]
    pub fn check(&self, code: CUresult) -> Result<()> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(self.error(code))
        }
    }
}
