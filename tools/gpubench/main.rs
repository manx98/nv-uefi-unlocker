#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::mem;
use std::path::PathBuf;
use std::process::Command;
use std::ptr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type CuResult = i32;
type CuDevice = i32;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuEvent = *mut c_void;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetLastError() -> u32;
}

type CuInit = unsafe extern "system" fn(u32) -> CuResult;
type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult;
type CuDeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, CuDevice) -> CuResult;
type CuDeviceTotalMem = unsafe extern "system" fn(*mut usize, CuDevice) -> CuResult;
type CuDriverGetVersion = unsafe extern "system" fn(*mut i32) -> CuResult;
type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
type CuCtxSynchronize = unsafe extern "system" fn() -> CuResult;
type CuModuleLoadData = unsafe extern "system" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;
type CuMemcpyHtoD = unsafe extern "system" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type CuLaunchKernel = unsafe extern "system" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuEventCreate = unsafe extern "system" fn(*mut CuEvent, u32) -> CuResult;
type CuEventRecord = unsafe extern "system" fn(CuEvent, *mut c_void) -> CuResult;
type CuEventSynchronize = unsafe extern "system" fn(CuEvent) -> CuResult;
type CuEventElapsedTime = unsafe extern "system" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type CuEventDestroy = unsafe extern "system" fn(CuEvent) -> CuResult;
type CuGetErrorName = unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult;
type CuGetErrorString = unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult;

const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const THREADS_PER_BLOCK: u32 = 128;
const WARPS_PER_BLOCK: u64 = 4;
const MMA_ITERS: u32 = 256;
const STRESS_MMA_ITERS: u32 = 16_384;
const FMA_ITERS: u32 = 65_536;
const TIMED_SAMPLES: usize = 3;

struct Api {
    library: *mut c_void,
    init: CuInit,
    device_get_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_get_name: CuDeviceGetName,
    device_get_attribute: CuDeviceGetAttribute,
    device_total_mem: CuDeviceTotalMem,
    driver_get_version: CuDriverGetVersion,
    ctx_create: CuCtxCreate,
    ctx_destroy: CuCtxDestroy,
    ctx_synchronize: CuCtxSynchronize,
    module_load_data: CuModuleLoadData,
    module_unload: CuModuleUnload,
    module_get_function: CuModuleGetFunction,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    launch_kernel: CuLaunchKernel,
    event_create: CuEventCreate,
    event_record: CuEventRecord,
    event_synchronize: CuEventSynchronize,
    event_elapsed_time: CuEventElapsedTime,
    event_destroy: CuEventDestroy,
    get_error_name: CuGetErrorName,
    get_error_string: CuGetErrorString,
}

impl Api {
    unsafe fn load_symbol<T: Copy>(library: *mut c_void, name: &'static [u8]) -> Result<T, String> {
        let address = GetProcAddress(library, name.as_ptr());
        if address.is_null() {
            return Err(format!(
                "CUDA driver is missing export {}",
                String::from_utf8_lossy(&name[..name.len() - 1])
            ));
        }
        Ok(mem::transmute_copy(&address))
    }

    fn load() -> Result<Self, String> {
        let wide: Vec<u16> = "nvcuda.dll".encode_utf16().chain([0]).collect();
        let library = unsafe { LoadLibraryW(wide.as_ptr()) };
        if library.is_null() {
            return Err(format!(
                "Unable to load nvcuda.dll (Windows error {})",
                unsafe { GetLastError() }
            ));
        }
        unsafe {
            let loaded = (|| -> Result<Self, String> {
                Ok(Self {
                    library,
                    init: Self::load_symbol(library, b"cuInit\0")?,
                    device_get_count: Self::load_symbol(library, b"cuDeviceGetCount\0")?,
                    device_get: Self::load_symbol(library, b"cuDeviceGet\0")?,
                    device_get_name: Self::load_symbol(library, b"cuDeviceGetName\0")?,
                    device_get_attribute: Self::load_symbol(library, b"cuDeviceGetAttribute\0")?,
                    device_total_mem: Self::load_symbol(library, b"cuDeviceTotalMem_v2\0")?,
                    driver_get_version: Self::load_symbol(library, b"cuDriverGetVersion\0")?,
                    ctx_create: Self::load_symbol(library, b"cuCtxCreate_v2\0")?,
                    ctx_destroy: Self::load_symbol(library, b"cuCtxDestroy_v2\0")?,
                    ctx_synchronize: Self::load_symbol(library, b"cuCtxSynchronize\0")?,
                    module_load_data: Self::load_symbol(library, b"cuModuleLoadData\0")?,
                    module_unload: Self::load_symbol(library, b"cuModuleUnload\0")?,
                    module_get_function: Self::load_symbol(library, b"cuModuleGetFunction\0")?,
                    mem_alloc: Self::load_symbol(library, b"cuMemAlloc_v2\0")?,
                    mem_free: Self::load_symbol(library, b"cuMemFree_v2\0")?,
                    memcpy_htod: Self::load_symbol(library, b"cuMemcpyHtoD_v2\0")?,
                    memcpy_dtoh: Self::load_symbol(library, b"cuMemcpyDtoH_v2\0")?,
                    launch_kernel: Self::load_symbol(library, b"cuLaunchKernel\0")?,
                    event_create: Self::load_symbol(library, b"cuEventCreate\0")?,
                    event_record: Self::load_symbol(library, b"cuEventRecord\0")?,
                    event_synchronize: Self::load_symbol(library, b"cuEventSynchronize\0")?,
                    event_elapsed_time: Self::load_symbol(library, b"cuEventElapsedTime\0")?,
                    event_destroy: Self::load_symbol(library, b"cuEventDestroy_v2\0")?,
                    get_error_name: Self::load_symbol(library, b"cuGetErrorName\0")?,
                    get_error_string: Self::load_symbol(library, b"cuGetErrorString\0")?,
                })
            })();
            if loaded.is_err() {
                FreeLibrary(library);
            }
            loaded
        }
    }

    fn error(&self, code: CuResult, operation: &str) -> String {
        let mut name = ptr::null();
        let mut description = ptr::null();
        unsafe {
            (self.get_error_name)(code, &mut name);
            (self.get_error_string)(code, &mut description);
        }
        let name = if name.is_null() {
            "CUDA_ERROR".to_owned()
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };
        let description = if description.is_null() {
            "unknown CUDA driver error".to_owned()
        } else {
            unsafe { CStr::from_ptr(description) }
                .to_string_lossy()
                .into_owned()
        };
        format!("{operation}: {name} ({code}) - {description}")
    }

    fn check(&self, code: CuResult, operation: &str) -> Result<(), String> {
        if code == 0 {
            Ok(())
        } else {
            Err(self.error(code, operation))
        }
    }
}

impl Drop for Api {
    fn drop(&mut self) {
        if !self.library.is_null() {
            unsafe {
                FreeLibrary(self.library);
            }
        }
    }
}

struct Session {
    api: Api,
    context: CuContext,
    device_index: i32,
    name: String,
    major: i32,
    minor: i32,
    sms: i32,
    memory_bytes: usize,
    driver_version: i32,
    blocks: u32,
    output: CuDevicePtr,
    output_bytes: usize,
    event_start: CuEvent,
    event_stop: CuEvent,
}

impl Session {
    fn new(index: i32) -> Result<Self, String> {
        let api = Api::load()?;
        api.check(unsafe { (api.init)(0) }, "cuInit")?;
        let mut count = 0;
        api.check(
            unsafe { (api.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if count == 0 {
            return Err("No CUDA-capable NVIDIA GPU was found.".into());
        }
        if index < 0 || index >= count {
            return Err(format!(
                "GPU index {index} is out of range (0..{}).",
                count - 1
            ));
        }

        let mut device = 0;
        api.check(
            unsafe { (api.device_get)(&mut device, index) },
            "cuDeviceGet",
        )?;
        let mut name_buf = [0i8; 256];
        api.check(
            unsafe { (api.device_get_name)(name_buf.as_mut_ptr(), name_buf.len() as i32, device) },
            "cuDeviceGetName",
        )?;
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut major = 0;
        let mut minor = 0;
        let mut sms = 0;
        let mut memory_bytes = 0usize;
        let mut driver_version = 0;
        api.check(
            unsafe {
                (api.device_get_attribute)(&mut major, ATTR_COMPUTE_CAPABILITY_MAJOR, device)
            },
            "query compute capability major",
        )?;
        api.check(
            unsafe {
                (api.device_get_attribute)(&mut minor, ATTR_COMPUTE_CAPABILITY_MINOR, device)
            },
            "query compute capability minor",
        )?;
        api.check(
            unsafe { (api.device_get_attribute)(&mut sms, ATTR_MULTIPROCESSOR_COUNT, device) },
            "query SM count",
        )?;
        api.check(
            unsafe { (api.device_total_mem)(&mut memory_bytes, device) },
            "query GPU memory",
        )?;
        api.check(
            unsafe { (api.driver_get_version)(&mut driver_version) },
            "cuDriverGetVersion",
        )?;

        let mut context = ptr::null_mut();
        api.check(
            unsafe { (api.ctx_create)(&mut context, 0, device) },
            "cuCtxCreate",
        )?;
        let blocks = (sms.max(1) as u32).saturating_mul(8).max(1);
        let output_bytes = blocks as usize * THREADS_PER_BLOCK as usize * 16 * 4;
        let mut output = 0u64;
        let mut event_start = ptr::null_mut();
        let mut event_stop = ptr::null_mut();
        let setup = (|| -> Result<(), String> {
            api.check(
                unsafe { (api.mem_alloc)(&mut output, output_bytes) },
                "cuMemAlloc",
            )?;
            api.check(
                unsafe { (api.event_create)(&mut event_start, 0) },
                "create start event",
            )?;
            api.check(
                unsafe { (api.event_create)(&mut event_stop, 0) },
                "create stop event",
            )?;
            Ok(())
        })();
        if let Err(error) = setup {
            unsafe {
                if !event_stop.is_null() {
                    (api.event_destroy)(event_stop);
                }
                if !event_start.is_null() {
                    (api.event_destroy)(event_start);
                }
                if output != 0 {
                    (api.mem_free)(output);
                }
                (api.ctx_destroy)(context);
            }
            return Err(error);
        }

        Ok(Self {
            api,
            context,
            device_index: index,
            name,
            major,
            minor,
            sms,
            memory_bytes,
            driver_version,
            blocks,
            output,
            output_bytes,
            event_start,
            event_stop,
        })
    }

    fn launch(&self, function: CuFunction, iterations: u32) -> Result<(), String> {
        let mut output = self.output;
        let mut iterations = iterations;
        let mut parameters: [*mut c_void; 2] = [
            (&mut output as *mut CuDevicePtr).cast(),
            (&mut iterations as *mut u32).cast(),
        ];
        self.api.check(
            unsafe {
                (self.api.launch_kernel)(
                    function,
                    self.blocks,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    fn launch_gemm(
        &self,
        function: CuFunction,
        a: CuDevicePtr,
        b_transposed: CuDevicePtr,
        c: CuDevicePtr,
        n: u32,
    ) -> Result<(), String> {
        let mut a = a;
        let mut b_transposed = b_transposed;
        let mut c = c;
        let mut n = n;
        let mut parameters: [*mut c_void; 4] = [
            (&mut a as *mut CuDevicePtr).cast(),
            (&mut b_transposed as *mut CuDevicePtr).cast(),
            (&mut c as *mut CuDevicePtr).cast(),
            (&mut n as *mut u32).cast(),
        ];
        self.api.check(
            unsafe {
                (self.api.launch_kernel)(
                    function,
                    n / 16,
                    n / 16,
                    1,
                    32,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launch FP16 WMMA GEMM",
        )
    }

    fn timed_gemm(
        &self,
        function: CuFunction,
        a: CuDevicePtr,
        b_transposed: CuDevicePtr,
        c: CuDevicePtr,
        n: u32,
    ) -> Result<f64, String> {
        self.api.check(
            unsafe { (self.api.event_record)(self.event_start, ptr::null_mut()) },
            "record GEMM start event",
        )?;
        self.launch_gemm(function, a, b_transposed, c, n)?;
        self.api.check(
            unsafe { (self.api.event_record)(self.event_stop, ptr::null_mut()) },
            "record GEMM stop event",
        )?;
        self.api.check(
            unsafe { (self.api.event_synchronize)(self.event_stop) },
            "synchronize GEMM stop event",
        )?;
        let mut milliseconds = 0.0f32;
        self.api.check(
            unsafe {
                (self.api.event_elapsed_time)(&mut milliseconds, self.event_start, self.event_stop)
            },
            "read GEMM elapsed time",
        )?;
        Ok(milliseconds as f64)
    }

    fn allocate_device(&self, bytes: usize, operation: &str) -> Result<CuDevicePtr, String> {
        let mut pointer = 0u64;
        self.api.check(
            unsafe { (self.api.mem_alloc)(&mut pointer, bytes) },
            operation,
        )?;
        Ok(pointer)
    }

    fn copy_to_device(
        &self,
        destination: CuDevicePtr,
        source: *const c_void,
        bytes: usize,
    ) -> Result<(), String> {
        self.api.check(
            unsafe { (self.api.memcpy_htod)(destination, source, bytes) },
            "copy GEMM input to device",
        )
    }

    fn synchronize(&self) -> Result<(), String> {
        self.api
            .check(unsafe { (self.api.ctx_synchronize)() }, "cuCtxSynchronize")
    }

    fn timed_launch(&self, function: CuFunction, iterations: u32) -> Result<f64, String> {
        self.api.check(
            unsafe { (self.api.event_record)(self.event_start, ptr::null_mut()) },
            "record start event",
        )?;
        self.launch(function, iterations)?;
        self.api.check(
            unsafe { (self.api.event_record)(self.event_stop, ptr::null_mut()) },
            "record stop event",
        )?;
        self.api.check(
            unsafe { (self.api.event_synchronize)(self.event_stop) },
            "synchronize stop event",
        )?;
        let mut milliseconds = 0.0f32;
        self.api.check(
            unsafe {
                (self.api.event_elapsed_time)(&mut milliseconds, self.event_start, self.event_stop)
            },
            "read event elapsed time",
        )?;
        Ok(milliseconds as f64)
    }

    fn read_output(&self, bytes: usize) -> Result<Vec<f32>, String> {
        if bytes > self.output_bytes {
            return Err("Internal output buffer size calculation was too small.".into());
        }
        let mut result = vec![0.0f32; bytes / 4];
        self.api.check(
            unsafe { (self.api.memcpy_dtoh)(result.as_mut_ptr().cast(), self.output, bytes) },
            "copy benchmark result to host",
        )?;
        Ok(result)
    }

    fn load_kernel(&self, ptx: &str, name: &str) -> Result<(CuModule, CuFunction), String> {
        let source =
            CString::new(ptx).map_err(|_| "PTX source contains an embedded NUL byte".to_owned())?;
        let mut module = ptr::null_mut();
        self.api.check(
            unsafe { (self.api.module_load_data)(&mut module, source.as_ptr().cast()) },
            &format!("JIT compile {name} PTX"),
        )?;
        let symbol = CString::new(name).unwrap();
        let mut function = ptr::null_mut();
        if let Err(error) = self.api.check(
            unsafe { (self.api.module_get_function)(&mut function, module, symbol.as_ptr()) },
            &format!("find CUDA kernel {name}"),
        ) {
            unsafe {
                (self.api.module_unload)(module);
            }
            return Err(error);
        }
        Ok((module, function))
    }

    fn unload_module(&self, module: CuModule) {
        if !module.is_null() {
            unsafe {
                (self.api.module_unload)(module);
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            if !self.event_stop.is_null() {
                (self.api.event_destroy)(self.event_stop);
            }
            if !self.event_start.is_null() {
                (self.api.event_destroy)(self.event_start);
            }
            if self.output != 0 {
                (self.api.mem_free)(self.output);
            }
            if !self.context.is_null() {
                (self.api.ctx_destroy)(self.context);
            }
        }
    }
}

#[derive(Clone)]
struct Options {
    device: i32,
    list: bool,
    output: Option<PathBuf>,
    stress_seconds: u64,
    stress_precision: String,
    max_temp_c: i32,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut result = Self {
            device: 0,
            list: false,
            output: None,
            stress_seconds: 0,
            stress_precision: "fp16".into(),
            max_temp_c: 85,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--list" => result.list = true,
                "--device" => result.device = parse_next(&mut args, "--device")?,
                "--output" => result.output = Some(PathBuf::from(next_arg(&mut args, "--output")?)),
                "--stress-seconds" => {
                    result.stress_seconds = parse_next(&mut args, "--stress-seconds")?
                }
                "--stress-precision" => {
                    result.stress_precision =
                        next_arg(&mut args, "--stress-precision")?.to_ascii_lowercase()
                }
                "--max-temp-c" => result.max_temp_c = parse_next(&mut args, "--max-temp-c")?,
                _ => return Err(format!("Unknown argument: {arg}. Use --help for usage.")),
            }
        }
        if result.stress_seconds > 3600 {
            return Err("--stress-seconds must be between 1 and 3600 (or 0 to disable).".into());
        }
        if !(65..=95).contains(&result.max_temp_c) {
            return Err("--max-temp-c must be between 65 and 95.".into());
        }
        if !["fp16", "bf16", "tf32", "fp8", "fp32"].contains(&result.stress_precision.as_str()) {
            return Err("--stress-precision must be fp16, bf16, tf32, fp8, or fp32.".into());
        }
        Ok(result)
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Missing value for {option}"))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    let value = next_arg(args, option)?;
    value
        .parse()
        .map_err(|_| format!("Invalid value for {option}: {value}"))
}

fn print_help() {
    println!("GPU Benchmark — NVIDIA CUDA-core / Tensor Core throughput and stability test\n\
Usage:\n\
  gpu-benchmark-x86_64-windows.exe [options]\n\
Options:\n\
  --list                       List CUDA GPUs and exit\n\
  --device INDEX               GPU index to test (default: 0)\n\
  --output PATH                JSON result path (default: auto-generated)\n\
  --stress-seconds N           Optional sustained-load test, 1..3600 seconds\n\
  --stress-precision MODE      fp16, bf16, tf32, fp8, or fp32 (default: fp16)\n\
  --max-temp-c C               Stop stress at this GPU temperature (default: 85 C)\n\
  -h, --help                   Show this help\n\n\
The short suite includes a 512x512 FP16 WMMA GEMM, FP32 FMA and Tensor Core issue-rate tests.\n\
The direct-load GEMM is a correctness/reference workload, not a tuned cuBLAS replacement.\n\
It performs no voltage, clock, power-limit, firmware, or register changes. Stress mode is opt-in and requires nvidia-smi telemetry.");
}

fn list_devices() -> Result<(), String> {
    let api = Api::load()?;
    api.check(unsafe { (api.init)(0) }, "cuInit")?;
    let mut count = 0;
    api.check(
        unsafe { (api.device_get_count)(&mut count) },
        "cuDeviceGetCount",
    )?;
    if count == 0 {
        println!("No CUDA-capable NVIDIA GPU found.");
        return Ok(());
    }
    for index in 0..count {
        let mut device = 0;
        api.check(
            unsafe { (api.device_get)(&mut device, index) },
            "cuDeviceGet",
        )?;
        let mut name = [0i8; 256];
        let mut major = 0;
        let mut minor = 0;
        let mut sms = 0;
        api.check(
            unsafe { (api.device_get_name)(name.as_mut_ptr(), name.len() as i32, device) },
            "cuDeviceGetName",
        )?;
        api.check(
            unsafe {
                (api.device_get_attribute)(&mut major, ATTR_COMPUTE_CAPABILITY_MAJOR, device)
            },
            "query compute capability",
        )?;
        api.check(
            unsafe {
                (api.device_get_attribute)(&mut minor, ATTR_COMPUTE_CAPABILITY_MINOR, device)
            },
            "query compute capability",
        )?;
        api.check(
            unsafe { (api.device_get_attribute)(&mut sms, ATTR_MULTIPROCESSOR_COUNT, device) },
            "query SM count",
        )?;
        println!(
            "[{index}] {}  compute {}.{}  {} SM",
            unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy(),
            major,
            minor,
            sms
        );
    }
    Ok(())
}

#[derive(Clone)]
struct BenchmarkResult {
    name: String,
    precision: String,
    kind: String,
    iterations: u32,
    flops: u64,
    samples_ms: Vec<f64>,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    tflops: f64,
    correctness: String,
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn tensor_ptx(
    version: &str,
    target: &str,
    instruction: &str,
    packed: &str,
    a_count: usize,
    b_count: usize,
) -> String {
    let mut ptx = format!(
        r#".version {version}
.target {target}
.address_size 64
.visible .entry tensor_mma(
    .param .u64 out,
    .param .u32 iters
)
{{
    .reg .pred %p<1>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .reg .b32 %a<{a_count}>;
    .reg .b32 %b<{b_count}>;
    .reg .b32 %c<16>;
"#
    );
    for index in 0..a_count {
        let _ = writeln!(ptx, "    mov.b32 %a{index}, {packed};");
    }
    for index in 0..b_count {
        let _ = writeln!(ptx, "    mov.b32 %b{index}, {packed};");
    }
    for index in 0..16 {
        let _ = writeln!(ptx, "    mov.b32 %c{index}, 0;");
    }
    let a_tuple = (0..a_count)
        .map(|i| format!("%a{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let b_tuple = (0..b_count)
        .map(|i| format!("%b{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    ptx.push_str(
        r#"
    mov.u32 %r0, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.u32 %r3, %r2, 128, %r0;
    mul.wide.u32 %rd0, %r3, 64;
    ld.param.u64 %rd1, [out];
    add.u64 %rd1, %rd1, %rd0;
    ld.param.u32 %r4, [iters];
    mov.u32 %r5, 0;

LOOP:
"#,
    );
    for group in 0..4 {
        let base = group * 4;
        let c_tuple = (base..base + 4)
            .map(|i| format!("%c{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            ptx,
            "    {instruction} {{{c_tuple}}}, {{{a_tuple}}}, {{{b_tuple}}}, {{{c_tuple}}};"
        );
    }
    ptx.push_str(
        r#"    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p0, %r5, %r4;
    @%p0 bra LOOP;

"#,
    );
    for index in 0..16 {
        let _ = writeln!(ptx, "    st.global.f32 [%rd1+{}], %c{};", index * 4, index);
    }
    ptx.push_str("    ret;\n}\n");
    ptx
}

fn fp32_ptx() -> &'static str {
    r#".version 7.0
.target sm_75
.address_size 64
.visible .entry fp32_fma(
    .param .u64 out,
    .param .u32 iters
)
{
    .reg .pred %p<1>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<24>;

    mov.f32 %f0, 1.0; mov.f32 %f1, 1.0; mov.f32 %f2, 1.0; mov.f32 %f3, 1.0;
    mov.f32 %f4, 1.0; mov.f32 %f5, 1.0; mov.f32 %f6, 1.0; mov.f32 %f7, 1.0;
    mov.f32 %f8, 1.00001; mov.f32 %f9, 1.00001; mov.f32 %f10, 1.00001; mov.f32 %f11, 1.00001;
    mov.f32 %f12, 1.00001; mov.f32 %f13, 1.00001; mov.f32 %f14, 1.00001; mov.f32 %f15, 1.00001;
    mov.f32 %f16, 0.000001; mov.f32 %f17, 0.000001; mov.f32 %f18, 0.000001; mov.f32 %f19, 0.000001;
    mov.f32 %f20, 0.000001; mov.f32 %f21, 0.000001; mov.f32 %f22, 0.000001; mov.f32 %f23, 0.000001;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mad.lo.u32 %r2, %r1, 128, %r0;
    mul.wide.u32 %rd0, %r2, 32;
    ld.param.u64 %rd1, [out];
    add.u64 %rd1, %rd1, %rd0;
    ld.param.u32 %r3, [iters];
    mov.u32 %r4, 0;

LOOP:
    fma.rn.f32 %f0, %f0, %f8, %f16;
    fma.rn.f32 %f1, %f1, %f9, %f17;
    fma.rn.f32 %f2, %f2, %f10, %f18;
    fma.rn.f32 %f3, %f3, %f11, %f19;
    fma.rn.f32 %f4, %f4, %f12, %f20;
    fma.rn.f32 %f5, %f5, %f13, %f21;
    fma.rn.f32 %f6, %f6, %f14, %f22;
    fma.rn.f32 %f7, %f7, %f15, %f23;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p0, %r4, %r3;
    @%p0 bra LOOP;

    st.global.f32 [%rd1], %f0;
    st.global.f32 [%rd1+4], %f1;
    st.global.f32 [%rd1+8], %f2;
    st.global.f32 [%rd1+12], %f3;
    st.global.f32 [%rd1+16], %f4;
    st.global.f32 [%rd1+20], %f5;
    st.global.f32 [%rd1+24], %f6;
    st.global.f32 [%rd1+28], %f7;
    ret;
}
"#
}

fn fp16_gemm_ptx() -> &'static str {
    r#".version 7.0
.target sm_70
.address_size 64
.visible .entry fp16_wmma_gemm(
    .param .u64 a_ptr,
    .param .u64 b_transposed_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
{
    .reg .pred %p<1>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .b32 %a<8>;
    .reg .b32 %b<8>;
    .reg .b32 %c<8>;

    ld.param.u64 %rd0, [a_ptr];
    ld.param.u64 %rd1, [b_transposed_ptr];
    ld.param.u64 %rd2, [c_ptr];
    ld.param.u32 %r0, [n];

    mov.u32 %r1, %ctaid.y;
    shl.b32 %r2, %r1, 4;
    mov.u32 %r3, %ctaid.x;
    shl.b32 %r4, %r3, 4;

    mul.lo.u32 %r5, %r2, %r0;
    mul.wide.u32 %rd3, %r5, 2;
    add.u64 %rd4, %rd0, %rd3;
    mul.lo.u32 %r6, %r4, %r0;
    mul.wide.u32 %rd5, %r6, 2;
    add.u64 %rd6, %rd1, %rd5;
    mad.lo.u32 %r7, %r2, %r0, %r4;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd2, %rd7;

    mov.b32 %c0, 0; mov.b32 %c1, 0; mov.b32 %c2, 0; mov.b32 %c3, 0;
    mov.b32 %c4, 0; mov.b32 %c5, 0; mov.b32 %c6, 0; mov.b32 %c7, 0;
    mov.u32 %r8, 0;

K_LOOP:
    mul.wide.u32 %rd9, %r8, 2;
    add.u64 %rd10, %rd4, %rd9;
    add.u64 %rd11, %rd6, %rd9;
    wmma.load.a.sync.aligned.m16n16k16.global.row.f16 {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, [%rd10], %r0;
    wmma.load.b.sync.aligned.m16n16k16.global.col.f16 {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}, [%rd11], %r0;
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
        {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7},
        {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7},
        {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7};
    add.u32 %r8, %r8, 16;
    setp.lt.u32 %p0, %r8, %r0;
    @%p0 bra K_LOOP;

    wmma.store.d.sync.aligned.m16n16k16.global.row.f32 [%rd8],
        {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, %r0;
    ret;
}
"#
}

fn eighths_to_f16(value: i32) -> u16 {
    match value {
        -4 => 0xb800,
        -3 => 0xb600,
        -2 => 0xb400,
        -1 => 0xb000,
        0 => 0x0000,
        1 => 0x3000,
        2 => 0x3400,
        3 => 0x3600,
        4 => 0x3800,
        _ => unreachable!("pattern must be in -4..=4"),
    }
}

fn pattern_eighths(row: usize, col: usize) -> i32 {
    ((row * 17 + col * 13 + (row * col) % 7) % 9) as i32 - 4
}

fn run_fp16_gemm(session: &Session) -> Result<BenchmarkResult, String> {
    const N: usize = 512;
    let element_count = N * N;
    let input_bytes = element_count * mem::size_of::<u16>();
    let output_bytes = element_count * mem::size_of::<f32>();
    let mut a_half = Vec::with_capacity(element_count);
    let mut b_transposed_half = Vec::with_capacity(element_count);
    let mut a_float = Vec::with_capacity(element_count);
    let mut b_transposed_float = Vec::with_capacity(element_count);
    for row in 0..N {
        for col in 0..N {
            let a = pattern_eighths(row, col);
            let b = pattern_eighths(col + 3, row + 5);
            a_half.push(eighths_to_f16(a));
            b_transposed_half.push(eighths_to_f16(b));
            a_float.push(a as f32 * 0.125);
            b_transposed_float.push(b as f32 * 0.125);
        }
    }

    let d_a = session.allocate_device(input_bytes, "allocate GEMM A")?;
    let d_b = match session.allocate_device(input_bytes, "allocate GEMM B") {
        Ok(pointer) => pointer,
        Err(error) => {
            unsafe {
                (session.api.mem_free)(d_a);
            }
            return Err(error);
        }
    };
    let d_c = match session.allocate_device(output_bytes, "allocate GEMM C") {
        Ok(pointer) => pointer,
        Err(error) => {
            unsafe {
                (session.api.mem_free)(d_b);
                (session.api.mem_free)(d_a);
            }
            return Err(error);
        }
    };
    let (module, function) = match session.load_kernel(fp16_gemm_ptx(), "fp16_wmma_gemm") {
        Ok(pair) => pair,
        Err(error) => {
            unsafe {
                (session.api.mem_free)(d_c);
                (session.api.mem_free)(d_b);
                (session.api.mem_free)(d_a);
            }
            return Err(error);
        }
    };

    let result = (|| -> Result<BenchmarkResult, String> {
        session.copy_to_device(d_a, a_half.as_ptr().cast(), input_bytes)?;
        session.copy_to_device(d_b, b_transposed_half.as_ptr().cast(), input_bytes)?;
        session.launch_gemm(function, d_a, d_b, d_c, N as u32)?;
        session.synchronize()?;
        let mut samples = Vec::with_capacity(TIMED_SAMPLES);
        for _ in 0..TIMED_SAMPLES {
            samples.push(session.timed_gemm(function, d_a, d_b, d_c, N as u32)?);
        }
        let mut output = vec![0.0f32; element_count];
        session.api.check(
            unsafe { (session.api.memcpy_dtoh)(output.as_mut_ptr().cast(), d_c, output_bytes) },
            "copy GEMM output to host",
        )?;

        let mut max_absolute_error = 0.0f64;
        let mut max_reference = 0.0f64;
        for row in 0..N {
            for col in 0..N {
                let mut reference = 0.0f64;
                for k in 0..N {
                    reference +=
                        a_float[row * N + k] as f64 * b_transposed_float[col * N + k] as f64;
                }
                let actual = output[row * N + col] as f64;
                if !actual.is_finite() {
                    return Err("FP16 GEMM produced a non-finite output.".into());
                }
                max_absolute_error = max_absolute_error.max((actual - reference).abs());
                max_reference = max_reference.max(reference.abs());
            }
        }
        let max_relative_error = max_absolute_error / max_reference.max(1.0e-12);
        if max_relative_error > 0.002 {
            return Err(format!("FP16 GEMM correctness check failed: max relative error {max_relative_error:.6} (limit 0.002)."));
        }
        let median_ms = median(&samples);
        let flops = 2 * (N as u64).pow(3);
        let tflops = flops as f64 / (median_ms / 1000.0) / 1.0e12;
        Ok(BenchmarkResult {
            name: format!("FP16 GEMM {N}x{N}"),
            precision: "FP16".into(),
            kind: "WMMA GEMM with direct global-memory tile loads".into(),
            iterations: (N / 16) as u32,
            flops,
            min_ms: samples.iter().copied().fold(f64::INFINITY, f64::min),
            max_ms: samples.iter().copied().fold(0.0, f64::max),
            samples_ms: samples,
            median_ms,
            tflops,
            correctness: format!("PASS (max_rel_error={max_relative_error:.6})"),
        })
    })();

    session.unload_module(module);
    unsafe {
        (session.api.mem_free)(d_c);
        (session.api.mem_free)(d_b);
        (session.api.mem_free)(d_a);
    }
    result
}

fn benchmark(
    session: &Session,
    name: &str,
    precision: &str,
    kind: &str,
    ptx: &str,
    kernel_name: &str,
    iterations: u32,
    flops_per_warp_instruction: u64,
    expected: f32,
    output_floats_per_warp: usize,
) -> Result<BenchmarkResult, String> {
    let (module, function) = session.load_kernel(ptx, kernel_name)?;
    let run = (|| -> Result<BenchmarkResult, String> {
        for _ in 0..2 {
            session.launch(function, iterations)?;
        }
        session.synchronize()?;
        let mut samples = Vec::with_capacity(TIMED_SAMPLES);
        for _ in 0..TIMED_SAMPLES {
            samples.push(session.timed_launch(function, iterations)?);
        }
        let output_bytes =
            session.blocks as usize * THREADS_PER_BLOCK as usize * output_floats_per_warp * 4;
        let output = session.read_output(output_bytes)?;
        let tolerance = (expected.abs() * 0.00001).max(0.0001);
        let bad = output
            .iter()
            .enumerate()
            .filter(|(_, value)| !value.is_finite() || (**value - expected).abs() > tolerance)
            .take(4)
            .collect::<Vec<_>>();
        if !bad.is_empty() {
            return Err(format!("{name} result verification failed; expected {expected}, sample mismatches: {bad:?}"));
        }
        let blocks = session.blocks as u64;
        let flops = if precision == "FP32" {
            blocks * THREADS_PER_BLOCK as u64 * iterations as u64 * 8 * 2
        } else {
            blocks * WARPS_PER_BLOCK * iterations as u64 * 4 * flops_per_warp_instruction
        };
        let median_ms = median(&samples);
        let seconds = median_ms / 1000.0;
        let tflops = if seconds > 0.0 {
            flops as f64 / seconds / 1.0e12
        } else {
            0.0
        };
        Ok(BenchmarkResult {
            name: name.into(),
            precision: precision.into(),
            kind: kind.into(),
            iterations,
            flops,
            min_ms: samples.iter().copied().fold(f64::INFINITY, f64::min),
            max_ms: samples.iter().copied().fold(0.0, f64::max),
            samples_ms: samples,
            median_ms,
            tflops,
            correctness: "PASS".into(),
        })
    })();
    session.unload_module(module);
    run
}

#[derive(Clone)]
struct Telemetry {
    elapsed_seconds: f64,
    temperature_c: Option<i32>,
    power_w: Option<f64>,
    sm_clock_mhz: Option<i32>,
    utilization_percent: Option<i32>,
    pstate: String,
}

fn parse_optional<T: std::str::FromStr>(value: &str) -> Option<T> {
    let value = value.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("n/a")
        || value.eq_ignore_ascii_case("[not supported]")
    {
        None
    } else {
        value.parse().ok()
    }
}

fn query_telemetry(index: i32, elapsed: f64) -> Result<Telemetry, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--id",
            &index.to_string(),
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("Unable to run nvidia-smi for stress telemetry: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi telemetry query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .to_owned();
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    if fields.len() < 5 {
        return Err(format!("Unexpected nvidia-smi telemetry response: {line}"));
    }
    let temperature_c = parse_optional(fields[0]);
    if temperature_c.is_none() {
        return Err("GPU temperature is unavailable; refusing to run sustained stress without thermal telemetry.".into());
    }
    Ok(Telemetry {
        elapsed_seconds: elapsed,
        temperature_c,
        power_w: parse_optional(fields[1]),
        sm_clock_mhz: parse_optional(fields[2]),
        utilization_percent: parse_optional(fields[3]),
        pstate: fields[4].to_owned(),
    })
}

fn query_nvidia_driver_version(index: i32) -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--id",
            &index.to_string(),
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if version.is_empty() || version.eq_ignore_ascii_case("n/a") {
        None
    } else {
        Some(version)
    }
}

fn run_stress(
    session: &Session,
    seconds: u64,
    precision: &str,
    max_temp: i32,
) -> Result<(Vec<Telemetry>, String), String> {
    let (version, target, instruction, packed, a_count, b_count, kernel, iterations) =
        match precision {
            "fp16" if session.major >= 8 => (
                "7.0",
                "sm_80",
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
                "0x3c003c00",
                4,
                2,
                "tensor_mma",
                STRESS_MMA_ITERS,
            ),
            "fp16" => (
                "7.0",
                "sm_75",
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
                "0x3c003c00",
                2,
                1,
                "tensor_mma",
                STRESS_MMA_ITERS,
            ),
            "bf16" if session.major >= 8 => (
                "7.0",
                "sm_80",
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                "0x3f803f80",
                4,
                2,
                "tensor_mma",
                STRESS_MMA_ITERS,
            ),
            "tf32" if session.major >= 8 => (
                "7.0",
                "sm_80",
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
                "0x3f800000",
                4,
                2,
                "tensor_mma",
                STRESS_MMA_ITERS,
            ),
            "fp8"
                if (session.major, session.minor) >= (8, 9)
                    && session.driver_version >= 12_040 =>
            (
                "8.4",
                "sm_89",
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
                "0x38383838",
                4,
                2,
                "tensor_mma",
                STRESS_MMA_ITERS,
            ),
            "fp32" => ("7.0", "sm_75", "", "", 0, 0, "fp32_fma", FMA_ITERS),
        _ => {
            return Err(format!(
                "Stress precision {precision} is not supported by this GPU / driver (compute {}.{}, CUDA Driver API {}).",
                session.major,
                session.minor,
                session.driver_version
            ))
            }
        };
    let ptx = if precision == "fp32" {
        fp32_ptx().to_owned()
    } else {
        tensor_ptx(version, target, instruction, packed, a_count, b_count)
    };
    let (module, function) = session.load_kernel(&ptx, kernel)?;
    let result = (|| -> Result<(Vec<Telemetry>, String), String> {
        println!("\n[stress] {} seconds, {precision}, thermal stop at {max_temp} C. No clock/voltage/power settings will be changed.", seconds);
        let started = Instant::now();
        let mut samples = Vec::new();
        let mut status = "completed".to_owned();
        let mut last_sample = Duration::from_secs(0);
        loop {
            session.launch(function, iterations)?;
            session.synchronize()?;
            let elapsed = started.elapsed();
            if elapsed.saturating_sub(last_sample) >= Duration::from_secs(1)
                || samples.is_empty()
                || elapsed >= Duration::from_secs(seconds)
            {
                let sample = query_telemetry(session.device_index, elapsed.as_secs_f64())?;
                let temp = sample.temperature_c.unwrap_or(0);
                println!(
                    "[stress] {:>6.1}s  {:>2} C  {:>6.1} W  {:>4} MHz  {:>3}%  {}",
                    sample.elapsed_seconds,
                    temp,
                    sample.power_w.unwrap_or(0.0),
                    sample.sm_clock_mhz.unwrap_or(0),
                    sample.utilization_percent.unwrap_or(0),
                    sample.pstate
                );
                samples.push(sample);
                last_sample = elapsed;
                if temp >= max_temp {
                    status = "stopped_thermal_limit".into();
                    println!("[stress] thermal limit reached; stopping the workload.");
                    break;
                }
            }
            if elapsed >= Duration::from_secs(seconds) {
                break;
            }
        }
        Ok((samples, status))
    })();
    session.unload_module(module);
    result
}

fn json_escape(value: &str) -> String {
    let mut result = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(result, "\\u{:04x}", c as u32);
            }
            c => result.push(c),
        }
    }
    result
}

fn result_json(
    session: &Session,
    nvidia_driver_version: Option<&str>,
    results: &[BenchmarkResult],
    stress: &[Telemetry],
    stress_status: &str,
    requested_stress: u64,
    stress_precision: &str,
    max_temp_c: i32,
) -> String {
    let mut json = String::new();
    let driver_api = format!(
        "{}.{}",
        session.driver_version / 1000,
        (session.driver_version % 1000) / 10
    );
    let _ = writeln!(json, "{{");
    let _ = writeln!(json, "  \"tool\": \"gpu-benchmark\",");
    let _ = writeln!(json, "  \"version\": \"0.1.0\",");
    let _ = writeln!(
        json,
        "  \"timestamp_unix\": {},",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let nvidia_driver = nvidia_driver_version
        .map(|v| format!("\"{}\"", json_escape(v)))
        .unwrap_or_else(|| "null".into());
    let _ = writeln!(json, "  \"gpu\": {{\"index\": {}, \"name\": \"{}\", \"compute_capability\": \"{}.{}\", \"sms\": {}, \"memory_bytes\": {}, \"nvidia_driver_version\": {}, \"cuda_driver_api_version\": \"{}\"}},",
        session.device_index, json_escape(&session.name), session.major, session.minor, session.sms, session.memory_bytes, nvidia_driver, driver_api);
    let _ = writeln!(json, "  \"method\": \"512x512 FP16 WMMA GEMM plus warp-level CUDA-core / Tensor Core instruction-throughput microbenchmarks\",");
    let _ = writeln!(json, "  \"benchmarks\": [");
    for (index, result) in results.iter().enumerate() {
        let samples = result
            .samples_ms
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
            .join(", ");
        let comma = if index + 1 == results.len() { "" } else { "," };
        let _ = writeln!(json, "    {{\"name\": \"{}\", \"precision\": \"{}\", \"kind\": \"{}\", \"iterations\": {}, \"operation_count\": {}, \"samples_ms\": [{}], \"median_ms\": {:.4}, \"min_ms\": {:.4}, \"max_ms\": {:.4}, \"tflops\": {:.4}, \"correctness\": \"{}\"}}{}",
            json_escape(&result.name), json_escape(&result.precision), json_escape(&result.kind), result.iterations, result.flops, samples,
            result.median_ms, result.min_ms, result.max_ms, result.tflops, result.correctness, comma);
    }
    let _ = writeln!(json, "  ],");
    let _ = writeln!(
        json,
        "  \"stress\": {{\"requested_seconds\": {}, \"precision\": \"{}\", \"max_temperature_c\": {}, \"status\": \"{}\", \"samples\": [",
        requested_stress, json_escape(stress_precision), max_temp_c, stress_status
    );
    for (index, sample) in stress.iter().enumerate() {
        let comma = if index + 1 == stress.len() { "" } else { "," };
        let _ = writeln!(json, "    {{\"elapsed_seconds\": {:.2}, \"temperature_c\": {}, \"power_w\": {}, \"sm_clock_mhz\": {}, \"utilization_percent\": {}, \"pstate\": \"{}\"}}{}",
            sample.elapsed_seconds, sample.temperature_c.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
            sample.power_w.map(|v| format!("{v:.2}")).unwrap_or_else(|| "null".into()),
            sample.sm_clock_mhz.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
            sample.utilization_percent.map(|v| v.to_string()).unwrap_or_else(|| "null".into()), json_escape(&sample.pstate), comma);
    }
    let _ = writeln!(json, "  ]}}\n}}");
    json
}

fn run(options: Options) -> Result<(), String> {
    if options.list {
        return list_devices();
    }
    let session = Session::new(options.device)?;
    let nvidia_driver_version = query_nvidia_driver_version(session.device_index);
    println!(
        "GPU: {} (compute {}.{}, {} SM, {:.1} GiB VRAM)",
        session.name,
        session.major,
        session.minor,
        session.sms,
        session.memory_bytes as f64 / 1024.0f64.powi(3)
    );
    println!(
        "NVIDIA driver: {}  |  CUDA Driver API: {}.{}  |  launch geometry: {} blocks x {} threads",
        nvidia_driver_version
            .as_deref()
            .unwrap_or("not reported by nvidia-smi"),
        session.driver_version / 1000,
        (session.driver_version % 1000) / 10,
        session.blocks,
        THREADS_PER_BLOCK
    );
    println!("\nIncludes a 512x512 FP16 WMMA GEMM and short CUDA-core / Tensor Core issue-rate microbenchmarks.");
    println!(
        "The direct-load GEMM is a correctness/reference workload, not a tuned cuBLAS replacement."
    );
    println!("It does not change GPU clocks, voltage, power limits, firmware, or registers.");

    let mut results = Vec::new();
    let gemm = run_fp16_gemm(&session)?;
    println!(
        "[ok] {:<18} {:>8.3} TFLOPS  median {:>7.3} ms  {}",
        gemm.name, gemm.tflops, gemm.median_ms, gemm.correctness
    );
    results.push(gemm);

    let fma = benchmark(
        &session,
        "FP32 FMA",
        "FP32",
        "CUDA-core FMA issue-rate",
        fp32_ptx(),
        "fp32_fma",
        FMA_ITERS,
        0,
        {
            let mut value = 1.0f32;
            for _ in 0..FMA_ITERS {
                value = value.mul_add(1.00001, 0.000001);
            }
            value
        },
        8,
    )?;
    println!(
        "[ok] {:<10} {:>8.3} TFLOPS  median {:>7.3} ms  correctness {}",
        fma.precision, fma.tflops, fma.median_ms, fma.correctness
    );
    results.push(fma);

    let mut modes = if session.major >= 8 {
        vec![(
            "FP16",
            "FP16 Tensor Core",
            "7.0",
            "sm_80",
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
            "0x3c003c00",
            16u32,
            4096u64,
            4usize,
            2usize,
        )]
    } else {
        vec![(
            "FP16",
            "FP16 Tensor Core",
            "7.0",
            "sm_75",
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
            "0x3c003c00",
            8u32,
            2048u64,
            2usize,
            1usize,
        )]
    };
    if session.major >= 8 {
        modes.push((
            "BF16",
            "BF16 Tensor Core",
            "7.0",
            "sm_80",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "0x3f803f80",
            16,
            4096,
            4,
            2,
        ));
        modes.push((
            "TF32",
            "TF32 Tensor Core",
            "7.0",
            "sm_80",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
            "0x3f800000",
            8,
            2048,
            4,
            2,
        ));
    }
    if (session.major, session.minor) >= (8, 9) && session.driver_version >= 12_040 {
        modes.push((
            "FP8 E4M3",
            "FP8 Tensor Core",
            "8.4",
            "sm_89",
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            "0x38383838",
            32,
            8192,
            4,
            2,
        ));
    } else if (session.major, session.minor) >= (8, 9) {
        println!("[skip] FP8 E4M3 requires CUDA Driver API 12.4 or newer for PTX ISA 8.4.");
    }
    for (precision, name, version, target, instruction, packed, k, flops, a_count, b_count) in modes
    {
        let expected = MMA_ITERS as f32 * k as f32;
        let ptx = tensor_ptx(version, target, instruction, packed, a_count, b_count);
        let result = benchmark(
            &session,
            name,
            precision,
            "warp-level Tensor Core MMA issue-rate",
            &ptx,
            "tensor_mma",
            MMA_ITERS,
            flops,
            expected,
            16,
        )?;
        println!(
            "[ok] {:<10} {:>8.3} TFLOPS  median {:>7.3} ms  correctness {}",
            result.precision, result.tflops, result.median_ms, result.correctness
        );
        results.push(result);
    }

    let mut stress_samples = Vec::new();
    let mut stress_status = "not_requested".to_owned();
    if options.stress_seconds > 0 {
        let (samples, status) = run_stress(
            &session,
            options.stress_seconds,
            &options.stress_precision,
            options.max_temp_c,
        )?;
        stress_samples = samples;
        stress_status = status;
    }

    let json = result_json(
        &session,
        nvidia_driver_version.as_deref(),
        &results,
        &stress_samples,
        &stress_status,
        options.stress_seconds,
        &options.stress_precision,
        options.max_temp_c,
    );
    let output = options.output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "gpu-benchmark-result-{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ))
    });
    fs::write(&output, json)
        .map_err(|e| format!("Unable to write result file {}: {e}", output.display()))?;
    println!("\n[ok] JSON results saved to {}", output.display());
    if options.stress_seconds == 0 {
        println!(
            "Tip: use --stress-seconds 300 for an opt-in stability run (temperature guarded). "
        );
    }
    Ok(())
}

fn main() {
    let result = Options::parse().and_then(run);
    if let Err(error) = result {
        eprintln!("[error] {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_and_even_sample_counts() {
        assert_eq!(median(&[9.0, 1.0, 5.0]), 5.0);
        assert_eq!(median(&[8.0, 2.0, 6.0, 4.0]), 5.0);
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn json_escape_handles_quotes_slashes_and_control_chars() {
        assert_eq!(json_escape("GPU \"A\"\\B\n"), "GPU \\\"A\\\"\\\\B\\n");
    }

    #[test]
    fn turing_fp16_uses_the_sm75_supported_instruction_shape() {
        let ptx = tensor_ptx(
            "7.0",
            "sm_75",
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
            "0x3c003c00",
            2,
            1,
        );
        assert!(ptx.contains(".reg .b32 %a<2>;"));
        assert!(ptx.contains(".reg .b32 %b<1>;"));
        assert!(ptx.contains("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"));
    }

    #[test]
    fn ada_fp8_uses_the_required_ptx_isa_and_register_fragment_sizes() {
        let ptx = tensor_ptx(
            "8.4",
            "sm_89",
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            "0x38383838",
            4,
            2,
        );
        assert!(ptx.starts_with(".version 8.4\n.target sm_89"));
        assert!(ptx.contains(".reg .b32 %a<4>;"));
        assert!(ptx.contains(".reg .b32 %b<2>;"));
    }

    #[test]
    fn fp16_gemm_uses_wmma_and_fp32_accumulation() {
        let ptx = fp16_gemm_ptx();
        assert!(ptx.contains("wmma.load.a.sync.aligned.m16n16k16.global.row.f16"));
        assert!(ptx.contains("wmma.load.b.sync.aligned.m16n16k16.global.col.f16"));
        assert!(ptx.contains("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32"));
        assert!(ptx.contains("wmma.store.d.sync.aligned.m16n16k16.global.row.f32"));
    }

    #[test]
    fn eighths_are_encoded_as_exact_binary_fp16_values() {
        assert_eq!(eighths_to_f16(-4), 0xb800);
        assert_eq!(eighths_to_f16(-1), 0xb000);
        assert_eq!(eighths_to_f16(0), 0x0000);
        assert_eq!(eighths_to_f16(1), 0x3000);
        assert_eq!(eighths_to_f16(4), 0x3800);
    }
}
