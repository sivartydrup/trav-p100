use std::path::{Path, PathBuf};
use std::process::Command;

/// Compile `cu/kernels.cu` into a cubin for the P100 and embed it in the crate.
///
/// `P100_CUDA_ARCH` overrides the target (e.g. `sm_60`, or `sm_60;sm_61` is not
/// supported here because a cubin holds exactly one architecture; use `sm_60`).
fn main() {
    println!("cargo:rerun-if-changed=cu/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=P100_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=P100_PTXAS_VERBOSE");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let arch = std::env::var("P100_CUDA_ARCH").unwrap_or_else(|_| "sm_60".to_string());
    let nvcc = find_nvcc();
    let src = Path::new("cu/kernels.cu");
    let dst = out_dir.join("kernels.cubin");

    let mut cmd = Command::new(&nvcc);
    cmd.args([
        "-cubin",
        "-std=c++17",
        "-O3",
        "-arch",
        &arch,
        "--compiler-options",
        "-fno-strict-aliasing",
        "-o",
    ])
    .arg(&dst)
    .arg(src);
    if std::env::var("P100_PTXAS_VERBOSE").is_ok() {
        cmd.args(["-Xptxas", "-v"]);
    }

    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));
    if !out.status.success() {
        panic!(
            "nvcc failed for {arch}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // Surface ptxas register/spill information in the build log; it is the only
    // occupancy signal available without nvprof on this box.
    let err = String::from_utf8_lossy(&out.stderr);
    for line in err.lines() {
        if line.contains("registers") || line.contains("spill") || line.contains("Function properties")
        {
            println!("cargo:warning={line}");
        }
    }
    println!("cargo:rustc-env=P100_KERNEL_ARCH={arch}");
}

fn find_nvcc() -> PathBuf {
    if let Ok(cuda) = std::env::var("CUDA_HOME") {
        let p = Path::new(&cuda).join("bin/nvcc");
        if p.exists() {
            return p;
        }
    }
    for c in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc"] {
        if Path::new(c).exists() {
            return PathBuf::from(c);
        }
    }
    // Fall back to PATH.
    PathBuf::from("nvcc")
}
