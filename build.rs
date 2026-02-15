use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Cargo sets this automatically when feature is enabled
    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();

    if !gpu_enabled {
        // GPU feature not enabled → do nothing
        return;
    }

    let kernel_dir = Path::new("cuda_kernels");

    for entry in std::fs::read_dir(kernel_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) == Some("cu") {
            println!("cargo:rerun-if-changed={}", path.display());

            let out = path.with_extension("ptx");

            let status = Command::new("nvcc")
                .args([
                    "-ptx",
                    path.to_str().unwrap(),
                    "-o",
                    out.to_str().unwrap(),
                    "-O3",
                    "--use_fast_math",
                ])
                .status()
                .expect("failed to run nvcc");

            assert!(status.success(), "nvcc failed on {:?}", path);
        }
    }
}
