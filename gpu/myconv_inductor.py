import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

def measure_kernel_and_host(run_fn, tag="run"):
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=acts, record_shapes=False, profile_memory=False) as prof:
        torch.cuda.synchronize()
        with record_function(tag):
            run_fn()                      
        torch.cuda.synchronize()

    ka = prof.key_averages()

    def cuda_ms(e, self_only=True):
        us = (getattr(e, "self_cuda_time_total", None) if self_only else getattr(e, "cuda_time_total", None))
        if us is None:
            us = getattr(e, "cuda_time_total", None) if self_only else getattr(e, "self_cuda_time_total", None)
        return (us or 0.0) / 1000.0

    def cpu_ms(e, self_only=True):
        us = getattr(e, "self_cpu_time_total", None) if self_only else getattr(e, "cpu_time_total", None)
        if us is None:
            us = getattr(e, "cpu_time_total", 0.0)
        return us / 1000.0

    # 汇总
    total_kernel_ms = sum(cuda_ms(e, self_only=True) for e in ka)
    total_host_ms   = sum(cpu_ms(e,  self_only=True) for e in ka)

    # Pure kernel (excluding memcpy/memset, can be kept/removed as needed)
    def is_kernel_row(e):
        name = e.key.lower()
        return not ("memcpy" in name or "memset" in name)

    pure_kernel_ms = sum(e.self_cuda_time_total for e in ka if is_kernel_row(e)) / 1000.0

    # Top-N (CUDA)
    top_cuda = sorted(ka, key=lambda e: e.self_cuda_time_total, reverse=True)[:15]

    print(f"[{tag}] kernel(ms)={total_kernel_ms:.3f}  "
          f"pure_kernel(ms)={pure_kernel_ms:.3f}  host(ms)={total_host_ms:.3f}")
    for e in top_cuda:
        if e.self_cuda_time_total > 0:
            print(f"  {e.key:45s}  cuda={e.self_cuda_time_total/1000:.3f} ms  "
                  f"cpu={e.self_cpu_time_total/1000:.3f} ms  #{e.count}")

    return {"kernel_ms": total_kernel_ms,
            "pure_kernel_ms": pure_kernel_ms,
            "host_ms": total_host_ms, "prof": prof}

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 2, 3, 19, 19
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()

    # # Torch-Inductor compilation
    # scripted_model = torch.compile(model, backend="inductor")
    # out = scripted_model(x) 

    # Profiling tests
    kernel_sizes = [3, 5, 7]
    input_sizes = [32, 64, 128]

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        device = "cuda"
        activities += [ProfilerActivity.CUDA]

    test_count = 0
    results = []
    
    for H in input_sizes:
        W = H
        x = torch.randn(N, C, H, W).cuda()
        
        for kernel_size in kernel_sizes:
            test_count += 1
            print(f"Running Test {test_count}: Input size {H}x{W}, Kernel size {kernel_size}")
            
            model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=kernel_size, stride=1, padding=1).cuda().eval()
            
            def run_fn():
                scripted_model = torch.compile(model, backend="inductor")
                out = scripted_model(x)
                return out

            result = measure_kernel_and_host(
                run_fn, 
                tag=f"inductor_{H}x{W}_k{kernel_size}"
            )

            results.append({
                "test": test_count,
                "input_size": f"{H}x{W}",
                "kernel_size": kernel_size,
                "total_kernel_ms": result["kernel_ms"],
                "total_pure_kernel_ms": result["pure_kernel_ms"],
                "total_host_ms": result["host_ms"]
            })
            
    print("=== Summary ===")
    print(f"Completed {test_count} tests with PyTorch Inductor")
    print()
    print("Test | Input Size | Kernel | Total Kernel(ms) | Pure Kernel(ms) | Host(ms)")
    print("-" * 70)
    for r in results:
        print(f"{r['test']:4d} | {r['input_size']:10s} | {r['kernel_size']:6d} | "
              f"{r['total_kernel_ms']:15.3f} | {r['total_pure_kernel_ms']:13.3f} | {r['total_host_ms']:8.3f}")

    # # Test your solution
    # conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    # print("Inductor --- shape check:", out.shape == conv_ref.shape)
    # print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))