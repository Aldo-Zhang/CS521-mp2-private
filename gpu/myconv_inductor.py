import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import argparse

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

    total_kernel_ms = sum(cuda_ms(e, self_only=True) for e in ka)
    total_host_ms   = sum(cpu_ms(e,  self_only=True) for e in ka)

    return {"kernel_ms": total_kernel_ms,
            "host_ms": total_host_ms, "prof": prof}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Inductor Performance Test')
    parser.add_argument('--input_size', type=int, required=True, 
                       help='Input size (H=W, e.g., 32 for 32x32)')
    parser.add_argument('--kernel_size', type=int, required=True,
                       help='Kernel size (e.g., 3, 5, 7)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--channels', type=int, default=3,
                       help='Input channels (default: 4)')
    parser.add_argument('--out_channels', type=int, default=8,
                       help='Output channels (default: 8)')
    
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    # 从命令行参数获取配置
    H = W = args.input_size
    kernel_size = args.kernel_size
    N = args.batch_size
    C = args.channels
    out_channels = args.out_channels
    
    print(f"=== PyTorch Inductor Performance Test ===")
    print(f"Input size: {H}x{W}")
    print(f"Kernel size: {kernel_size}")
    print(f"Batch size: {N}")
    print(f"Input channels: {C}")
    print(f"Output channels: {out_channels}")
    print()
    
    # 创建输入数据
    x = torch.randn(N, C, H, W).cuda()
    
    # 创建模型
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).cuda().eval()
    
    # 定义run_fn: 包含compile + 推理
    def run_fn():
        scripted_model = torch.compile(model, backend="inductor")
        out = scripted_model(x)
        return out
    
    # 测量compile + 推理的总时间
    result = measure_kernel_and_host(
        run_fn, 
        tag=f"inductor_{H}x{W}_k{kernel_size}"
    )
    

    print()
    print("=== Results ===")
    print()
    print("=== Performance Summary ===")
    print(f"Total kernel time: {result['kernel_ms']:.3f} ms")
    print(f"Host time:         {result['host_ms']:.3f} ms")