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

    def cuda_us(e, self_only=True):
        # PyTorch profiler returns time in microseconds (μs)
        # Try different attribute names
        us = None
        if hasattr(e, "self_cuda_time_total"):
            us = e.self_cuda_time_total
        elif hasattr(e, "cuda_time_total"):
            us = e.cuda_time_total
        elif hasattr(e, "cuda_time"):
            us = e.cuda_time
        return us or 0.0

    def cpu_us(e, self_only=True):
        # PyTorch profiler returns time in microseconds (μs)
        us = None
        if hasattr(e, "self_cpu_time_total"):
            us = e.self_cpu_time_total
        elif hasattr(e, "cpu_time_total"):
            us = e.cpu_time_total
        elif hasattr(e, "cpu_time"):
            us = e.cpu_time
        return us or 0.0

    # 计算总时间（微秒）
    total_kernel_us = sum(cuda_us(e, self_only=True) for e in ka)
    total_host_us   = sum(cpu_us(e,  self_only=True) for e in ka)
    
    # 转换为毫秒
    total_kernel_ms = total_kernel_us / 1000.0
    total_host_ms = total_host_us / 1000.0
    
    # 添加详细调试信息
    print(f"[DEBUG] Total events: {len(ka)}")
    print(f"[DEBUG] CUDA events with time > 0: {len([e for e in ka if cuda_us(e) > 0])}")
    print(f"[DEBUG] CPU events with time > 0: {len([e for e in ka if cpu_us(e) > 0])}")
    
    # 显示所有事件的基本信息
    print(f"[DEBUG] Event details:")
    for i, e in enumerate(ka[:10]):  # 只显示前10个事件
        cuda_time = cuda_us(e)
        cpu_time = cpu_us(e)
        print(f"  Event {i}: {e.key}")
        print(f"    CUDA time: {cuda_time:.3f} μs")
        print(f"    CPU time:  {cpu_time:.3f} μs")
        print(f"    Count:     {getattr(e, 'count', 'N/A')}")
    
    # 显示前几个最耗时的CUDA操作
    cuda_events = [(e.key, cuda_us(e)) for e in ka if cuda_us(e) > 0]
    cuda_events.sort(key=lambda x: x[1], reverse=True)
    print(f"[DEBUG] Top CUDA operations (μs):")
    for key, time_us in cuda_events[:5]:
        print(f"  {key}: {time_us:.3f} μs")

    return {"kernel_ms": total_kernel_ms,
            "kernel_us": total_kernel_us,
            "host_ms": total_host_ms,
            "host_us": total_host_us,
            "prof": prof}

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
    print(f"Total kernel time: {result['kernel_us']:.3f} μs ({result['kernel_ms']:.3f} ms) - CUDA operations")
    print(f"Host time:         {result['host_us']:.3f} μs ({result['host_ms']:.3f} ms) - CPU operations")
    print(f"Total time:        {result['kernel_us'] + result['host_us']:.3f} μs ({(result['kernel_ms'] + result['host_ms']):.3f} ms)")