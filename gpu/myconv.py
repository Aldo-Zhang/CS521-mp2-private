import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity as Act
import argparse

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 

        # Precompute output size
        #out_size = (in_size + 2*padding - kernel_size) / stride + 1
        self.out_h = (H + 2 * padding - kernel_size) // stride + 1
        self.out_w = (W + 2 * padding - kernel_size) // stride + 1

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.
        
        # Create output tensor to store patches
        patches = torch.zeros(N, out_h * out_w, C * KH * KW, device=x.device, dtype=x.dtype)
        
        # Iterate over each output position and decide the corresponding input
        for i in range(out_h):
            for j in range(out_w):
                start_h = i * S
                start_w = j * S
                end_h = start_h + KH
                end_w = start_w + KW
                
                # Extract from the original X[N, C, H, W]
                window = x_pad[:, :, start_h:end_h, start_w:end_w]
                flat_window = window.reshape(N, C * KH * KW) 
                
                # Store the patches
                patches[:, i * out_w + j, :] = flat_window
        
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size
        C = self.in_channels

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        flat_weight = self.weight.reshape(C_out, C*KH*KW)

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        cols_reshaped = cols.reshape(-1, C*KH*KW)  # Reshape the cols to be a 2-d matrix
        R = cols_reshaped.shape[0]
        out_flat = torch.empty(R, C_out, device=x.device, dtype=x.dtype)
        RB = 128    # rows tile
        CB = 128    # cols tile
        for r0 in range(0, R, RB):
            r1 = min(r0 + RB, R)
            A = cols_reshaped[r0:r1, :]             # (rb, K)

            for c0 in range(0, C_out, CB):
                c1 = min(c0 + CB, C_out)
                B = flat_weight[c0:c1, :].T         # (K, cb)
                out_flat[r0:r1, c0:c1] = torch.mm(A, B)
        
        # TO DO: 4) Add bias.
        out_flat = out_flat + self.bias  # (N*out_h*out_w, C_out)

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out_flat.reshape(N, self.out_h * self.out_w, C_out)  
        out = out.permute(0, 2, 1) 
        out = out.reshape(N, C_out, self.out_h, self.out_w)

        return out

    def forward(self, x):
        return self.conv2d_manual(x)


def measure_kernel_and_host(run_fn, tag="run"):
    acts = [Act.CPU, Act.CUDA]

    with profile(activities=acts, record_shapes=False, profile_memory=False) as prof:
        torch.cuda.synchronize()
        with record_function(tag):
            run_fn()                      
        torch.cuda.synchronize()

    ka = prof.key_averages()

    def cuda_us(e, self_only=True):
        # PyTorch profiler returns time in microseconds (μs)
        us = None
        if hasattr(e, "self_cuda_time_total"):
            us = e.self_cuda_time_total
        elif hasattr(e, "cuda_time_total"):
            us = e.cuda_time_total
        elif hasattr(e, "device_time"):
            us = e.device_time
        return us or 0.0

    def cpu_us(e, self_only=True):
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
    
    # 详细分析GPU时间
    print(f"[DEBUG] Total events: {len(ka)}")
    print(f"[DEBUG] CUDA events with time > 0: {len([e for e in ka if cuda_us(e) > 0])}")
    print(f"[DEBUG] CPU events with time > 0: {len([e for e in ka if cpu_us(e) > 0])}")
    
    # 分析GPU时间组成
    cuda_events = [(e.key, cuda_us(e), cpu_us(e)) for e in ka if cuda_us(e) > 0]
    cuda_events.sort(key=lambda x: x[1], reverse=True)
    
    print(f"[DEBUG] GPU Time Analysis:")
    print(f"  Total GPU time: {total_kernel_us:.3f} μs")
    
    # 分类GPU操作
    kernel_ops = []
    memory_ops = []
    other_ops = []
    
    for key, cuda_time, cpu_time in cuda_events:
        if 'memcpy' in key.lower() or 'memset' in key.lower():
            memory_ops.append((key, cuda_time, cpu_time))
        elif any(kw in key.lower() for kw in ['conv', 'gemm', 'add', 'mul', 'relu']):
            kernel_ops.append((key, cuda_time, cpu_time))
        else:
            other_ops.append((key, cuda_time, cpu_time))
    
    # 计算各类操作的时间
    kernel_time = sum(cuda_time for _, cuda_time, _ in kernel_ops)
    memory_time = sum(cuda_time for _, cuda_time, _ in memory_ops)
    other_time = sum(cuda_time for _, cuda_time, _ in other_ops)
    
    print(f"  Pure kernel operations: {kernel_time:.3f} μs ({kernel_time/total_kernel_us*100:.1f}%)")
    print(f"  Memory operations: {memory_time:.3f} μs ({memory_time/total_kernel_us*100:.1f}%)")
    print(f"  Other operations: {other_time:.3f} μs ({other_time/total_kernel_us*100:.1f}%)")
    
    # 显示详细的GPU操作
    print(f"[DEBUG] Detailed GPU Operations:")
    for key, cuda_time, cpu_time in cuda_events[:10]:
        print(f"  {key:40s} GPU: {cuda_time:8.3f} μs  CPU: {cpu_time:8.3f} μs")
    
    # 计算GPU wall time (包含所有GPU相关开销)
    gpu_wall_time = total_kernel_us
    gpu_kernel_time = kernel_time
    
    print(f"[DEBUG] GPU Time Breakdown:")
    print(f"  GPU Wall Time (total): {gpu_wall_time:.3f} μs")
    print(f"  GPU Kernel Time (pure): {gpu_kernel_time:.3f} μs")
    print(f"  GPU Overhead: {gpu_wall_time - gpu_kernel_time:.3f} μs ({((gpu_wall_time - gpu_kernel_time)/gpu_wall_time*100):.1f}%)")

    return {
        "kernel_ms": total_kernel_ms,
        "kernel_us": total_kernel_us,
        "host_ms": total_host_ms,
        "host_us": total_host_us,
        "gpu_wall_time_us": gpu_wall_time,
        "gpu_kernel_time_us": gpu_kernel_time,
        "gpu_overhead_us": gpu_wall_time - gpu_kernel_time,
        "memory_time_us": memory_time,
        "other_time_us": other_time,
        "prof": prof
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Baseline Performance Test')
    parser.add_argument('--input_size', type=int, required=True, 
                       help='Input size (H=W, e.g., 32 for 32x32)')
    parser.add_argument('--kernel_size', type=int, required=True,
                       help='Kernel size (e.g., 3, 5, 7)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--channels', type=int, default=4,
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
    
    print(f"=== PyTorch Baseline Performance Test ===")
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
    
    # 定义run_fn: 只包含推理
    def run_fn():
        out = model(x)
        return out
    
    # 测量推理时间
    result = measure_kernel_and_host(
        run_fn, 
        tag=f"baseline_{H}x{W}_k{kernel_size}"
    )
    
    # 正确性测试
    out = model(x)
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    shape_check = out.shape == conv_ref.shape
    correctness_check = torch.allclose(out, conv_ref, atol=1e-4)
    
    print()
    print("=== Results ===")
    print()
    print("=== Performance Summary ===")
    print(f"GPU Wall Time:    {result['gpu_wall_time_us']:.3f} μs ({result['gpu_wall_time_us']/1000:.3f} ms)")
    print(f"GPU Kernel Time:  {result['gpu_kernel_time_us']:.3f} μs ({result['gpu_kernel_time_us']/1000:.3f} ms)")
    print(f"GPU Overhead:     {result['gpu_overhead_us']:.3f} μs ({result['gpu_overhead_us']/1000:.3f} ms)")
    print(f"Memory Operations: {result['memory_time_us']:.3f} μs ({result['memory_time_us']/1000:.3f} ms)")
    print(f"Other Operations:  {result['other_time_us']:.3f} μs ({result['other_time_us']/1000:.3f} ms)")
    print(f"CPU Host Time:    {result['host_us']:.3f} μs ({result['host_ms']:.3f} ms)")
    print(f"Total Time:       {result['kernel_us'] + result['host_us']:.3f} μs ({(result['kernel_ms'] + result['host_ms']):.3f} ms)")
    print()
    print("=== GPU Time Analysis ===")
    overhead_percent = (result['gpu_overhead_us'] / result['gpu_wall_time_us']) * 100
    kernel_percent = (result['gpu_kernel_time_us'] / result['gpu_wall_time_us']) * 100
    memory_percent = (result['memory_time_us'] / result['gpu_wall_time_us']) * 100
    print(f"Kernel efficiency: {kernel_percent:.1f}% (pure computation)")
    print(f"Memory overhead:   {memory_percent:.1f}% (data transfer)")
    print(f"Other overhead:    {overhead_percent - memory_percent:.1f}% (synchronization, etc.)")
    print()
    print("=== Correctness Check ===")
    print(f"Shape check: {shape_check}")
    print(f"Correctness check: {correctness_check}")
