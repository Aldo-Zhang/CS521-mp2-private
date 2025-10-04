import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler
import argparse
import time
import os

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.
    
    patches_list = []
    for i in range(out_h):
        for j in range(out_w):
            sh, sw = i * S, j * S
            window = x_pad[:, :, sh:sh + KH, sw:sw + KW]  # (N, C, KH, KW)
            patches_list.append(window.reshape(N, C * KH * KW))  # (N, CK2)

    patches = jnp.stack(patches_list, axis=1)  # (N, out_h*out_w, C*KH*KW)
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    S, P = stride, padding
    out_h = (H + 2 * P - KH) // S + 1
    out_w = (W + 2 * P - KW) // S + 1
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)
    K = C * KH * KW
    R = out_h * out_w
    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    flat_weight = weight.reshape(C_out, K)

    # TO DO: 3) perform tiled matmul after required reshaping is done.
    cols_reshaped = cols.reshape(N * R, K)                            
    out_flat = jnp.zeros((N * R, C_out), dtype=cols_reshaped.dtype)

    RB = 2048   # row tile
    CB = 256    # col tile

    for r0 in range(0, N * R, RB):
        r1 = min(r0 + RB, N * R)
        A = cols_reshaped[r0:r1, :]                                   
        for c0 in range(0, C_out, CB):
            c1 = min(c0 + CB, C_out)
            B = flat_weight[c0:c1, :].T                               
            tile = A @ B                                             
            out_flat = out_flat.at[r0:r1, c0:c1].set(tile)

    # TO DO: 4) Add bias.
    out_flat = out_flat + bias                                        

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = out_flat.reshape(N, R, C_out)                          
    out = jnp.transpose(out, (0, 2, 1))                              
    out = out.reshape(N, C_out, out_h, out_w)                         
    return out

def measure_jax_performance(run_fn, tag="jax_run"):
    """测量JAX性能 - 使用wall time"""
    print(f"[{tag}] Measuring JAX performance...")
    
    # 使用wall time测量总时间
    start_time = time.time()
    result = run_fn()
    end_time = time.time()
    
    total_time_us = (end_time - start_time) * 1_000_000  # 转换为微秒
    total_time_ms = total_time_us / 1000.0
    
    print(f"[{tag}] Total wall time: {total_time_us:.3f} μs ({total_time_ms:.3f} ms)")
    
    return {
        "total_time_us": total_time_us,
        "total_time_ms": total_time_ms,
        "result": result
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JAX Performance Test')
    parser.add_argument('--input_size', type=int, required=True, 
                       help='Input size (H=W, e.g., 32 for 32x32)')
    parser.add_argument('--kernel_size', type=int, required=True,
                       help='Kernel size (e.g., 3, 5, 7)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--channels', type=int, default=3,
                       help='Input channels (default: 3)')
    parser.add_argument('--out_channels', type=int, default=8,
                       help='Output channels (default: 8)')
    
    args = parser.parse_args()
    
    # 尝试使用GPU，如果失败则回退到CPU
    try:
        # 检查JAX GPU可用性
        gpu_devices = jax.devices('gpu')
        if len(gpu_devices) > 0:
            print(f"Using JAX GPU backend: {gpu_devices}")
            # 确保数据在GPU上
            jax.config.update('jax_platform_name', 'gpu')
        else:
            raise RuntimeError("No GPU devices available")
    except Exception as e:
        print(f"GPU not available ({e}), falling back to CPU")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    H = W = args.input_size
    kernel_size = args.kernel_size
    N = args.batch_size
    C = args.channels
    out_channels = args.out_channels
    
    print(f"=== JAX Performance Test ===")
    print(f"Input size: {H}x{W}")
    print(f"Kernel size: {kernel_size}")
    print(f"Batch size: {N}")
    print(f"Input channels: {C}")
    print(f"Output channels: {out_channels}")
    print(f"JAX platform: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()
    
    # 创建PyTorch模型作为参考
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1)
    model.eval()

    # 创建输入数据
    x_torch = torch.randn(N, C, H, W)

    # 导出权重和偏置
    params = {
        "weight": model.weight.detach().cpu().numpy(),
        "bias": model.bias.detach().cpu().numpy()
    }

    # 转换为JAX数组
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # 启用JIT编译
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # 定义运行函数
    def run_jax():
        out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
        out_jax.block_until_ready()
        return out_jax

    # 测量性能
    result = measure_jax_performance(run_jax, f"jax_{H}x{W}_k{kernel_size}")

    # 正确性测试
    out_jax = result["result"]
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    out_jax_torch = torch.from_numpy(np.array(out_jax)).to(conv_ref.dtype)
    shape_check = out_jax_torch.shape == conv_ref.shape
    correctness_check = torch.allclose(out_jax_torch, conv_ref, atol=1e-1)
    
    print()
    print("=== Results ===")
    print()
    print("=== Performance Summary ===")
    print(f"Total wall time: {result['total_time_us']:.3f} μs ({result['total_time_ms']:.3f} ms)")
    print()
    print("=== Correctness Check ===")
    print(f"Shape check: {shape_check}")
    print(f"Correctness check: {correctness_check}")
