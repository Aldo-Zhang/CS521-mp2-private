import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler

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

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(out_jax, conv_ref, atol=1e-1))
