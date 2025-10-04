#!/bin/bash

# 无超时的虚拟机自动实验脚本
# 完全移除超时限制，让Inductor有足够时间编译

set -e  # 遇到错误立即退出

echo "🚀 Starting No-Timeout VM Experiments - $(date)"
echo "=============================================="

# 创建结果目录
mkdir -p results
mkdir -p plots

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="results/results_${TIMESTAMP}.txt"

echo "📁 Results will be saved to: $RESULTS_FILE"
echo ""

# 实验配置
ALL_INPUT_SIZES=(16 32 64 128)  # 所有输入尺寸，包括128×128
ALL_KERNEL_SIZES=(3 5 7)        # 所有卷积核尺寸
INDUCTOR_INPUT_SIZES=(16 32)    # Inductor只跑小尺寸
INDUCTOR_KERNEL_SIZES=(3 5)     # Inductor只跑小卷积核

# 运行PyTorch Baseline和JAX的所有配置
echo "🧪 Running PyTorch Baseline and JAX for all configurations..."
for input_size in "${ALL_INPUT_SIZES[@]}"; do
    for kernel_size in "${ALL_KERNEL_SIZES[@]}"; do
        echo "🧪 Testing input ${input_size}x${input_size}, kernel ${kernel_size}x${kernel_size}"
        echo "=================================================="
        
        # PyTorch Baseline - 无超时
        echo "Running PyTorch Baseline..."
        python3 gpu/myconv.py --input_size $input_size --kernel_size $kernel_size >> $RESULTS_FILE 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Baseline completed"
        else
            echo "⚠️  Baseline failed"
        fi
        
        # JAX - 无超时
        echo "Running JAX..."
        python3 gpu/myconv_jax.py --input_size $input_size --kernel_size $kernel_size >> $RESULTS_FILE 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ JAX completed"
        else
            echo "⚠️  JAX failed"
        fi
        
        echo ""
    done
done

# 运行PyTorch Inductor的小配置
echo "🧪 Running PyTorch Inductor for small configurations only..."
for input_size in "${INDUCTOR_INPUT_SIZES[@]}"; do
    for kernel_size in "${INDUCTOR_KERNEL_SIZES[@]}"; do
        echo "🧪 Testing Inductor input ${input_size}x${input_size}, kernel ${kernel_size}x${kernel_size}"
        echo "=================================================="
        
        # PyTorch Inductor - 无超时，让编译完成
        echo "Running PyTorch Inductor (no timeout - compilation may take time)..."
        python3 gpu/myconv_inductor.py --input_size $input_size --kernel_size $kernel_size >> $RESULTS_FILE 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Inductor completed"
        else
            echo "⚠️  Inductor failed"
        fi
        
        echo ""
    done
done

echo "🎯 All experiments completed!"
echo "📊 Generating plots..."

# 生成图表（如果脚本存在）
if [ -f "generate_walltime_plot.py" ]; then
    python3 generate_walltime_plot.py $RESULTS_FILE --output_dir plots
    echo "📈 Plots generated in plots/ directory"
else
    echo "⚠️  Plot generation script not found, skipping plots"
fi

echo ""
echo "💾 Saving results to Git..."

# Git操作
git add results/ plots/ gpu/*.py *.py *.sh
git commit -m "Extended VM Experiment Results - $TIMESTAMP

- PyTorch Baseline & JAX: all sizes (16,32,64,128) x all kernels (3,5,7) = 24 experiments
- PyTorch Inductor: small sizes (16,32) x small kernels (3,5) = 4 experiments  
- Total experiments: 28
- Timestamp: $TIMESTAMP
- Includes 128x128 for comprehensive analysis"

echo "✅ Results committed to Git"

echo ""
echo "🔄 Shutting down in 60 seconds..."
echo "Press Ctrl+C to cancel"

# 倒计时
for i in {60..1}; do
    echo -n "Shutting down in $i seconds...\r"
    sleep 1
done

echo ""
echo "🛑 Shutting down now!"
sudo shutdown -h now
