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

# 实验配置 - 减少配置以避免超时
INPUT_SIZES=(32 64)  # 减少输入尺寸
KERNEL_SIZES=(3 5)   # 减少卷积核尺寸

# 运行所有实验
for input_size in "${INPUT_SIZES[@]}"; do
    for kernel_size in "${KERNEL_SIZES[@]}"; do
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
        
        # PyTorch Inductor - 无超时，让编译完成
        echo "Running PyTorch Inductor (no timeout - compilation may take time)..."
        python3 gpu/myconv_inductor.py --input_size $input_size --kernel_size $kernel_size >> $RESULTS_FILE 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Inductor completed"
        else
            echo "⚠️  Inductor failed"
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
git commit -m "No-Timeout VM Experiment Results - $TIMESTAMP

- Input sizes: ${INPUT_SIZES[*]}
- Kernel sizes: ${KERNEL_SIZES[*]}
- Total experiments: $((${#INPUT_SIZES[@]} * ${#KERNEL_SIZES[@]} * 3))
- Timestamp: $TIMESTAMP
- No timeout mode: unlimited compilation time"

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
