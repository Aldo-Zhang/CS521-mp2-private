#!/bin/bash

# 简化版实验脚本
echo "Starting PyTorch Inductor experiments..."

# 创建结果文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results_${TIMESTAMP}.txt"
mkdir -p results

# 实验配置
declare -a configs=(
    "32 3"
    "32 5" 
    "32 7"
    "64 3"
    "64 5"
    "64 7"
    "128 3"
    "128 5"
    "128 7"
)

echo "=== PyTorch Inductor Experiments ===" > $RESULT_FILE
echo "Start: $(date)" >> $RESULT_FILE
echo "" >> $RESULT_FILE

# 运行9个实验
for i in "${!configs[@]}"; do
    config=(${configs[$i]})
    input_size=${config[0]}
    kernel_size=${config[1]}
    
    echo "Test $((i+1))/9: ${input_size}x${input_size}, kernel ${kernel_size}x${kernel_size}"
    echo "Test $((i+1))/9: ${input_size}x${input_size}, kernel ${kernel_size}x${kernel_size}" >> $RESULT_FILE
    echo "----------------------------------------" >> $RESULT_FILE
    
    python3 gpu/myconv_inductor.py --input_size $input_size --kernel_size $kernel_size >> $RESULT_FILE 2>&1
    
    echo "Test $((i+1)) completed" >> $RESULT_FILE
    echo "" >> $RESULT_FILE
done

echo "End: $(date)" >> $RESULT_FILE
echo "All experiments completed!"

# Git提交
git add $RESULT_FILE
git commit -m "Experiment results $(date '+%Y-%m-%d %H:%M:%S')"

# 关闭实例
echo "Shutting down in 10 seconds..."
sleep 10
sudo shutdown -h now
