#!/bin/bash

# 实验配置
INPUT_SIZES=(32 64 128)
KERNEL_SIZES=(3 5 7)

# 创建结果目录
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results/experiment_results_${TIMESTAMP}.txt"

echo "=== PyTorch Inductor Performance Experiments ===" > $RESULT_FILE
echo "Start time: $(date)" >> $RESULT_FILE
echo "================================================" >> $RESULT_FILE
echo "" >> $RESULT_FILE

# 计数器
test_count=0
total_tests=$((${#INPUT_SIZES[@]} * ${#KERNEL_SIZES[@]}))

echo "Starting $total_tests experiments..."
echo "Results will be saved to: $RESULT_FILE"

# 运行所有实验
for input_size in "${INPUT_SIZES[@]}"; do
    for kernel_size in "${KERNEL_SIZES[@]}"; do
        test_count=$((test_count + 1))
        
        echo "" >> $RESULT_FILE
        echo "================================================" >> $RESULT_FILE
        echo "Test $test_count/$total_tests: Input size ${input_size}x${input_size}, Kernel size ${kernel_size}x${kernel_size}" >> $RESULT_FILE
        echo "Start time: $(date)" >> $RESULT_FILE
        echo "================================================" >> $RESULT_FILE
        
        echo "Running Test $test_count/$total_tests: Input size ${input_size}x${input_size}, Kernel size ${kernel_size}x${kernel_size}"
        
        # 运行实验并捕获输出
        python3 gpu/myconv_inductor.py --input_size $input_size --kernel_size $kernel_size >> $RESULT_FILE 2>&1
        
        # 检查实验是否成功
        if [ $? -eq 0 ]; then
            echo "Test $test_count completed successfully" >> $RESULT_FILE
        else
            echo "Test $test_count FAILED with exit code $?" >> $RESULT_FILE
        fi
        
        echo "End time: $(date)" >> $RESULT_FILE
        echo "" >> $RESULT_FILE
        
        # 显示进度
        echo "Completed $test_count/$total_tests tests"
    done
done

# 添加总结信息
echo "" >> $RESULT_FILE
echo "================================================" >> $RESULT_FILE
echo "EXPERIMENT SUMMARY" >> $RESULT_FILE
echo "================================================" >> $RESULT_FILE
echo "Total tests completed: $test_count/$total_tests" >> $RESULT_FILE
echo "End time: $(date)" >> $RESULT_FILE
echo "Result file: $RESULT_FILE" >> $RESULT_FILE

echo ""
echo "All experiments completed!"
echo "Results saved to: $RESULT_FILE"

# Git操作
echo "Committing results to git..."
git add $RESULT_FILE
git add results/
git commit -m "Add experiment results from $(date '+%Y-%m-%d %H:%M:%S') - $test_count tests completed"

if [ $? -eq 0 ]; then
    echo "Git commit successful"
else
    echo "Git commit failed"
fi

# 显示结果文件内容
echo ""
echo "=== Final Results ==="
cat $RESULT_FILE

# 关闭AWS实例
echo ""
echo "Shutting down AWS instance..."
sudo shutdown -h now
