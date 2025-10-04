#!/bin/bash

# 实验配置
INPUT_SIZES=(32 64 128)
KERNEL_SIZES=(3 5 7)

# 创建结果目录
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results/experiment_results_${TIMESTAMP}.txt"
LOG_FILE="results/experiment_log_${TIMESTAMP}.txt"

# 日志函数
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

# 错误处理
handle_error() {
    log "ERROR: $1"
    echo "ERROR: $1" >> $RESULT_FILE
}

# 开始实验
log "=== PyTorch Inductor Performance Experiments ==="
echo "=== PyTorch Inductor Performance Experiments ===" > $RESULT_FILE
echo "Start time: $(date)" >> $RESULT_FILE
echo "================================================" >> $RESULT_FILE
echo "" >> $RESULT_FILE

# 计数器
test_count=0
total_tests=$((${#INPUT_SIZES[@]} * ${#KERNEL_SIZES[@]}))
successful_tests=0
failed_tests=0

log "Starting $total_tests experiments..."
log "Results will be saved to: $RESULT_FILE"
log "Log will be saved to: $LOG_FILE"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    handle_error "Python3 not found"
    exit 1
fi

# 检查CUDA环境
if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    handle_error "CUDA not available"
    exit 1
fi

log "Environment check passed"

# 运行所有实验
for input_size in "${INPUT_SIZES[@]}"; do
    for kernel_size in "${KERNEL_SIZES[@]}"; do
        test_count=$((test_count + 1))
        
        echo "" >> $RESULT_FILE
        echo "================================================" >> $RESULT_FILE
        echo "Test $test_count/$total_tests: Input size ${input_size}x${input_size}, Kernel size ${kernel_size}x${kernel_size}" >> $RESULT_FILE
        echo "Start time: $(date)" >> $RESULT_FILE
        echo "================================================" >> $RESULT_FILE
        
        log "Running Test $test_count/$total_tests: Input size ${input_size}x${input_size}, Kernel size ${kernel_size}x${kernel_size}"
        
        # 运行实验并捕获输出
        timeout 300 python3 gpu/myconv_inductor.py --input_size $input_size --kernel_size $kernel_size >> $RESULT_FILE 2>&1
        exit_code=$?
        
        # 检查实验是否成功
        if [ $exit_code -eq 0 ]; then
            log "Test $test_count completed successfully"
            echo "Test $test_count completed successfully" >> $RESULT_FILE
            successful_tests=$((successful_tests + 1))
        elif [ $exit_code -eq 124 ]; then
            handle_error "Test $test_count TIMEOUT (5 minutes)"
            failed_tests=$((failed_tests + 1))
        else
            handle_error "Test $test_count FAILED with exit code $exit_code"
            failed_tests=$((failed_tests + 1))
        fi
        
        echo "End time: $(date)" >> $RESULT_FILE
        echo "" >> $RESULT_FILE
        
        # 显示进度
        log "Progress: $test_count/$total_tests tests completed (Success: $successful_tests, Failed: $failed_tests)"
        
        # 短暂休息
        sleep 2
    done
done

# 添加总结信息
echo "" >> $RESULT_FILE
echo "================================================" >> $RESULT_FILE
echo "EXPERIMENT SUMMARY" >> $RESULT_FILE
echo "================================================" >> $RESULT_FILE
echo "Total tests: $total_tests" >> $RESULT_FILE
echo "Successful: $successful_tests" >> $RESULT_FILE
echo "Failed: $failed_tests" >> $RESULT_FILE
echo "Success rate: $(( successful_tests * 100 / total_tests ))%" >> $RESULT_FILE
echo "End time: $(date)" >> $RESULT_FILE
echo "Result file: $RESULT_FILE" >> $RESULT_FILE
echo "Log file: $LOG_FILE" >> $RESULT_FILE

log "All experiments completed!"
log "Successful: $successful_tests, Failed: $failed_tests"

# Git操作
log "Committing results to git..."
git add $RESULT_FILE $LOG_FILE
git add results/
git commit -m "Add experiment results from $(date '+%Y-%m-%d %H:%M:%S') - $successful_tests/$total_tests tests successful"

if [ $? -eq 0 ]; then
    log "Git commit successful"
else
    log "Git commit failed"
fi

# 显示结果文件内容
echo ""
echo "=== Final Results ==="
cat $RESULT_FILE

# 等待用户确认
echo ""
echo "Press Enter to shutdown the instance, or Ctrl+C to cancel..."
read -r

# 关闭AWS实例
log "Shutting down AWS instance..."
sudo shutdown -h now
