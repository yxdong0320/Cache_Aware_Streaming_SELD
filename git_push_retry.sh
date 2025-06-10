#!/bin/bash

# git_push_retry.sh
# 循环推送脚本，直到成功为止

# chmod +x git_push_retry.sh

# ./git_push_retry.sh

echo "开始尝试推送到GitHub..."
attempt=1
max_attempts=100  # 最大尝试次数，避免无限循环

while [ $attempt -le $max_attempts ]; do
    echo "第 $attempt 次尝试推送..."
    
    # 尝试推送
    if git push; then
        echo "✅ 推送成功！"
        exit 0
    else
        echo "❌ 第 $attempt 次推送失败"
        
        # 等待一段时间后重试
        sleep_time=$((attempt * 2))  # 递增等待时间
        if [ $sleep_time -gt 30 ]; then
            sleep_time=30  # 最大等待30秒
        fi
        
        echo "等待 $sleep_time 秒后重试..."
        sleep $sleep_time
        
        attempt=$((attempt + 1))
    fi
done

echo "❌ 达到最大尝试次数 ($max_attempts)，推送失败"
exit 1