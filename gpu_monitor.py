import subprocess
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import json
import os

class GPUMonitor:
    def __init__(self, config_path='gpu_monitor_config.json'):
        self.config = self.load_config(config_path)
        # 记录每个GPU的状态，避免重复通知
        self.gpu_status = {0: False, 1: False, 2: False, 3: False}  # False表示正在使用
        
    def load_config(self, config_path):
        """加载配置文件"""
        if not os.path.exists(config_path):
            default_config = {
                "email": {
                    "sender": "your_email@example.com",
                    "password": "your_app_password",
                    "receiver": "your_email@example.com",
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587
                },
                "check_interval": 60  # 检查间隔（秒）
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created default config at {config_path}")
            print("Please update the config with your email credentials")
            exit(1)
            
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_gpu_status(self):
        """获取GPU使用情况"""
        try:
            # 使用nvidia-smi命令获取GPU信息
            cmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode('utf-8').strip().split('\n')
            
            gpu_info = {}
            for line in output:
                index, util, mem_used = map(float, line.split(', '))
                # 当GPU利用率和显存使用都较低时认为是空闲的
                is_idle = util < 15 and mem_used < 200
                gpu_info[int(index)] = is_idle
                
            return gpu_info
            
        except Exception as e:
            print(f"Error getting GPU status: {str(e)}")
            return None

    def send_email(self, gpu_id):
        """发送邮件通知"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subject = f"Server 202 GPU {gpu_id} Available!"
            message = f"GPU {gpu_id} is now available at {current_time}"
            
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.config["email"]["sender"]
            msg['To'] = self.config["email"]["receiver"]

            # 根据端口号选择不同的连接方式
            if self.config["email"]["smtp_port"] == 465:
                # 使用SSL连接
                server = smtplib.SMTP_SSL(self.config["email"]["smtp_server"], 
                                        self.config["email"]["smtp_port"])
            else:
                # 使用普通连接
                server = smtplib.SMTP(self.config["email"]["smtp_server"], 
                                    self.config["email"]["smtp_port"])
                server.starttls()  # 启用TLS加密

            server.login(self.config["email"]["sender"], 
                        self.config["email"]["password"])
            server.send_message(msg)
            server.quit()
            
            print(f"Notification sent for GPU {gpu_id}")
            return True
            
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

    def monitor(self):
        """监控主循环"""
        print("GPU monitor started...")
        print("Monitoring GPUs 0, 1, 2, 3...")
        print(f"Check interval: {self.config['check_interval']} seconds")
        
        while True:
            try:
                current_status = self.get_gpu_status()
                if current_status:
                    for gpu_id in range(4):
                        # 如果GPU从使用变为空闲，发送通知
                        if current_status[gpu_id] and not self.gpu_status[gpu_id]:
                            self.send_email(gpu_id)
                            self.gpu_status[gpu_id] = True
                        # 更新GPU状态
                        elif not current_status[gpu_id]:
                            self.gpu_status[gpu_id] = False
                
                time.sleep(self.config["check_interval"])
                
            except KeyboardInterrupt:
                print("\nGPU monitor stopped.")
                break
            except Exception as e:
                print(f"Error in monitor loop: {str(e)}")
                time.sleep(self.config["check_interval"])

if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.monitor()

# nohup python gpu_monitor.py > gpu_monitor.log 2>&1 &
# 27574