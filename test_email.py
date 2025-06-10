import json
import smtplib
from email.mime.text import MIMEText

def test_email_config(config_path='gpu_monitor_config.json'):
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        # 准备邮件内容
        msg = MIMEText("This is a test email from GPU monitor")
        msg['Subject'] = 'GPU Monitor Test Email'
        msg['From'] = config["email"]["sender"]
        msg['To'] = config["email"]["receiver"]

        # 连接邮件服务器
        if config["email"]["smtp_port"] == 465:
            server = smtplib.SMTP_SSL(config["email"]["smtp_server"], 
                                    config["email"]["smtp_port"])
        else:
            server = smtplib.SMTP(config["email"]["smtp_server"], 
                                config["email"]["smtp_port"])
            server.starttls()

        # 登录并发送
        server.login(config["email"]["sender"], config["email"]["password"])
        server.send_message(msg)
        server.quit()
        
        print("Test email sent successfully!")
        
    except Exception as e:
        print(f"Failed to send test email: {str(e)}")

if __name__ == "__main__":
    test_email_config()