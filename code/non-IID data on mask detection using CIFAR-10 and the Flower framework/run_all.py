import subprocess
import time
import os

def run_and_log(command, logfile):
    """启动子进程并将输出重定向到日志文件"""
    f = open(logfile, "w", encoding="utf-8")
    return subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, text=True)

# ✅ 清理旧日志
if not os.path.exists("logs"):
    os.makedirs("logs")

# 启动服务器
print("🚀 启动联邦服务器...")
server_process = run_and_log(["python", "server.py"], "logs/server.log")
time.sleep(3)  # 等服务器初始化

# 启动 5 个客户端
client_processes = []
for i in range(5):
    print(f"🚀 启动客户端 {i}")
    p = run_and_log(["python", "client.py", str(i)], f"logs/client_{i}.log")
    client_processes.append((i, p))
    time.sleep(1)  # 避免端口冲突

# 等待所有客户端完成
for i, p in client_processes:
    try:
        p.wait(timeout=3600)
        print(f"🎉 客户端 {i} 完成")
    except subprocess.TimeoutExpired:
        print(f"❌ 客户端 {i} 超时，强制终止")
        p.terminate()

# 等服务器完成
try:
    server_process.wait(timeout=3600)
    print("🎉 服务器完成")
except subprocess.TimeoutExpired:
    print("❌ 服务器超时，强制终止")
    server_process.terminate()

print("✅ 所有进程完成。请检查 logs/ 目录下的输出日志。")
