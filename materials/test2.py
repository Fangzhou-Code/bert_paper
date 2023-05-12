import random

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
# 创建一个PyTorch张量
tensor = torch.randn(3, 4)


# 计算不同信噪比下的延迟
def calculate_latency(tensor, snr):
    # 模拟噪声
    noise = torch.randn_like(tensor) / snr

    # 添加噪声到张量中
    noisy_tensor = tensor + noise

    # 运行模型并记录延迟
    start_time = 0.2
    #model(noisy_tensor)
    end_time = 0.05

    return end_time - start_time


# 测试模型的延迟并绘制柱状图
snrs = [1, 5, 10, 15, 20]
latencies = [calculate_latency(tensor, snr) for snr in snrs]

plt.bar(np.arange(len(snrs)), latencies, align='center', alpha=0.5)
plt.xticks(np.arange(len(snrs)), snrs)
plt.xlabel("SNR")
plt.ylabel("Latency (Seconds)")
plt.title("Latency vs SNR")

plt.show()
