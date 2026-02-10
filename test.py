# coding:UTF-8
# @Author:RuHe
# @Time:2025/1/1 18:06
import torch

# 检测系统中是否有可用的GPU
print("检测系统中是否有可用的GPU:", torch.cuda.is_available())

if torch.cuda.is_available():
    # 输出可用的GPU设备数量
    print(f"GPU可用，可用的GPU设备数量：{torch.cuda.device_count()}")
    # 输出每个可用GPU设备的名称
    for i in range(torch.cuda.device_count()):
        print(f"GPU设备{i}: {torch.cuda.get_device_name(i)}")
