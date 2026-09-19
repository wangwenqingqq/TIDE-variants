import torch
import os

# 模拟设置环境变量（如果你在命令行设置了，这行可以注释掉）
# 确保屏蔽掉坏的 0 号卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def test_gpu():
    if not torch.cuda.is_available():
        print("❌ 错误: CUDA 不可用")
        return

    device_count = torch.cuda.device_count()
    print(f"✅ 检测到逻辑设备数量: {device_count} (预期应为 3)")

    for i in range(device_count):
        print("-" * 40)
        try:
            # 指定设备
            device = torch.device(f"cuda:{i}")
            
            # 获取物理信息
            props = torch.cuda.get_device_properties(i)
            print(f"正在测试逻辑设备 cuda:{i}")
            print(f"  -> 显卡型号: {props.name}")
            print(f"  -> 显存大小: {props.total_memory / 1024**3:.2f} GB")
            
            # 1. 测试显存分配
            x = torch.randn(10000, 10000, device=device)
            y = torch.randn(10000, 10000, device=device)
            print("  -> 显存分配: 成功")
            
            # 2. 测试计算 (矩阵乘法)
            # H100 跑这个应该瞬间完成
            z = torch.matmul(x, y)
            # 强制同步，确保计算真的完成了
            torch.cuda.synchronize()
            print("  -> 矩阵计算: 成功")
            
            print(f"✅ 逻辑设备 cuda:{i} (物理卡 1,2,3 之一) 测试通过")
            
            # 清理显存
            del x, y, z
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 逻辑设备 cuda:{i} 测试失败!")
            print(f"错误信息: {e}")

if __name__ == "__main__":
    test_gpu()