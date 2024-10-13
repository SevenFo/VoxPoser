import base64
import numpy as np
import requests
from PIL import Image

# 假设 vlm_server.frame_shape 是已知的
frame_shape = (3, 480, 480)  # 示例形状，请根据实际情况调整

def read_image_as_numpy(image_path):
    # 读取图像并转换为指定大小
    img = Image.open(image_path).resize((frame_shape[2], frame_shape[1]))
    img_np = np.array(img.convert('RGB'))
    
    # 转换为 CHW 格式
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

def send_process_sole_frame_request(url, labels, data_array):
    # 将 NumPy 数组转换为字节数据
    data_bytes = data_array.tobytes()
    
    # 将字节数据编码为 base64
    data_base64 = base64.b64encode(data_bytes).decode('utf-8')
    
    # 构建请求数据
    data = {
        "labels": labels,
        "data": data_base64,
        "shape": data_array.shape
    }
    
    # 发送 POST 请求
    response = requests.post(url, json=data)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析响应 JSON 数据
        result = response.json()
        return np.array(result)
    else:
        print(f"Request failed with status code {response.status_code}")
        return None

def visualize_mask_on_image(image_path, mask, output_path):
    # 读取原始图像
    img = Image.open(image_path).resize((frame_shape[2], frame_shape[1]))  # 调整图像大小 (宽, 高)
    
    # 检查 mask 的形状
    if mask.shape[0] == 1:
        # 如果 mask 是单通道的，去掉通道维度
        mask = mask[0]

    # 确保 mask 是二维的
    if len(mask.shape) != 2:
        raise ValueError("Mask should be a 2D array")

    # 创建一个透明的 RGBA 图像
    overlay = Image.fromarray((mask * 255).astype(np.uint8), mode='L').convert("RGBA")
    
    # 合并原图和 mask
    img = img.convert("RGBA")
    combined = Image.blend(img, overlay, alpha=0.5)
    
    # 保存结果
    combined.save(output_path)

# 示例调用
url = "http://127.0.0.1:5000/process_sole_frame"
labels = ["dog", "cat"]
image_path = "/shared/codes/VoxPoser.worktrees/VoxPoser/pexels-photo-1490908.jpg"
output_path = "/shared/codes/VoxPoser.worktrees/VoxPoser/combined_image.png"

# 从文件读取图像并转换为 NumPy 数组
data_array = read_image_as_numpy(image_path)

# 发送请求并获取结果
result = send_process_sole_frame_request(url, labels, data_array)

if result is not None:
    print("Processing result:", result)
    visualize_mask_on_image(image_path, result, output_path)
    print(f"Combined image saved to {output_path}")
else:
    print("Failed to process image")
