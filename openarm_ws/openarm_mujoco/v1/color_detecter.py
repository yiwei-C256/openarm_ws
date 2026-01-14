import cv2
import numpy as np

# ===================== 固定图片路径 =====================
IMAGE_PATH = "/home/hxzzz/ros2_ws/v1/meshes/011_banana/texture_map.png"

def analyze_image_colors():
    """
    自动分析整张图片的颜色，输出香蕉（黄色）的HSV范围，无需手动框选
    """
    # 加载图片
    print(f"🔍 正在加载图片：{IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"\n❌ 无法加载图片！请检查：")
        print(f"1. 路径是否正确：{IMAGE_PATH}")
        print(f"2. 图片是否存在/未损坏（格式是否为png/jpg）")
        return
    
    # 转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # ===================== 核心：提取香蕉（黄色）的HSV范围 =====================
    # 第一步：筛选黄色像素（H通道在10-40之间，这是香蕉黄色的核心范围）
    # 构建黄色像素的掩码
    h_channel = img_hsv[:, :, 0]  # 色相通道
    yellow_mask = np.logical_and(h_channel >= 10, h_channel <= 40)
    
    # 提取所有黄色像素的HSV值
    yellow_pixels = img_hsv[yellow_mask]
    if len(yellow_pixels) == 0:
        print("\n⚠️ 未检测到黄色像素！可能图片不是香蕉纹理图，或调整H通道范围：")
        print("建议尝试 H范围：5-45 或 0-50")
        # 兜底：输出整张图片的HSV范围
        all_min_hsv = np.min(img_hsv, axis=(0, 1)).astype(np.int32)
        all_max_hsv = np.max(img_hsv, axis=(0, 1)).astype(np.int32)
        print(f"\n整张图片的HSV范围：")
        print(f"lower_banana = np.array({all_min_hsv.tolist()})")
        print(f"upper_banana = np.array({all_max_hsv.tolist()})")
        return
    
    # 第二步：计算黄色像素的HSV最小/最大/平均值（精准匹配香蕉颜色）
    min_hsv = np.min(yellow_pixels, axis=0).astype(np.int32)
    max_hsv = np.max(yellow_pixels, axis=0).astype(np.int32)
    avg_hsv = np.mean(yellow_pixels, axis=0).astype(np.int32)
    
    # 第三步：输出可直接复制的阈值（增加5%容错，避免检测漏检）
    # 容错调整：H±2，S±10，V±10（保证覆盖香蕉所有黄色调）
    lower_hsv = [
        max(0, min_hsv[0] - 2),    # H通道，最小0
        max(0, min_hsv[1] - 10),   # S通道，最小0
        max(0, min_hsv[2] - 10)    # V通道，最小0
    ]
    upper_hsv = [
        min(180, max_hsv[0] + 2),  # H通道，最大180
        min(255, max_hsv[1] + 10), # S通道，最大255
        min(255, max_hsv[2] + 10)  # V通道，最大255
    ]
    
    # ===================== 输出结果 =====================
    print("\n✅ 香蕉颜色分析完成！")
    print("\n========== 核心颜色参数（可直接复制） ==========")
    print(f"# 香蕉黄色像素的精准HSV范围（带容错）")
    print(f"lower_banana = np.array({lower_hsv})")
    print(f"upper_banana = np.array({upper_hsv})")
    print("\n# 详细参考值")
    print(f"黄色像素平均HSV值：{avg_hsv}")
    print(f"黄色像素原始最小HSV：{min_hsv}")
    print(f"黄色像素原始最大HSV：{max_hsv}")
    print("==================================================")

if __name__ == "__main__":
    # 直接运行，无交互、无窗口
    analyze_image_colors()
    print("\n👋 分析完成！请将上述阈值复制到手眼标定代码中。")