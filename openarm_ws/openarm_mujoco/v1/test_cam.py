import mujoco
import cv2
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# 摄像头配置
CAMERA_NAME = "d435"  # 可切换为front_camera/side_camera
WIDTH, HEIGHT = 640, 480

# 创建渲染器（新版Mujoco原生支持离屏渲染）
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# 主循环
cv2.namedWindow(f"Camera: {CAMERA_NAME}", cv2.WINDOW_NORMAL)
cv2.resizeWindow(f"Camera: {CAMERA_NAME}", WIDTH, HEIGHT)

try:
    while True:
        # 运行仿真步
        mujoco.mj_step(model, data)
        
        # 渲染摄像头画面（新版API极简）
        renderer.update_scene(data, camera=CAMERA_NAME)
        img = renderer.render()
        
        # 转换格式并显示
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"Camera: {CAMERA_NAME}", img)
        
        # ESC/q退出
        if cv2.waitKey(1) in [27, 113]:
            break
finally:
    cv2.destroyAllWindows()