import mujoco
import mujoco.viewer as viewer
import cv2
import numpy as np
import time

# ===================== 核心优化配置（关键！可根据硬件调整） =====================
# 1. 摄像头分辨率（越低越流畅）
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480
# 2. 显示更新间隔（每N步仿真更新一次显示）
DISPLAY_UPDATE_STEP = 10  # 可改20更流畅
# 3. 摄像头名称
CAMERA_NAME = "d435"
# 4. 禁用3D查看器（优先设为True提升帧率，避免GLFW警告）
DISABLE_VIEWER = False

# ===================== 初始化（适配旧版本Mujoco/OpenCV） =====================
# 加载模型
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# 旧版本Mujoco兼容：仅保留分辨率配置（报错则跳过）
try:
    model.vis.global_.offwidth = CAMERA_WIDTH
    model.vis.global_.offheight = CAMERA_HEIGHT
except AttributeError:
    print("注意：旧版本Mujoco不支持修改全局渲染分辨率，已跳过该优化")

# 创建轻量化渲染器（旧版本兼容）
renderer = mujoco.Renderer(
    model, 
    height=CAMERA_HEIGHT, 
    width=CAMERA_WIDTH
)

# 创建摄像头窗口（极致轻量化）
cv2.namedWindow(f"Camera: {CAMERA_NAME}", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.resizeWindow(f"Camera: {CAMERA_NAME}", CAMERA_WIDTH, CAMERA_HEIGHT)

# ===================== 单线程核心逻辑（稳定无错） =====================
def main():
    global DISABLE_VIEWER
    step_counter = 0  # 步数计数器（控制显示更新）
    viewer_instance = None
    
    # 启动3D查看器（可选禁用，避免GLFW警告）
    if not DISABLE_VIEWER:
        try:
            # 包裹GLFW初始化逻辑，避免警告
            viewer_instance = viewer.launch_passive(model, data)
        except Exception as e:
            print(f"3D查看器启动失败（已自动禁用）：{e}")
            DISABLE_VIEWER = True
    else:
        print("已禁用3D查看器，仅显示摄像头画面（流畅度最优）")

    try:
        while True:
            # 1. 执行仿真步（高频运行，保证物理精度）
            mujoco.mj_step(model, data)
            step_counter += 1

            # 2. 仅每N步更新一次显示（核心优化）
            if step_counter % DISPLAY_UPDATE_STEP == 0:
                # -------------------- 渲染摄像头（修复cvtColor参数错误） --------------------
                try:
                    renderer.update_scene(data, camera=CAMERA_NAME)
                    img = renderer.render()
                    # 关键修复：移除无效的dtype参数，仅保留必要参数
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Camera: {CAMERA_NAME}", img)
                except Exception as e:
                    print(f"摄像头渲染异常: {e}")
                    # 仅打印异常，不退出（避免程序崩溃）
                    continue

                # -------------------- 同步3D查看器（处理GLFW警告） --------------------
                if not DISABLE_VIEWER and viewer_instance:
                    try:
                        if viewer_instance.is_running():
                            viewer_instance.sync()
                    except Exception as e:
                        print(f"3D查看器同步警告: {e}")
                        DISABLE_VIEWER = True

                # -------------------- 非阻塞按键检测 --------------------
                key = cv2.waitKey(1)
                if key in [27, 113]:  # ESC/q退出
                    print("用户退出")
                    break

            # 3. 检查查看器状态（避免空跑）
            if not DISABLE_VIEWER and viewer_instance and not viewer_instance.is_running():
                print("3D查看器已关闭")
                break

    except KeyboardInterrupt:
        print("\n程序中断")
    finally:
        # 清理资源（优化GLFW关闭逻辑）
        cv2.destroyAllWindows()
        if not DISABLE_VIEWER and viewer_instance:
            try:
                viewer_instance.close()
            except:
                pass  # 忽略GLFW关闭时的警告
        print("程序正常退出")

# ===================== 启动（服务器必加xvfb优化） =====================
if __name__ == "__main__":
    # 服务器运行命令（复制执行）：
    # xvfb-run -s "-screen 0 640x480x24 -render-accel -noreset -nolisten tcp -shmem" python3 test_mujoco.py
    main()