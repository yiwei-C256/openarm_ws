import mujoco
import mujoco.viewer
import time
import math

model_path = '/home/yhzzz/openarm_ws/openarm_mujoco/v1/openarm_bimanual.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

# 打印执行器
print("=== 模型执行器列表 ===")
for i in range(model.nu):  # nu 是执行器数量
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i}: {name}")
print("======================")

# 打印相机
print("=== 模型相机列表 ===")
for i in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"{i}: {name}")
print("======================")

# 选择第一个执行器做示例
aid = 0
act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)

# 启动 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 尝试切换到你挂的深度相机
    cam_name_list = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    if 'body_depth_camera' in cam_name_list:
        cam_id = cam_name_list.index('body_depth_camera')
        viewer.cam.fixedcamid = cam_id
        print(f"切换到相机视角: 'body_depth_camera' (id={cam_id})")
    else:
        print("未找到名为 'body_depth_camera' 的相机，使用默认视角。")

    step = 0
    while viewer.is_running():
        data.ctrl[aid] = 1.5 * math.sin(0.002 * step)
        mujoco.mj_step(model, data)
        print(f"step={step:4d}  {act_name} ctrl={data.ctrl[aid]:+.2f}  qpos={data.qpos[0]:+.2f}")
        viewer.sync()
        step += 1
        time.sleep(0.002)
