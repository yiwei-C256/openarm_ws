import mujoco
import numpy as np

XML_PATH = "/home/yhzzz/openarm_ws/openarm_mujoco/v1/openarm_bimanual.xml"

# ⚠️ 把 handeye_groundtruth.py 打印出来的矩阵粘到这里
T_TCP_CAM = np.array([
    [-0.0001,  1.0000,  0.0000,  0.2000],
    [ 0.0000,  0.0000, -1.0000,  0.1580],
    [-1.0000, -0.0001, -0.0000, -0.4181],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
])

def T_from_pos_mat(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.array(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T

# 1. 加载模型
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

tcp_body_id = model.body("openarm_left_hand_tcp").id
board_body_id = model.body("calibration_board").id
camera_id = model.camera("body_depth_camera").id

# 2. 运行仿真
for _ in range(10):
    mujoco.mj_step(model, data)

# === 真值 world ===
p_world_true = data.xpos[board_body_id].copy()

# === Camera 中的观测 ===
T_world_cam = T_from_pos_mat(
    data.cam_xpos[camera_id],
    data.cam_xmat[camera_id]
)

T_cam_world = np.linalg.inv(T_world_cam)

T_world_board = T_from_pos_mat(
    data.xpos[board_body_id],
    data.xmat[board_body_id]
)

T_board_cam = T_cam_world @ T_world_board

# === TCP -> world ===
T_world_tcp = T_from_pos_mat(
    data.xpos[tcp_body_id],
    data.xmat[tcp_body_id]
)

# === ✅ 正确反算链路 ===
T_board_world_est = T_world_tcp @ T_TCP_CAM @ T_board_cam
p_world_est = T_board_world_est[:3, 3]

# === 误差 ===
err = np.linalg.norm(p_world_true - p_world_est)

print("真实 world:", p_world_true)
print("视觉反算:", p_world_est)
print(f"定位误差: {err:.6f} m")

