import mujoco
import numpy as np
import time

XML_PATH = "/home/elbe/openarm_ws/openarm_mujoco/v1/openarm_bimanual.xml"

# ===== 工具函数 =====
def mat_from_pos_mat(pos, mat):
    T = np.eye(4)
    T[:3, :3] = np.array(mat).reshape(3, 3)
    T[:3, 3] = pos
    return T

# ===== 加载模型 =====
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

tcp_body_id = model.body("openarm_left_hand_tcp").id
board_body_id = model.body("calibration_board").id
camera_id = model.camera("body_depth_camera").id

# ===== 手眼标定结果（从 handeye_calibration.py 复制）=====
X = np.array([
    [-0.,  1.,  0., -0.4   ],
    [-0.,  0., -1.,  0.158 ],
    [-1., -0.,  0.,  0.3819],
    [ 0.,  0.,  0.,  1.    ]
])

print("开始验证手眼标定稳定性（Ctrl+C 结束）\n")

while True:
    mujoco.mj_step(model, data)

    # TCP -> world
    tcp_pos = data.xpos[tcp_body_id].copy()
    tcp_mat = data.xmat[tcp_body_id].copy()
    T_tcp_world = mat_from_pos_mat(tcp_pos, tcp_mat)

    # Board -> world
    board_pos = data.xpos[board_body_id].copy()
    board_mat = data.xmat[board_body_id].copy()
    T_board_world = mat_from_pos_mat(board_pos, board_mat)

    # Camera -> world
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_mat = data.cam_xmat[camera_id].copy()
    T_cam_world = mat_from_pos_mat(cam_pos, cam_mat)

    # Board -> camera
    T_board_cam = np.linalg.inv(T_cam_world) @ T_board_world

    # Board -> TCP
    T_board_tcp = np.linalg.inv(T_tcp_world) @ X @ T_board_cam

    p = T_board_tcp[:3, 3]

    print(f"Board in TCP frame: x={p[0]: .4f}, y={p[1]: .4f}, z={p[2]: .4f}")
    time.sleep(0.2)
