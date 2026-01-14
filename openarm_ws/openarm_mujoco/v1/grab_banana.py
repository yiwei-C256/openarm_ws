import mujoco
import mujoco.viewer as viewer
import cv2
import numpy as np
import time
import signal
import sys

# ===================== 核心配置（含手眼标定矩阵） =====================
# 1. 基础配置
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480
DISPLAY_UPDATE_STEP = 5
CAMERA_NAME = "d435"
DISABLE_VIEWER = False

# 2. 手眼标定矩阵（替换为你的标定结果）
CALIBRATION_MATRIX = np.array([
    [0.020608, -0.999720, 0.011610, -0.015986],
    [-0.745379, -0.023102, -0.666240, 0.429087],
    [0.666322, 0.005076, -0.745647, 0.485279],
    [0.000000, 0.000000, 0.000000, 1.000000],
], dtype=np.float32)

# 3. 摄像头内参（标定得到）
CAMERA_INTRINSIC = np.array([
    [431.47147, 0.0, 240.5],
    [0.0, 431.47147, 240.5],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# 4. 机械臂/香蕉配置
BANANA_BODY_NAME = "banana"          
END_EFFECTOR_SITE_NAME = "ee_site"   
LEFT_ARM_JOINTS = [                  
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7"
]
LEFT_GRIPPER_JOINTS = [             
    "openarm_left_finger_joint1",
    "openarm_left_finger_joint2"
]

# 5. 抓取参数（适配香蕉碰撞体+修正夹爪值）
POSITION_TOLERANCE = 0.003  # 更高精度
GRIPPER_CLOSE_VAL = 0.0    # 你说的“0=关上”
GRIPPER_OPEN_VAL = 1.0     # 你说的“1=开满（张开）”
STEP_DELAY = 0.001
MAX_MOVE_STEPS = 6000       # 增加步数提高精度
SAFE_HEIGHT = 0.12          # 微调安全高度，减少多余移动
GRASP_HEIGHT_OFFSET = 0.015  # 更贴近香蕉高度
# 核心配置：放宽抓取距离阈值（匹配日志里的可抓取距离）
GRASP_DISTANCE_THRESHOLD = 0.08  # 抓取有效距离阈值（8cm）

# 6. 香蕉检测配置（修改：轮廓红线参数）
BANANA_DETECTION_COLOR_LOW = np.array([8, 30, 30])    # 香蕉HSV下限
BANANA_DETECTION_COLOR_HIGH = np.array([45, 255, 255]) # 香蕉HSV上限
DETECTION_KERNEL_SIZE = (7, 7)                        # 去噪核大小
RED_CONTOUR_THICKNESS = 2                             # 红线厚度
RED_CONTOUR_COLOR = (0, 0, 255)                       # 红线颜色(BGR)
DETECTION_FPS = 30                                    # 检测帧率

# ===================== 全局变量（防段错误） =====================
renderer = None
viewer_instance = None
model = None
data = None
is_running = True
arm_joint_ids = []
gripper_joint_ids = []
ee_site_id = -1
banana_body_id = -1
camera_id = -1
detection_window_name = f"{CAMERA_NAME}_banana_contour"  # 轮廓检测窗口名

# ===================== 信号处理（安全退出） =====================
def signal_handler(sig, frame):
    global is_running
    print("\n⏹️  接收到退出信号，正在安全退出...")
    is_running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===================== 核心工具函数（优化版） =====================
def init_mujoco():
    """初始化Mujoco，适配openarm_bimanual_cam_v2.xml结构"""
    global model, data, renderer, viewer_instance, arm_joint_ids, gripper_joint_ids
    global ee_site_id, banana_body_id, camera_id
    
    # 加载模型（直接读取openarm_bimanual_cam_v2.xml的原生初始位姿）
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # 获取核心ID
    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in LEFT_ARM_JOINTS]
    gripper_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in LEFT_GRIPPER_JOINTS]
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE_NAME)
    banana_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BANANA_BODY_NAME)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    
    # 验证ID有效性
    if -1 in arm_joint_ids:
        print(f"⚠️  机械臂关节ID无效: {[id for id in arm_joint_ids if id == -1]}")
    if -1 in gripper_joint_ids:
        print(f"⚠️  夹爪关节ID无效: {[id for id in gripper_joint_ids if id == -1]}")
    if ee_site_id == -1:
        print("⚠️  末端执行器位点ID无效！")
    if banana_body_id == -1:
        print("⚠️  香蕉物体ID无效！")
    if camera_id == -1:
        print("⚠️  摄像头ID无效！")
    
    # 创建渲染器
    try:
        renderer = mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
    except Exception as e:
        print(f"⚠️  Renderer初始化警告: {e}")
        renderer = None
    
    # 启动查看器
    if not DISABLE_VIEWER:
        try:
            viewer_instance = viewer.launch_passive(model, data)
            print("🖥️  3D查看器启动成功")
        except Exception as e:
            print(f"⚠️  3D查看器启动失败：{e}")
            viewer_instance = None
    
    # 创建两个窗口：原始调试窗口 + 香蕉轮廓检测窗口
    try:
        # 原始调试窗口
        cv2.namedWindow(CAMERA_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)
        
        # 香蕉轮廓检测窗口（d435视角，红线描边）
        cv2.namedWindow(detection_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(detection_window_name, CAMERA_WIDTH, CAMERA_HEIGHT)
        print(f"🖥️  {detection_window_name} 检测窗口启动成功（红线描香蕉轮廓）")
    except Exception as e:
        print(f"⚠️  窗口创建警告: {e}")
    
    # 只初始化夹爪，不改动手臂关节
    reset_gripper_only()
    
    return model, data

def reset_gripper_only():
    """仅初始化夹爪为张开状态，不修改手臂关节（保留openarm_bimanual_cam_v2.xml原生初始位姿）"""
    # 只张开夹爪，不碰手臂关节
    for gid in gripper_joint_ids:
        if gid != -1:
            data.ctrl[gid] = GRIPPER_OPEN_VAL  # 初始是“开满（张开）”
    
    # 稳定初始状态（让模型加载后稳定，不修改关节值）
    for _ in range(1000):
        mujoco.mj_step(model, data)
    
    # 打印当前左手初始位姿（验证是否是openarm_bimanual_cam_v2.xml原生值）
    current_arm_pose = []
    for jid in arm_joint_ids:
        if jid != -1:
            current_arm_pose.append(round(data.qpos[jid], 3))
    print(f"✅ 左手保留openarm_bimanual_cam_v2.xml原生初始位姿: {current_arm_pose}")
    print("✅ 夹爪已初始化为开满（张开）状态")

def detect_banana_with_red_contour(img_bgr):
    """
    修改版：实时检测香蕉并沿轮廓绘制红线
    :param img_bgr: BGR格式的相机图像
    :return: 绘制红线后的图像, 香蕉是否被检测到, 香蕉中心坐标
    """
    # 1. 颜色空间转换 + 颜色滤波
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BANANA_DETECTION_COLOR_LOW, BANANA_DETECTION_COLOR_HIGH)
    
    # 2. 形态学操作去噪（优化轮廓提取效果）
    kernel = np.ones(DETECTION_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 开运算去除噪点
    
    # 3. 查找轮廓并筛选最大轮廓（香蕉）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    banana_detected = False
    banana_center = None
    img_with_contour = img_bgr.copy()
    
    if contours:
        # 取面积最大的轮廓（香蕉）
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 50:  # 过滤小噪点
            banana_detected = True
            
            # 核心修改：沿轮廓绘制红线（替换红框）
            cv2.drawContours(img_with_contour, [max_contour], -1, RED_CONTOUR_COLOR, RED_CONTOUR_THICKNESS)
            
            # 计算轮廓中心坐标
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                banana_center = (cx, cy)
                # 绘制中心点 + 标注文字
                cv2.circle(img_with_contour, (cx, cy), 4, (0, 255, 0), -1)  # 绿色中心点
                cv2.putText(img_with_contour, "Banana", (cx - 20, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_CONTOUR_COLOR, 2)
    
    # 4. 标注检测状态
    status_text = "Banana: DETECTED" if banana_detected else "Banana: NOT FOUND"
    cv2.putText(img_with_contour, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_with_contour, banana_detected, banana_center

def pixel_to_world(pixel_xy, z=0.305):
    """将2D像素坐标转换为3D世界坐标（核心：应用手眼标定矩阵）"""
    # 1. 像素坐标转相机坐标系（已知Z=0.305，桌面高度）
    fx, fy = CAMERA_INTRINSIC[0,0], CAMERA_INTRINSIC[1,1]
    cx, cy = CAMERA_INTRINSIC[0,2], CAMERA_INTRINSIC[1,2]
    
    x_cam = (pixel_xy[0] - cx) * z / fx
    y_cam = (pixel_xy[1] - cy) * z / fy
    z_cam = z
    
    # 2. 相机坐标系转世界坐标系（应用手眼标定矩阵）
    cam_point = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    world_point = np.dot(np.linalg.inv(CALIBRATION_MATRIX), cam_point)
    
    return world_point[:3]

def get_ee_pos():
    """获取末端执行器位置"""
    if ee_site_id == -1:
        return np.array([0.0, 0.0, 0.0])
    return data.site_xpos[ee_site_id].copy()

def get_banana_pos():
    """获取香蕉的实际3D位置（备用）"""
    if banana_body_id == -1:
        return np.array([0.35, 0.0, 0.305])
    return data.xpos[banana_body_id].copy()

# 新增：计算末端执行器与香蕉的欧式距离
def get_ee_banana_distance():
    """获取末端执行器（爪子）到香蕉的直线距离"""
    ee_pos = get_ee_pos()
    banana_pos = get_banana_pos()
    distance = np.linalg.norm(ee_pos - banana_pos)
    return distance

def move_arm_to_target(target_pos):
    """
    优化版机械臂运动控制：
    1. 适配openarm_bimanual_cam_v2.xml的position控制器
    2. 分步运动，先Z轴再XY轴，防止碰撞
    3. 位置闭环控制，更稳定
    4. 新增：末端-香蕉距离判定 + 返回是否达到抓取距离
    """
    current_pos = get_ee_pos()
    # 新增：打印初始距离
    initial_distance = get_ee_banana_distance()
    print(f"\n📌 移动到目标位置: {target_pos} (当前: {current_pos}) | 初始爪-蕉距离: {initial_distance:.4f}m")
    
    # 分步运动：先抬升到安全高度
    safe_target = target_pos.copy()
    safe_target[2] = current_pos[2] if current_pos[2] > SAFE_HEIGHT else SAFE_HEIGHT
    print(f"🔼 先抬升到安全高度: {safe_target}")
    
    # 新增变量标记是否达到抓取距离
    grasp_reached = False
    step = 0
    while step < MAX_MOVE_STEPS and is_running and not grasp_reached:
        current_pos = get_ee_pos()
        error = safe_target - current_pos
        error_norm = np.linalg.norm(error)
        # 新增：实时计算距离
        current_distance = get_ee_banana_distance()
        
        # 调试信息（新增距离显示）
        if step % 200 == 0:
            print(f"   步数{step}: 位置误差={error_norm:.4f}m | 爪-蕉距离={current_distance:.4f}m")
        
        # 判断是否达到抓取距离，达到则标记并退出循环
        if current_distance < GRASP_DISTANCE_THRESHOLD:
            grasp_reached = True
            print(f"✅ 达到抓取距离！爪-蕉距离={current_distance:.4f}m < {GRASP_DISTANCE_THRESHOLD}m，停止运动")
            break
        
        # 到达目标（原逻辑保留）
        if error_norm < POSITION_TOLERANCE:
            break
        
        # 位置控制器：设置ctrl目标（适配openarm_bimanual_cam_v2.xml的position actuator）
        # 计算雅克比矩阵
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)
        jacp_arm = jacp[:, arm_joint_ids]
        
        # 伪逆求解关节增量（比例系数从5→10，加快调整）
        if np.linalg.matrix_rank(jacp_arm) >= 3:
            joint_delta = np.dot(np.linalg.pinv(jacp_arm), error * 10.0)  
            for i, jid in enumerate(arm_joint_ids):
                if jid != -1:
                    new_pos = data.qpos[jid] + joint_delta[i] * STEP_DELAY * 10
                    data.ctrl[jid] = np.clip(new_pos, 
                                           model.jnt_range[jid][0], 
                                           model.jnt_range[jid][1])
        
        mujoco.mj_step(model, data)
        
        # 渲染和同步
        if step % DISPLAY_UPDATE_STEP == 0:
            render_debug_info()
            time.sleep(STEP_DELAY)
        
        step += 1
    
    # 第二步：移动到目标XY，保持Z（仅当未达到抓取距离时执行）
    if not grasp_reached:
        step = 0
        while step < MAX_MOVE_STEPS and is_running:
            current_pos = get_ee_pos()
            error = target_pos - current_pos
            error[2] *= 2.0  # Z轴权重更高
            error_norm = np.linalg.norm(error)
            # 新增：实时计算距离
            current_distance = get_ee_banana_distance()
            
            if step % 200 == 0:
                print(f"   步数{step}: 位置误差={error_norm:.4f}m | 爪-蕉距离={current_distance:.4f}m")
            
            # 第二步也判断抓取距离
            if current_distance < GRASP_DISTANCE_THRESHOLD:
                grasp_reached = True
                print(f"✅ 达到抓取距离！爪-蕉距离={current_distance:.4f}m < {GRASP_DISTANCE_THRESHOLD}m，停止运动")
                break
            
            # 到达目标（原逻辑保留）
            if error_norm < POSITION_TOLERANCE:
                final_distance = get_ee_banana_distance()
                print(f"✅ 到达目标！最终位置误差={error_norm:.4f}m | 最终爪-蕉距离={final_distance:.4f}m")
                break
            
            # 位置控制（比例系数从5→10）
            jacp = np.zeros((3, model.nv))
            mujoco.mj_step(model, data)
            mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)
            jacp_arm = jacp[:, arm_joint_ids]
            
            if np.linalg.matrix_rank(jacp_arm) >= 3:
                joint_delta = np.dot(np.linalg.pinv(jacp_arm), error * 10.0)
                for i, jid in enumerate(arm_joint_ids):
                    if jid != -1:
                        new_pos = data.qpos[jid] + joint_delta[i] * STEP_DELAY * 10
                        data.ctrl[jid] = np.clip(new_pos,
                                               model.jnt_range[jid][0],
                                               model.jnt_range[jid][1])
            
            mujoco.mj_step(model, data)
            
            if step % DISPLAY_UPDATE_STEP == 0:
                render_debug_info()
                time.sleep(STEP_DELAY)
            
            step += 1
        
        if step >= MAX_MOVE_STEPS:
            final_error = np.linalg.norm(get_ee_pos() - target_pos)
            final_distance = get_ee_banana_distance()
            print(f"⚠️  未到达目标，最终位置误差={final_error:.4f}m | 最终爪-蕉距离={final_distance:.4f}m")
    
    # 返回是否达到抓取距离的标记
    return grasp_reached

def control_gripper(is_close):
    """优化版夹爪控制（适配你的“0关1开”规则）"""
    val = GRIPPER_CLOSE_VAL if is_close else GRIPPER_OPEN_VAL
    action = "闭合" if is_close else "张开"
    # 新增：打印夹爪动作时的距离
    current_distance = get_ee_banana_distance()
    print(f"\n🤏 {action}夹爪 (值={val}) | 当前爪-蕉距离={current_distance:.4f}m")
    
    # 同时控制两个finger，确保同步开/关
    for gid in gripper_joint_ids:
        if gid != -1:
            data.ctrl[gid] = val
    
    # 稳定夹爪状态
    for i in range(1000):
        if not is_running:
            return
        mujoco.mj_step(model, data)
        if i % DISPLAY_UPDATE_STEP == 0:
            render_debug_info()
            time.sleep(STEP_DELAY / 2)
    
    print(f"✅ 夹爪{action}完成")

def render_debug_info():
    """渲染调试信息（修改：调用轮廓红线检测）"""
    try:
        if renderer and is_running:
            # 1. 渲染d435相机画面
            renderer.update_scene(data, camera=CAMERA_NAME)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 2. 实时检测香蕉并绘制轮廓红线（核心修改）
            img_with_contour, banana_detected, banana_center = detect_banana_with_red_contour(img_bgr)
            
            # 3. 显示原始调试窗口（保留原有信息）
            ee_pos = get_ee_pos()
            banana_pos = get_banana_pos()
            ee_banana_dist = get_ee_banana_distance()
            debug_text = [
                f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                f"Banana: [{banana_pos[0]:.3f}, {banana_pos[1]:.3f}, {banana_pos[2]:.3f}]",
                f"EE-Banana Dist: {ee_banana_dist:.4f}m",
                f"Gripper: {data.ctrl[gripper_joint_ids[0]] if gripper_joint_ids[0] != -1 else 'N/A'}",
                f"Banana Detected: {banana_detected}"
            ]
            
            img_debug = img_bgr.copy()
            y_offset = 30
            for text in debug_text:
                cv2.putText(img_debug, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                y_offset += 25
            
            # 4. 显示两个窗口
            cv2.imshow(CAMERA_NAME, img_debug)                  # 原始调试窗口
            cv2.imshow(detection_window_name, img_with_contour) # 轮廓红线检测窗口
            
        if is_running:
            cv2.waitKey(1)
        
        # 同步3D查看器
        if viewer_instance and not DISABLE_VIEWER and is_running:
            viewer_instance.sync()
    except Exception as e:
        print(f"⚠️  渲染警告: {e}")

# ========== 新增：独立的关节4举升函数（完全不影响原有逻辑） ==========
def lift_banana_with_joint4(target_angle=-1.0):
    """
    夹到香蕉后，单独控制关节4转动举升香蕉
    :param target_angle: 关节4目标角度（rad），默认-1.0（能明显举升香蕉）
    """
    # 获取关节4的ID（LEFT_ARM_JOINTS[3]是joint4）
    joint4_id = arm_joint_ids[3]
    if joint4_id == -1:
        print("❌ 关节4 ID无效，无法控制")
        return
    
    # 获取关节4当前角度和运动范围
    current_angle = data.qpos[joint4_id]
    joint4_min = model.jnt_range[joint4_id][0]
    joint4_max = model.jnt_range[joint4_id][1]
    target_angle = np.clip(target_angle, joint4_min, joint4_max)
    
    print(f"\n📈 纯关节4举升控制：开始转动关节4举升香蕉...")
    print(f"   关节4初始角度: {current_angle:.4f} rad")
    print(f"   关节4目标角度: {target_angle:.4f} rad")
    print(f"   关节4运动范围: {joint4_min:.2f} ~ {joint4_max:.2f} rad")
    
    step = 0
    # 增大循环步数，确保关节4有足够时间转动到位
    while step < 5000 and is_running:
        current_angle = data.qpos[joint4_id]
        error = target_angle - current_angle
        error_abs = abs(error)
        
        # 每5步打印一次进度（更密集，方便观察）
        if step % 5 == 0:
            moved_angle = current_angle - (data.qpos[joint4_id] - error)
            print(f"   步数{step}: 当前角度={current_angle:.4f} rad | 已转动={moved_angle:.4f} rad")
        
        # 到达目标角度（误差<0.05rad即可，避免无限循环）
        if error_abs < 0.05:
            print(f"✅ 关节4举升完成！最终角度: {current_angle:.4f} rad")
            break
        
        # 核心：直接设置关节4的目标角度（适配position控制器）
        data.ctrl[joint4_id] = target_angle
        
        # 步进仿真，确保关节响应
        mujoco.mj_step(model, data)
        
        # 渲染调试信息，保持画面更新
        if step % DISPLAY_UPDATE_STEP == 0:
            render_debug_info()
            time.sleep(STEP_DELAY / 5)  # 减少延迟，加快转动响应
        
        step += 1
    
    if step >= 5000:
        final_angle = data.qpos[joint4_id]
        print(f"⚠️  关节4举升步数超限，强制停止 | 最终角度: {final_angle:.4f} rad")

# ===================== 核心抓取逻辑（修复闪退+仅保留到关节4举升完成） =====================
def auto_grasp():
    """自动抓取香蕉（仅执行到关节4举升完成，后续动作全部跳过，避免甩掉香蕉）"""
    # 关键修复：声明使用全局变量is_running
    global is_running
    print("\n===== 开始香蕉抓取流程（v2原生初始位姿版）======")
    
    # 1. 直接获取香蕉的实际3D位置（跳过2D检测，避免干扰）
    print("\n📍 直接获取香蕉实际3D位置...")
    banana_3d = get_banana_pos()
    # 新增：打印初始爪-蕉距离
    initial_dist = get_ee_banana_distance()
    print(f"🍌 香蕉实际3D坐标: {banana_3d} | 初始爪-蕉距离: {initial_dist:.4f}m")
    
    # 2. 确认夹爪是张开状态（双重确认）
    print("\n✋ 确认夹爪张开...")
    control_gripper(False)
    
    # 3. 移动到香蕉上方安全位置（从v2原生初始位姿出发）
    safe_pos = banana_3d.copy()
    safe_pos[2] += SAFE_HEIGHT
    move_arm_to_target(safe_pos)
    
    # 4. 下降到抓取位置（更贴近香蕉）
    grasp_pos = banana_3d.copy()
    grasp_pos[2] += GRASP_HEIGHT_OFFSET  # 更贴近香蕉高度
    # 接收move_arm_to_target返回的“是否达到抓取距离”标记
    grasp_reached = move_arm_to_target(grasp_pos)
    
    # 如果达到抓取距离，直接闭合夹爪（不再继续后续无效运动）
    if grasp_reached:
        print("\n🚀 已达到可抓取距离，立即闭合夹爪！")
        control_gripper(True)
    else:
        # 原逻辑：未达到则尝试闭合（保留）
        control_gripper(True)
    
    # ========== 调用关节4举升函数 ==========
    lift_banana_with_joint4(target_angle=-1.0)
    
    # ========== 核心修改：关节4举升完成后，安全保持画面显示 ==========
    print("\n🎉 抓取+关节4举升完成！已停止所有后续动作，避免香蕉被甩掉！")
    print("\n📌 按ESC/q退出查看状态...")
    
    # 修复：安全的画面保持循环，添加异常处理
    try:
        while is_running:
            # 仅步进仿真，不移动机械臂
            mujoco.mj_step(model, data)
            render_debug_info()
            
            # 检测退出按键，避免卡死
            key = cv2.waitKey(1) & 0xFF
            if key in [27, 113]:  # ESC/q
                is_running = False
                break
    except Exception as e:
        print(f"\n⚠️  画面保持循环警告: {e}")
        is_running = False
    
    return True

# ===================== 安全退出函数 =====================
def safe_cleanup():
    """安全清理所有资源，避免GLFW错误"""
    global is_running, viewer_instance, renderer
    is_running = False
    
    # 先关闭CV窗口
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"⚠️  关闭CV窗口警告: {e}")
    
    # 再关闭3D查看器
    try:
        if viewer_instance:
            viewer_instance.close()
            viewer_instance = None
    except Exception as e:
        print(f"⚠️  关闭查看器警告: {e}")
    
    # 最后释放渲染器
    try:
        if renderer:
            renderer.close()
            renderer = None
    except Exception as e:
        print(f"⚠️  释放渲染器警告: {e}")

# ===================== 主函数（新增按键等待逻辑） =====================
def main():
    global is_running
    
    # 初始化（只加载openarm_bimanual_cam_v2.xml原生初始位姿，不手动修改）
    try:
        init_mujoco()
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        safe_cleanup()
        return
    
    # 初始稳定（不修改关节，仅让模型稳定）
    print("\n🔄 稳定openarm_bimanual_cam_v2.xml原生初始状态...")
    try:
        for i in range(1000):
            if not is_running:
                break
            mujoco.mj_step(model, data)
            render_debug_info()
    except Exception as e:
        print(f"\n⚠️  初始稳定循环警告: {e}")
    
    # ========== 新增：等待用户按键后再开始抓取 ==========
    print("\n=====================================")
    print("📢 准备就绪！按【Enter键】开始抓取香蕉")
    print("=====================================")
    # 等待用户按Enter键（可修改为其他按键，比如输入's'再回车）
    input("")  # 空input()表示等待任意输入+Enter
    
    # 自动抓取（执行到关节4举升完成即停止）
    try:
        auto_grasp()
    except Exception as e:
        print(f"\n❌ 抓取流程出错: {e}")
    
    # 安全退出
    print("\n🔌 正在安全退出...")
    safe_cleanup()
    print("✅ 程序正常退出")

# ===================== 启动 =====================
if __name__ == "__main__":
    # 终极防护：捕获所有未处理异常
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序崩溃: {e}")
        safe_cleanup()