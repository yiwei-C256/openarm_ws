import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, TransformStamped # 新增
import tf2_ros # 新增：用于发布坐标变换
import mujoco
import numpy as np
import glfw
from cv_bridge import CvBridge
import  time

class OpenArmMujocoSync(Node):
    def __init__(self):
        super().__init__("openarm_mujoco_sync")
        
        # ========== 1. 初始化数据 ==========
        self.joint_data = {}
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) # TF 广播器
        
        # 鼠标交互状态
        self.button_left = self.button_middle = self.button_right = False
        self.last_x = self.last_y = 0

        # ========== 2. 加载模型 ==========
        model_path = '/home/ldp/openarm_ws/openarm_mujoco/v1/scene.xml' 
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 获取 body 的 ID
        self.tcp_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_hand_tcp")
        
        if self.tcp_body_id == -1:
            print("警告: 未找到名为 openarm_left_hand_tcp 的 body")

        # 末端 body 名称
        self.arm_ee_name = "openarm_left_hand_tcp"
        self.finger_ee_name = "openarm_left_right_finger"
        # 夹爪 actuator 名字一定跟 XML 对应
        self.grip_act_name = 'left_finger1_ctrl'   # ← 用 XML 里的名字
        self.grip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        self.grip_act_name)
        
        # 与 XML 完全一致
        self.arm_act_names = [f'left_joint{i}_ctrl' for i in range(1, 8)]
        self.arm_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.arm_act_names
        ]
        if -1 in self.arm_act_ids:
            self.get_logger().fatal(f'臂 actuator 缺失：{self.arm_act_names}')

        if self.grip_id == -1:
            self.get_logger().fatal(f'找不到夹爪 actuator: {self.grip_act_name}')

        # 关节映射
        self.arm_joint_names = [f"openarm_left_joint{i}" for i in range(1, 8)] + \
                               [f"openarm_right_joint{i}" for i in range(1, 8)] + \
                               ["openarm_left_finger_joint1", "openarm_right_finger_joint1"]
        self.joint_qpos_map = {n: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] 
                               for n in self.arm_joint_names if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) != -1}

        # --- 获取香蕉 ID ---
        self.banana_name = "banana"
        self.banana_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.banana_name)
        if self.banana_id == -1:
            self.get_logger().error(f"模型中找不到物体: {self.banana_name}")

        # ========== 3. GLFW 初始化 (保持双窗口逻辑) ==========
        if not glfw.init(): raise Exception("GLFW 无法初始化")
        self.last_render_time = time.time()

        # A. 交互窗口
        self.win_main = glfw.create_window(1200, 900, 'OpenArm - Interaction & Banana Tracking', None, None)
        glfw.make_context_current(self.win_main)
        self.ctx_main = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.scn_main = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam_main = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam_main)
        # --- 自定义初始视角 ---
        # 1. 目标点 (相机盯着哪看): [x, y, z]
        self.cam_main.lookat = np.array([0.35, 0.0, 0.32]) 

        # 2. 距离 (相机离目标点多远)
        self.cam_main.distance = 2.5 

        # 3. 方位角 (水平旋转，单位：度)
        # 0: 从正 X 轴看, 90: 从正 Y 轴看, 180: 从负 X 轴看, 270: 从负 Y 轴看
        self.cam_main.azimuth = 180

        # 4. 仰角 (垂直旋转，单位：度)
        # -90: 正上方往下看, 0: 水平看, -20: 稍微斜向下看
        self.cam_main.elevation = -25.0 

    
        glfw.set_cursor_pos_callback(self.win_main, self.mouse_move)
        glfw.set_mouse_button_callback(self.win_main, self.mouse_button)
        glfw.set_scroll_callback(self.win_main, self.scroll)

        # B. 相机发布窗口 (隐藏)
        self.cam_pub_width, self.cam_pub_height = 640, 480
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.win_offscreen = glfw.create_window(self.cam_pub_width, self.cam_pub_height, "Offscreen", None, None)
        glfw.make_context_current(self.win_offscreen)
        self.ctx_off = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.scn_off = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam_fixed = mujoco.MjvCamera()
        
        # D435 相机设置
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435")
        self.cam_fixed.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam_fixed.fixedcamid = cam_id if cam_id != -1 else 0
        self.vopt = mujoco.MjvOption()

        # ========== 4. ROS 通信 ==========
        self.image_pub = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.banana_pose_pub = self.create_publisher(PoseStamped, "/banana_pose", 10) # 发布香蕉位姿 Topic
        self.subscription = self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)

        # self.left_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")]
         # 1. 获取 Mocap Body 的 ID (注意：这是在 mocap 数组中的索引)
        # mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_mocap")
        # self.left_mocap_idx = self.model.body_mocapid[mocap_body_id]

        # 2. 获取末端 TCP 的 ID (假设它是一个 Site)
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "openarm_left_hand_tcp")

        self.log_tick = 0  # 增加日志计数器

        self.banana_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "banana")
        # 获取香蕉 自由关节 (freejoint) 的 qpos 起始地址
        # 假设 XML 中香蕉的 joint 叫 "banana_joint"
        banana_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "banana_joint")
        self.banana_qpos_addr = self.model.jnt_qposadr[banana_joint_id]
        self.banana_dof_addr = self.model.jnt_dofadr[banana_joint_id]

        self.is_grasped = False  # 抓取状态位
        self.GRASP_THRESHOLD = 0.04       # 判定抓取的距离阈值
        self.finger_joint_name = "openarm_left_finger_joint1" 
        self.FINGER_CLOSED_THRESHOLD = 0.01 # 手指关节值大于此值认为在关闭状态
        self.FINGER_OPEN_THRESHOLD = 0.005   # 手指关节值小于此值认为已张开

        # 新增：用于存储抓取瞬间的相对位姿
        self.relative_pos_offset = np.zeros(3)
        self.relative_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])

    def joint_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_qpos_map:
                self.joint_data[name] = msg.position[i]

    def get_tcp_pose(self):
        # 1. 获取全局位置 (World Position)
        # 返回的是 [x, y, z] 的 numpy 数组
        pos = self.data.xpos[self.tcp_body_id]

        # 2. 获取全局姿态 (World Orientation)
        # data.xmat 返回的是 3x3 的旋转矩阵
        # 我们通常将其转换为四元数 [w, x, y, z] 方便使用
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self.tcp_body_id])
        
        return pos, quat

    def publish_banana_tf(self):
        """获取并发布香蕉位姿到 TF 和 Topic"""
        if self.banana_id == -1: return

        # 从 MuJoCo 读取位姿 (World坐标系)
        pos = self.data.xpos[self.banana_id]
        quat = self.data.xquat[self.banana_id] # MuJoCo: [w, x, y, z]

        # 1. 发布 TF (让 RViz 看到香蕉)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'banana'
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        # 注意：ROS 2 四元数顺序是 [x, y, z, w]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]
        t.transform.rotation.w = quat[0]
        self.tf_broadcaster.sendTransform(t)

        # 2. 发布 PoseStamped Topic
        p = PoseStamped()
        p.header = t.header
        p.pose.position.x, p.pose.position.y, p.pose.position.z = pos[0], pos[1], pos[2]
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = quat[1], quat[2], quat[3], quat[0]
        self.banana_pose_pub.publish(p)

    def run_once(self):
        now = time.time()
        elapsed = now - self.last_render_time
        self.last_render_time = now

        dt = self.model.opt.timestep
        steps = min(int(elapsed / dt), 100)

        # 设定距离阈值（例如 5 厘米）
        # GRASP_THRESHOLD = 0.04 

        # === 关键修改：在每一个物理子步中都强制同步位置并清除速度 ===
        for _ in range(max(steps, 1)):
            # 1. 强制同步关节位置
            for name, pos in self.joint_data.items():
                if name in self.joint_qpos_map:
                    addr = self.joint_qpos_map[name]
                    self.data.qpos[addr] = pos
                    
                    # 2. 【核心】强制将该关节的速度设为 0
                    # 这样物理引擎就不会认为机械臂有巨大的动量
                    # 获取该关节对应的自由度索引 (DOF address)
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    dof_addr = self.model.jnt_dofadr[joint_id]
                    self.data.qvel[dof_addr] = 0 

                    # 处理手指联动
                    if "finger_joint1" in name:
                        m_name = name.replace("joint1", "joint2")
                        mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m_name)
                        if mid != -1:
                            self.data.qpos[self.model.jnt_qposadr[mid]] = pos
                            self.data.qvel[self.model.jnt_dofadr[mid]] = 0

            # 2. 获取当前 TCP 和香蕉的状态
            mujoco.mj_kinematics(self.model, self.data) # 刷新全局坐标
            tcp_pos = self.data.xpos[self.tcp_body_id].copy()
            tcp_mat = self.data.xmat[self.tcp_body_id].copy()
            banana_pos = self.data.xpos[self.banana_body_id].copy()
            # distance = np.linalg.norm(tcp_pos - banana_pos)
            
            # 获取手指当前的关节值
            finger_val = self.joint_data.get(self.finger_joint_name, 0.0)

            # 3. 【核心逻辑：抓取状态机】

            # A. 释放判定：如果手指打开超过阈值，强制解锁
            if finger_val > 0.03: 
                if self.is_grasped:
                    self.is_grasped = False
                    self.get_logger().info(f"Released! Finger opened (val: {finger_val:.4f})")

            # B. 抓取判定：如果目前没抓，且手指闭合(小于0.01) 且 距离近
            elif not self.is_grasped:
                distance = np.linalg.norm(tcp_pos - banana_pos)
                if finger_val < 0.01 and distance < self.GRASP_THRESHOLD:
                    self.is_grasped = True
                    
                    # === 核心：记录抓取瞬间的相对位姿 ===
                    # A. 计算相对位置偏移（在世界坐标系下）
                    self.relative_pos_offset = banana_pos - tcp_pos
                    
                    # B. 计算相对旋转偏移
                    # 公式: Banana_Quat = TCP_Quat * Relative_Quat
                    # 所以: Relative_Quat = inv(TCP_Quat) * Banana_Quat
                    tcp_quat = np.empty(4)
                    banana_quat = np.empty(4)
                    mujoco.mju_mat2Quat(tcp_quat, tcp_mat)
                    mujoco.mju_mat2Quat(banana_quat, self.data.xmat[self.banana_body_id])
                    
                    # 求 TCP 四元数的逆
                    tcp_quat_inv = np.empty(4)
                    mujoco.mju_negQuat(tcp_quat_inv, tcp_quat)
                    
                    # 计算相对四元数并保存
                    mujoco.mju_mulQuat(self.relative_quat_offset, tcp_quat_inv, banana_quat)
                    
                    self.get_logger().info("Grasped and Offset Recorded!========================")

            # 4. 执行同步（传送）
            if self.is_grasped:
                # self.get_logger().info("Grasped banana")
                # A. 同步位置：当前 TCP 位置 + 之前的相对偏移
                # 注意：如果希望香蕉随手转动，偏移量也需要随手旋转，这里简化处理直接叠加
                # 若要完美旋转位移，需使用 mju_rotVecQuat 处理 relative_pos_offset
                self.data.qpos[self.banana_qpos_addr : self.banana_qpos_addr + 3] = tcp_pos + self.relative_pos_offset
                
                # B. 同步姿态：保持相对旋转
                # 公式：当前香蕉姿态 = 当前 TCP 四元数 * 记录的相对四元数
                current_tcp_quat = np.empty(4)
                mujoco.mju_mat2Quat(current_tcp_quat, self.data.xmat[self.tcp_body_id])
                
                target_banana_quat = np.empty(4)
                mujoco.mju_mulQuat(target_banana_quat, current_tcp_quat, self.relative_quat_offset)
                
                self.data.qpos[self.banana_qpos_addr + 3 : self.banana_qpos_addr + 7] = target_banana_quat

                # C. 速度清零
                self.data.qvel[self.banana_dof_addr : self.banana_dof_addr + 6] = 0

            # 执行物理步进（计算香蕉的掉落、碰撞等）
            mujoco.mj_step(self.model, self.data)

        # --- 循环结束后获取并打印一次 ---
        # pos, quat = self.get_tcp_pose()

        # --- 关键：限频日志打印 ---
        # 假设 run_once 运行频率很高，我们每 50 次循环打印一次
        self.log_tick = 5
        if self.log_tick % 10 == 0:
            pos, quat = self.get_tcp_pose()
            finger_val = self.joint_data.get(self.finger_joint_name, 0.0)
            self.get_logger().info(f"\n" + "="*40)
            print(f"finger_val={finger_val}")
            
            # 使用 self.get_logger().info
            self.get_logger().info(
                f"\n[TCP Pose]\n"
                f"Pos: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}\n"
                f"Quat: W={quat[0]:.4f}, X={quat[1]:.4f}, Y={quat[2]:.4f}, Z={quat[3]:.4f}"
            )

            # 2. 获取香蕉位姿
            banana_pos = self.data.xpos[self.banana_body_id]
            banana_quat = np.empty(4)
            mujoco.mju_mat2Quat(banana_quat, self.data.xmat[self.banana_body_id])

            # 3. 计算欧式距离
            distance = np.linalg.norm(pos - banana_pos)
            
            # 4. 使用 self.get_logger().info 打印
            self.get_logger().info(
                f"\n[Relative Distance]: {distance:.4f} m" +
                f"\n[TCP Pos]   : X={tcp_pos[0]:.4f}, Y={tcp_pos[1]:.4f}, Z={tcp_pos[2]:.4f}" +
                f"\n[Banana Pos]: X={banana_pos[0]:.4f}, Y={banana_pos[1]:.4f}, Z={banana_pos[2]:.4f}" +
                f"\n" + "="*40
            )

        # 发布香蕉位置
        self.publish_banana_tf()

        # 渲染显示窗口
        glfw.make_context_current(self.win_main)
        w, h = glfw.get_framebuffer_size(self.win_main)
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, None, self.cam_main, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn_main)
        mujoco.mjr_render(mujoco.MjrRect(0,0,w,h), self.scn_main, self.ctx_main)
        glfw.swap_buffers(self.win_main)

        # 渲染发布图像
        glfw.make_context_current(self.win_offscreen)
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, None, self.cam_fixed, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn_off)
        mujoco.mjr_render(mujoco.MjrRect(0,0,self.cam_pub_width, self.cam_pub_height), self.scn_off, self.ctx_off)
        rgb = np.empty((self.cam_pub_height, self.cam_pub_width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0,0,self.cam_pub_width, self.cam_pub_height), self.ctx_off)
        img_msg = self.bridge.cv2_to_imgmsg(np.flipud(rgb), encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_link"
        self.image_pub.publish(img_msg)

        glfw.poll_events()

    # (mouse_button, mouse_move, scroll 回调函数保持不变...)
    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        dx, dy = xpos - self.last_x, ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos
        if not (self.button_left or self.button_middle or self.button_right): return
        width, height = glfw.get_window_size(window)
        action = mujoco.mjtMouse.mjMOUSE_PAN_V if self.button_middle else \
                 (mujoco.mjtMouse.mjMOUSE_ROTATE_V if self.button_left else mujoco.mjtMouse.mjMOUSE_ZOOM)
        mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scn_main, self.cam_main)

    def scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scn_main, self.cam_main)

    # 在 OpenArmMujocoSync 类里增加 =========================
    def pick_banana_mujoco(self):
        """
        纯 MuJoCo 闭环抓取，零 mocap 版。
        末端用 openarm_left_hand_tcp，夹爪用 openarm_left_right_finger 判断
        """
        self.get_logger().info('>>> 开始无 mocap 抓取 <<<')

        # 1. 必备 ID
        banana_id  = self.banana_id
        arm_ee_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.arm_ee_name)
        finger_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.finger_ee_name)
        if -1 in (banana_id, arm_ee_id, finger_id):
            self.get_logger().error('找不到 banana 或 arm_ee 或 finger')
            return

        # 2. 夹爪 actuator 索引（position 控制）
        # grip_act_name = 'openarm_left_finger_joint1'   # 你 xml 里控制开口的 actuator
        grip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.grip_act_name)
        if grip_id == -1:
            self.get_logger().error('找不到夹爪 actuator')
            return

        # 3. 臂关节 motor actuator 索引（力矩控制）
        # arm_act_names = [f'openarm_left_joint{i}_mtr' for i in range(1, 8)]
        # arm_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        #             for n in self.arm_act_names]
        # if -1 in arm_act_ids:
        #     self.get_logger().error('臂 motor actuator 缺失')
        #     return

        # 4. 控制参数
        Kp, Kv = 400, 40
        max_step = 4000
        phase = 0
        t_close = 0.0

        for _ in range(max_step):
            # ---- 4.1 读世界坐标 ----
            b_pos  = self.data.xpos[banana_id]
            ee_pos = self.data.xpos[arm_ee_id]

            # ---- 4.2 分阶段目标 ----
            if phase == 0:                       # 上方 10 cm
                target = b_pos + np.array([0, 0, 0.10])
                if np.linalg.norm(ee_pos - target) < 0.015:
                    phase = 1
            elif phase == 1:                     # 下降
                target = b_pos + np.array([0, 0, 0.015])
                if np.linalg.norm(ee_pos - target) < 0.01:
                    phase = 2
                    t_close = self.data.time
            elif phase == 2:                     # 夹
                self.data.ctrl[grip_id] = 0.0    # 闭合（视 gear 方向）
                if self.data.time - t_close > 1.0:
                    phase = 3
            elif phase == 3:                     # 抬升
                target = b_pos + np.array([0, 0, 0.25])
                if np.linalg.norm(ee_pos - target) < 0.02:
                    self.get_logger().info('>>> 抓取完成 <<<')
                    return

            # ---- 4.3 雅可比转置法 ----
            err_pos = target - ee_pos
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, None, arm_ee_id)
            # 力矩 = J^T * (Kp*err - Kv*J*qvel)
            tau = jacp.T @ (Kp * err_pos - Kv * (jacp @ self.data.qvel[:self.model.nv]))
            # 写进 motor
            for i, aid in enumerate(self.arm_act_ids):
                self.data.ctrl[aid] = tau[i]

            # ---- 4.4 步进 ----
            mujoco.mj_step(self.model, self.data)
            time.sleep(self.model.opt.timestep)

        self.get_logger().warn('超时未抓到')

def main(args=None):
    rclpy.init(args=args)
    node = OpenArmMujocoSync()
    try:
        while rclpy.ok() and not glfw.window_should_close(node.win_main):
            rclpy.spin_once(node, timeout_sec=0.005)
            node.run_once()
            # if glfw.get_key(node.win_main, glfw.KEY_G) == glfw.PRESS:
            #     if not hasattr(node, '_pick_done'):
            #         node._pick_done = True
            #         import threading
            #         threading.Thread(target=node.pick_banana_mujoco, daemon=True).start()
    except KeyboardInterrupt: pass
    finally:
        glfw.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()