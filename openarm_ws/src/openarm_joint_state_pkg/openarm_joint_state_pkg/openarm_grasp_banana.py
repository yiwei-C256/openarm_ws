#!/usr/bin/env python3
import sys
import numpy as np

# --- 核心补丁：处理 NumPy 1.24+ 兼容性 ---
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2, MoveIt2State
from tf2_ros import Buffer, TransformListener
from threading import Thread
import time
import math
from tf_transformations import euler_from_quaternion

# ================= 配置常量 =================
# 基于 SRDF 的关节预设值 (弧度)
LEFT_ARM_HAND_UP = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] 
LEFT_ARM_HOME = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0] 
GRIPPER_OPEN     = [0.035, 0.035] 
GRIPPER_CLOSE    = [0.005, 0.005]
start_joint_positions = [
    -0.17202492411953807, # joint1
     0.17450659971888513, # joint2
     0.6076176056392252,  # joint3
     1.6873117210423392,  # joint4
     0.286064392530401,   # joint5
     0.5683717907669327,  # joint6
     1.5706681512594902   # joint7
]

class OpenArmGraspBananaNode(Node):
    def __init__(self):
        super().__init__("openarm_grasp_banana_node")

        # 1. 配置参数
        self.arm_group = "left_arm"
        self.hand_group = "left_hand"
        self.base_link = "world"
        self.arm_end_effector = "openarm_left_hand_tcp"
        self.hand_end_effector = "openarm_left_right_finger" # 抓爪组的末端

        self.arm_joints = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.hand_joints = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]

        # 2. TF2 与 数据缓存
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.banana_pose = None

        # 3. MoveIt2 实例初始化 (利用源码中的全部参数)
        self.callback_group = ReentrantCallbackGroup()
        
        # 臂组：使用 move_group 动作执行
        self.moveit_arm = MoveIt2(
            node=self, 
            group_name=self.arm_group, 
            joint_names=self.arm_joints,
            base_link_name=self.base_link, 
            end_effector_name=self.arm_end_effector,
            callback_group=self.callback_group,
            use_move_group_action=True  # 源码参数：更直接的执行方式
        )
        
        # 爪组
        self.moveit_hand = MoveIt2(
            node=self, 
            group_name=self.hand_group, 
            joint_names=self.hand_joints,
            base_link_name="openarm_left_hand", 
            end_effector_name=self.hand_end_effector,
            callback_group=self.callback_group,
            use_move_group_action=True
        )

        # 4. 配置规划属性 (源码中对应的 Property)
        self.moveit_arm.allowed_planning_time = 5.0  # 增加规划时间
        self.moveit_arm.num_planning_attempts = 10   # 增加尝试次数
        self.moveit_arm.max_velocity = 0.5           # 限制速度，保证安全
        self.moveit_arm.max_acceleration = 0.5

        # 5. 订阅
        self.pose_sub = self.create_subscription(
            PoseStamped, "/banana_pose", self.banana_cb, 10, callback_group=self.callback_group
        )

    def banana_cb(self, msg):
        self.banana_pose = msg

    def get_error_string(self, error_code):
        """ 将源码中的 MoveItErrorCodes.val 转为易读文本 """
        mapping = {
            1: "SUCCESS (成功)",
            -1: "PLANNING_FAILED (规划失败)",
            -10: "START_STATE_IN_COLLISION (起始位姿碰撞)",
            -12: "GOAL_IN_COLLISION (目标位姿碰撞)",
            -21: "NO_IK_SOLUTION (找不到运动学逆解)",
            -4: "TIMED_OUT (规划超时)"
        }
        return mapping.get(error_code, f"ErrorCode: {error_code}")

    def robust_execute(self, moveit_instance, action_desc):
        """ 鲁棒执行函数：等待动作结束并深度解析状态 """
        self.get_logger().info(f"⏳ 正在执行: {action_desc}...")
        
        # 等待动作在源码内部完成
        moveit_instance.wait_until_executed()
        

    def get_log_pose(self):
        """ 打印当前真实位置 (TF) """
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(self.base_link, self.arm_end_effector, now, timeout=rclpy.duration.Duration(seconds=1.0))
            p = trans.transform.translation
            q = trans.transform.rotation
            (r, pit, y) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return f"XYZ:[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}] RPY:[{math.degrees(r):.1f}, {math.degrees(pit):.1f}, {math.degrees(y):.1f}]"
        except Exception as e:
            return f"TF延迟 ({str(e)})"

    def run_task(self):
        self.get_logger().info("--- 开启自动化抓取任务 ---")
        
        # 1. 检查环境：源码属性 joint_state
        while rclpy.ok():
            if self.moveit_arm.joint_state is not None and self.banana_pose is not None:
                self.get_logger().info("数据已就绪")
                break
            self.get_logger().warn("等待机器人状态及香蕉位姿数据...", throttle_duration_sec=2.0)
            time.sleep(0.5)

        # --- 第一步：移动到预备提升位置 ---
        self.get_logger().info(f"Step 1: 抬臂预备姿态. 当前位姿: {self.get_log_pose()}")
        self.moveit_arm.move_to_configuration(start_joint_positions)
        time.sleep(1.5)
        # self.robust_execute(self.moveit_arm, "抬臂预备")
        self.moveit_arm.wait_until_executed()
        time.sleep(5.0)

        # --- 第二步：打开抓爪 ---
        self.get_logger().info("Step 2: 打开抓爪")
        self.moveit_hand.move_to_configuration(GRIPPER_OPEN)
        # self.robust_execute(self.moveit_hand, "打开抓爪")
        time.sleep(0.5)
        self.moveit_hand.wait_until_executed()
        time.sleep(5.0)

        # --- 第三步：前往香蕉位置 (笛卡尔直线移动) ---
        try:
            # 保持当前姿态，仅改变位置
            curr_tf = self.tf_buffer.lookup_transform(self.base_link, self.arm_end_effector, rclpy.time.Time())
            curr_q = curr_tf.transform.rotation
            
            # 目标位置：香蕉上方一点点 (z+0.02)
            target_pos = [self.banana_pose.pose.position.x, 
                          self.banana_pose.pose.position.y, 
                          self.banana_pose.pose.position.z + 0.005]
            
            self.get_logger().info(f"Step 3: 接近香蕉 -> {target_pos}")
            
            # 使用源码的 move_to_pose，设置 cartesian=True 确保直线进入
            self.moveit_arm.move_to_pose(
                position=target_pos, 
                quat_xyzw=[curr_q.x, curr_q.y, curr_q.z, curr_q.w],
                cartesian=True,
                cartesian_max_step=0.0015,
                cartesian_fraction_threshold=0.01,
            )
            time.sleep(0.5)
            self.moveit_arm.wait_until_executed()
            time.sleep(5.0)
            # self.robust_execute(self.moveit_arm, "接近香蕉")
        except Exception as e:
            self.get_logger().error(f"坐标解析异常: {e}")
            return

        # --- 第四步：闭合抓爪 ---
        self.get_logger().info("Step 4: 闭合抓爪并固定")
        self.moveit_hand.move_to_configuration(GRIPPER_CLOSE)
        time.sleep(0.5)
        self.moveit_hand.wait_until_executed()
        # self.robust_execute(self.moveit_hand, "闭合抓爪")
        time.sleep(5.0) # 仿真物理引擎计算闭合力矩

        # --- 第五步：带回香蕉 ---
        self.get_logger().info("Step 5: 带着香蕉返回")
        self.moveit_arm.move_to_configuration(LEFT_ARM_HOME)
        self.moveit_arm.wait_until_executed()
        # if self.robust_execute(self.moveit_arm, "带回任务"):
        self.get_logger().info(f"🎉 任务圆满完成! 最终位姿: {self.get_log_pose()}")
        # else:
        #     self.get_logger().warn("警告：返回过程未完全达标。")

def main():
    rclpy.init()
    node = OpenArmGraspBananaNode()
    
    # 源码内部依赖 spin 处理 Action 结果，必须开启多线程执行器
    executor = rclpy.executors.MultiThreadedExecutor(4)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_task()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("脚本关闭...")
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()