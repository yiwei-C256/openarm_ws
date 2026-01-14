#!/usr/bin/env python3
import sys
import numpy as np

# --- 核心补丁 ---
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener
from threading import Thread
import time
import math
from tf_transformations import euler_from_quaternion

class OpenArmMoveItPoseNode(Node):
    def __init__(self):
        super().__init__("openarm_moveit_pose_node")

        self.joint_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
        self.base_link = "world"
        self.end_effector = "openarm_left_hand_tcp"
        self.banana_pose = None # 存储订阅到的香蕉位姿

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            group_name="left_arm",
            joint_names=self.joint_names,
            base_link_name=self.base_link,
            end_effector_name=self.end_effector,
            callback_group=self.callback_group
        )

        # --- 4. 订阅香蕉位姿 ---
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            "/banana_pose", 
            self.banana_cb, 
            10, 
            callback_group=self.callback_group
        )

    def banana_cb(self, msg):
        self.banana_pose = msg

    def get_current_pose(self):
        """ 获取当前末端执行器的实时位姿 """
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_link, self.end_effector, now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return trans
        except Exception as e:
            self.get_logger().error(f"无法获取 TF 变换: {e}")
            return None
        
    def get_frame_pose(self, target_frame):
        """ 通用的 TF 获取函数 """
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                self.base_link, target_frame, now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            return trans
        except Exception:
            return None
        
    def format_pose_to_string(self, name, transform_or_msg):
        """ 格式化打印函数 """
        if transform_or_msg is None:
            return f"{name:<10} | 数据未就绪"
        
        if hasattr(transform_or_msg, 'transform'): # TF
            p = transform_or_msg.transform.translation
            q = transform_or_msg.transform.rotation
        else: # PoseStamped
            p = transform_or_msg.pose.position
            q = transform_or_msg.pose.orientation

        (r, pit, y) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return (f"{name:<10} | "
                f"Pos: [{p.x:6.3f}, {p.y:6.3f}, {p.z:6.3f}] | "
                f"RPY: [{math.degrees(r):6.1f}, {math.degrees(pit):6.1f}, {math.degrees(y):6.1f}]")
        
    def print_env_status(self):
        """ 打印快照 """
        arm_tf = self.get_frame_pose(self.end_effector)
        self.get_logger().info(
            f"\n"
            f"{'='*80}\n"
            f"📍 [ 当前环境位姿快照 ]\n"
            f"{'-'*80}\n"
            f"{self.format_pose_to_string('机械臂末端', arm_tf)}\n"
            f"{self.format_pose_to_string('香蕉(目标)', self.banana_pose)}\n"
            f"{'='*80}"
        )

    def run_task(self):
        self.get_logger().info("正在等待系统同步 (MoveIt + TF + 香蕉位姿)...")
        while rclpy.ok():
            if self.moveit2.joint_state and self.get_current_pose() and self.banana_pose:
                break
            time.sleep(1.0)

        # --- 步骤 1: 获取起始状态 ---
        start_tf = self.get_current_pose()
        curr_p = start_tf.transform.translation
        curr_q = start_tf.transform.rotation
        
        # --- 步骤 2: 提取香蕉坐标作为目标 ---
        # 目标位置 = 香蕉的 XYZ
        target_pos = [
            self.banana_pose.pose.position.x,
            self.banana_pose.pose.position.y,
            self.banana_pose.pose.position.z + 0.05  # 建议加一个 5cm 的 Z 偏移，防止直接撞击桌面
        ]
        # 目标姿态 = 保持当前末端姿态不变
        target_quat = [curr_q.x, curr_q.y, curr_q.z, curr_q.w]

        # 计算欧拉角用于打印对比
        (s_r, s_p, s_y) = euler_from_quaternion([curr_q.x, curr_q.y, curr_q.z, curr_q.w])
        (t_r, t_p, t_y) = euler_from_quaternion(target_quat)

        # 打印详细对比表
        self.get_logger().info(
            f"\n"
            f"📊 [ 抓取任务规划 ]\n"
            f"{'参数':<10} | {'当前末端':<25} | {'目标(香蕉中心)':<25}\n"
            f"{'-'*70}\n"
            f"{'X (m)':<10} | {curr_p.x:<25.4f} | {target_pos[0]:<25.4f}\n"
            f"{'Y (m)':<10} | {curr_p.y:<25.4f} | {target_pos[1]:<25.4f}\n"
            f"{'Z (m)':<10} | {curr_p.z:<25.4f} | {target_pos[2]:<25.4f}\n"
            f"{'-'*70}\n"
            f"{'Roll (°)':<10} | {math.degrees(s_r):<25.2f} | {math.degrees(t_r):<25.2f}\n"
            f"{'Pitch(°)':<10} | {math.degrees(s_p):<25.2f} | {math.degrees(t_p):<25.2f}\n"
            f"{'Yaw   (°)':<10} | {math.degrees(s_y):<25.2f} | {math.degrees(t_y):<25.2f}\n"
            f"{'='*70}"
        )

        # --- 步骤 3: 发送规划请求 ---
        self.get_logger().info("正在向 MoveIt 发送抓取路径规划...")
        success = self.moveit2.move_to_pose(
            position=target_pos,
            quat_xyzw=target_quat,
            cartesian=True,                 # 笛卡尔直线运动
            cartesian_max_step=0.01,
            cartesian_fraction_threshold=0.5
        )

        # if success is None:
        self.get_logger().info("规划成功，正在执行运动...")
        self.moveit2.wait_until_executed()
        time.sleep(2.0) # 等待仿真稳定

        # --- 步骤 4: 再次获取位姿并验证 ---
        final_tf = self.get_current_pose()
        if final_tf:
            final_p = final_tf.transform.translation
            
            # 计算与香蕉目标的欧氏距离误差
            dist = math.sqrt(
                (final_p.x - target_pos[0])**2 +
                (final_p.y - target_pos[1])**2 +
                (final_p.z - target_pos[2])**2
            )
            
            self.get_logger().info(
                f"\n[ 到达反馈 ]\n"
                f"最终坐标 (XYZ): {final_p.x:.4f}, {final_p.y:.4f}, {final_p.z:.4f}\n"
                f"目标坐标 (XYZ): {target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}\n"
                f"🎯 距离香蕉误差: {dist*1000:.2f} 毫米"
            )

            if dist < 0.01: # 1厘米容差
                self.get_logger().info("✅ 成功到达香蕉位置！")
            else:
                self.get_logger().warn("⚠️ 运动已停止，但距离香蕉仍有偏差。")
        # else:
        #     self.get_logger().error("❌ MoveIt 规划失败，无法到达香蕉位置。")

def main():
    rclpy.init()
    node = OpenArmMoveItPoseNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_task()
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()