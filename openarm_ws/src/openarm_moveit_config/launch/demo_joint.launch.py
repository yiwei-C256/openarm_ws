from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch

def generate_launch_description():
    # 1. 构建 MoveIt 配置
    moveit_config = MoveItConfigsBuilder(
        "openarm", 
        package_name="openarm_moveit_config"
    ).to_moveit_configs()

    # 2. 获取标准的 MoveIt demo launch 描述对象
    # 它包含了 move_group, rviz, robot_state_publisher 等
    launch_description = generate_demo_launch(moveit_config)

    # 3. 定义你自己的节点
    # 假设你的可执行文件在 setup.py/CMakeLists 中命名的名字是 joint_state_subscriber
    custom_node = Node(
        package='openarm_joint_state_pkg',
        executable='joint_state_subscriber',
        name='openarm_mujoco_sync_node',
        output='screen',
        emulate_tty=True,  # 新增：模拟终端输出，能让日志更及时显示
        # 如果你在用 MuJoCo 仿真，建议开启模拟时间
        parameters=[{'use_sim_time': True}]
    )

    # 4. 将你的节点添加到启动序列中
    launch_description.add_action(custom_node)

    return launch_description