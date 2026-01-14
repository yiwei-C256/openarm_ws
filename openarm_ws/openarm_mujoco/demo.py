import os
# 适配WSLg的EGL渲染（核心配置）
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["DISPLAY"] = ":0"
os.environ["WAYLAND_DISPLAY"] = "wayland-1"

import genesis as gs

# 初始化Genesis
gs.init(backend=gs.cpu)

# 开启可视化窗口
scene = gs.Scene(show_viewer=True)  

# 添加实体
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='/home/ldp/openarm_ws/openarm_mujoco/v1/scene.xml'),
)

# 构建并运行
scene.build()
print("仿真界面已打开，运行1000步后关闭...")
for i in range(1000):
    scene.step()
    if i % 100 == 0:
        print(f"已运行 {i}/1000 步")

scene.close()
print("仿真结束！")