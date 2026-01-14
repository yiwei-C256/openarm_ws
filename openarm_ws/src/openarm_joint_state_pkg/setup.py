from setuptools import find_packages, setup

package_name = 'openarm_joint_state_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'sensor_msgs', 'numpy', 'mujoco', 'glfw'],
    zip_safe=False,
    maintainer='ldp',
    maintainer_email='ldp@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_state_subscriber = openarm_joint_state_pkg.openarm_joint_state_subscriber:main',
        ],
    },
)
