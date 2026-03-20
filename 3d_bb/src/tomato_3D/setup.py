from setuptools import find_packages, setup
import os            # <--- 이 두 줄을 
from glob import glob # <--- 추가해 줍니다.

package_name = 'tomato_3D'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'zed_frame_recorder = tomato_3D.zed_frame_recorder:main',
            'fastsam_3d_node = tomato_3D.fastsam_3d_node:main',
            'sam2_3d_node = tomato_3D.sam2_3d_node:main',
            'annotation_box2d_publisher = tomato_3D.annotation_box2d_publisher:main',
        ],
    }
)
