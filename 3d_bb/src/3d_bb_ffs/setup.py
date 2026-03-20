import os            # <--- 이 두 줄을 
from glob import glob # <--- 추가해 줍니다.
from setuptools import find_packages, setup

package_name = '3d_bb_ffs'

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
            'sam_ffs_node = 3d_bb_ffs.sam_ffs_node:main',
            'fastsam_ffs_node = 3d_bb_ffs.fastsam_ffs_node:main'
        ],
    },
)
