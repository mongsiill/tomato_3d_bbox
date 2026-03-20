import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. ZED ROS 2 Wrapper Launch 설정 (ZED 2i 모델 지정)
    try:
        zed_wrapper_dir = get_package_share_directory('zed_wrapper')
        zed_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(zed_wrapper_dir, 'launch', 'zed_camera.launch.py')
            ),
            launch_arguments={
                'camera_model': 'zed2i'
            }.items()
        )
    except Exception:
        # 만약 zed_wrapper가 안 깔려있을 때 런치 파일 전체가 죽는 걸 방지
        zed_launch = TimerAction(period=0.1, actions=[]) 

    # 2. 우리가 만든 FastSAM + FFS + 3D BBox 메인 노드
    sam_ffs_node = Node(
        package='tomato_3D',
        executable='fastsam_3d_node',
        name='fastsam_3d_node',
        output='screen'
    )

    # 3. [추가됨!] 2D BBox 초기화 퍼블리셔 노드
    annotation_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='tomato_3D',
                executable='annotation_box2d_publisher',
                name='annotation_box2d_publisher',
                output='screen'
            )
        ]
    )


    # 4. RViz2 자동 실행 (ZED와 다른 노드들이 켜질 시간을 벌기 위해 3초 지연)
    rviz_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen'
            )
        ]
    )

    # 리스트에 담긴 순서대로 실행됩니다!
    return LaunchDescription([
        zed_launch,
        sam_ffs_node,
        annotation_node,  # 방아쇠 장착!
        rviz_node
    ])