#!/usr/bin/env python3
import os
import signal
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class ZedFrameRecorder(Node):
    def __init__(self):
        super().__init__('zed_frame_recorder')

        # 파라미터 (ZED Wrapper 토픽에 맞춤)
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('camera_pose_topic', '/zed/zed_node/pose')
        self.declare_parameter('output_dir', 'data')
        self.declare_parameter('save_rate', 0.0)  # 0이면 키 입력/서비스로만 저장

        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.save_rate = self.get_parameter('save_rate').get_parameter_value().double_value
        self.camera_pose_topic = self.get_parameter('camera_pose_topic').get_parameter_value().string_value

        # 디렉토리 준비
        self.rgb_dir = os.path.join(self.output_dir, 'rgb')
        self.depth_dir = os.path.join(self.output_dir, 'depth')
        self.info_dir = os.path.join(self.output_dir, 'info')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.saved_first_frame = False
        self.last_rgb = None
        self.last_rgb_header = None
        self.last_depth = None
        self.last_depth_header = None
        self.last_camera_info = None
        self.last_pose = None 

        # ZED 센서 토픽과 맞추기 위한 QoS 설정
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # 구독
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.pose_sub = self.create_subscription(          # 추가
            PoseStamped, self.camera_pose_topic, self.pose_callback, sensor_qos
        )

        # save_rate > 0 이면 주기적으로 저장
        if self.save_rate > 0.0:
            period = 1.0 / self.save_rate
            self.timer = self.create_timer(period, self.timer_callback)
        else:
            self.timer = None

        self.frame_idx = 0
        self.get_logger().info('ZedFrameRecorder 노드 시작')

    def rgb_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_rgb = cv_img
            self.last_rgb_header = msg.header
        except Exception as e:
            self.get_logger().error(f'RGB 변환 실패: {e}')

    def depth_callback(self, msg: Image):
        try:
            # ZED depth는 통상 float32 (32FC1)
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.last_depth = cv_depth
            self.last_depth_header = msg.header
        except Exception as e:
            self.get_logger().error(f'Depth 변환 실패: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        self.last_camera_info = msg

    def pose_callback(self, msg: PoseStamped):
        self.last_pose = msg

    def timer_callback(self):
        # save_rate 로 주기적으로 저장하되 첫 프레임만 저장
        if not self.saved_first_frame:
            self.save_current_frame()

    def save_current_frame(self):
        # 이미 첫 프레임을 저장했다면 아무 것도 하지 않음
        if self.saved_first_frame:
            return

        if self.last_rgb is None or self.last_depth is None:
            self.get_logger().warn('RGB 또는 Depth 데이터가 아직 없음, 저장 생략')
            return

        # timestamp 기반 이름 or index 기반
        self.frame_idx += 1
        idx_str = f'{self.frame_idx:06d}'

        rgb_path = os.path.join(self.rgb_dir, f'frame_{idx_str}.png')
        depth_path = os.path.join(self.depth_dir, f'frame_{idx_str}.npy')
        info_path = os.path.join(self.info_dir, f'frame_{idx_str}.npz')

        # RGB 저장
        cv2.imwrite(rgb_path, self.last_rgb)

        # Depth 저장 (numpy)
        np.save(depth_path, self.last_depth)

        # CameraInfo + 헤더 정보도 같이 npz 로 저장
        if self.last_camera_info is not None:
            if getattr(self, "last_pose", None) is not None:
                np.savez(
                    info_path,
                    header_stamp_sec=self.last_rgb_header.stamp.sec,
                    header_stamp_nanosec=self.last_rgb_header.stamp.nanosec,
                    frame_id=self.last_rgb_header.frame_id,
                    K=np.array(self.last_camera_info.k).reshape(3, 3),
                    D=np.array(self.last_camera_info.d),
                    width=self.last_camera_info.width,
                    height=self.last_camera_info.height,
                    pose_position=np.array([
                        self.last_pose.pose.position.x,
                        self.last_pose.pose.position.y,
                        self.last_pose.pose.position.z,
                    ]),
                    pose_orientation=np.array([
                        self.last_pose.pose.orientation.x,
                        self.last_pose.pose.orientation.y,
                        self.last_pose.pose.orientation.z,
                        self.last_pose.pose.orientation.w,
                    ]),
                )
            else:
                np.savez(
                    info_path,
                    header_stamp_sec=self.last_rgb_header.stamp.sec,
                    header_stamp_nanosec=self.last_rgb_header.stamp.nanosec,
                    frame_id=self.last_rgb_header.frame_id,
                    K=np.array(self.last_camera_info.k).reshape(3, 3),
                    D=np.array(self.last_camera_info.d),
                    width=self.last_camera_info.width,
                    height=self.last_camera_info.height
                )
        else:
            np.savez(
                info_path,
                header_stamp_sec=self.last_rgb_header.stamp.sec,
                header_stamp_nanosec=self.last_rgb_header.stamp.nanosec,
                frame_id=self.last_rgb_header.frame_id
            )

        self.saved_first_frame = True
        self.get_logger().info(f'첫 프레임 {idx_str} 저장 완료, 노드 종료')

        # 첫 프레임 저장 후 노드 종료 요청
        rclpy.shutdown()

    def save_on_signal(self, signum, frame):
        # Ctrl+\ 등으로 수동 저장하고 싶을 때 signal 에 연결 가능
        self.get_logger().info('신호 수신: 현재 프레임 저장 시도')
        self.save_current_frame()


def main(args=None):
    rclpy.init(args=args)
    node = ZedFrameRecorder()

    # 예시: SIGUSR1 로 수동 저장
    try:
        signal.signal(signal.SIGUSR1, node.save_on_signal)
    except Exception:
        pass

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, 종료')
    finally:
        node.destroy_node()
        # 노드 내부에서 이미 rclpy.shutdown() 을 호출했을 수 있으므로 상태 확인
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()