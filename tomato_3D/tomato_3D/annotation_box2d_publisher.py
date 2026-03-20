#!/usr/bin/env python3
import os
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Polygon, Point32
from std_msgs.msg import Empty


class AnnotationBox2DPublisher(Node):
    def __init__(self):
        super().__init__('annotation_box2d_publisher')

        # 파라미터
        self.declare_parameter(
            'annotation_path',
            '/home/user/projects/3d_bb/data/annotations/frame_000001.json'
        )
        self.declare_parameter('box2d_topic', '/tomato/box2d')
        self.declare_parameter('ack_topic', '/tomato/box2d_ack')

        annotation_path = self.get_parameter('annotation_path').get_parameter_value().string_value
        box2d_topic = self.get_parameter('box2d_topic').get_parameter_value().string_value
        ack_topic = self.get_parameter('ack_topic').get_parameter_value().string_value

        self.box_pub = self.create_publisher(Polygon, box2d_topic, 10)
        self.ack_sub = self.create_subscription(Empty, ack_topic, self.ack_callback, 10)

        # 한 번만 publish 하고 끝낼 거라서, 타이머에서 바로 발행
        self.annotation_path = annotation_path
        self.published = False
        self.received_ack = False

        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('AnnotationBox2DPublisher 시작')

    def timer_callback(self):
        if self.published:
            return

        if not os.path.exists(self.annotation_path):
            self.get_logger().error(f'Annotation 파일 없음: {self.annotation_path}')
            return

        with open(self.annotation_path, 'r') as f:
            data = json.load(f)

        if 'boxes' not in data or len(data['boxes']) == 0:
            self.get_logger().error('boxes 항목이 비어 있음')
            return

        # boxes[0] 이 [x_min, y_min, x_max, y_max] 리스트
        x_min, y_min, x_max, y_max = data['boxes'][0]
        x_min = float(x_min)
        y_min = float(y_min)
        x_max = float(x_max)
        y_max = float(y_max)

        poly = Polygon()
        # 시계방향 4점
        poly.points.append(Point32(x=x_min, y=y_min, z=0.0))
        poly.points.append(Point32(x=x_max, y=y_min, z=0.0))
        poly.points.append(Point32(x=x_max, y=y_max, z=0.0))
        poly.points.append(Point32(x=x_min, y=y_max, z=0.0))

        self.box_pub.publish(poly)
        self.published = True
        self.get_logger().info(f'박스 1개 발행 완료: {self.annotation_path}')

    def ack_callback(self, msg: Empty):
        self.get_logger().info('box2d_ack 수신, 노드 종료')
        self.received_ack = True
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AnnotationBox2DPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()