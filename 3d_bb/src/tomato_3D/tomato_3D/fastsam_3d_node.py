#!/usr/bin/env python3

#FastSAM 사용
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Polygon  # 2D box 표현용
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import numpy as np
import cv2

try:
    import open3d as o3d
except Exception:
    # Open3D가 설치되어 있어도, numpy/scipy/sklearn ABI 충돌 등으로 import 단계에서
    # ImportError가 아닌 예외가 발생할 수 있어 안전하게 비활성화한다.
    o3d = None

import torch
from ultralytics import FastSAM, YOLO


class FastSam3DNode(Node):
    def __init__(self):
        super().__init__('fastsam_3d_node')

        # Ultralytics SAM/SAM2 설정
        self.declare_parameter('sam_model_path', '/home/user/projects/Tomato_3DBoundingBox/3d_bb/src/tomato_3D/resource/FastSAM-s.pt')
        self.declare_parameter('sam_device', 'cuda')  # auto / cpu / cuda
        self.declare_parameter('sam_imgsz', 1024)

        self.sam_model_path = self.get_parameter('sam_model_path').get_parameter_value().string_value
        self.sam_device_param = self.get_parameter('sam_device').get_parameter_value().string_value
        self.sam_imgsz = self.get_parameter('sam_imgsz').get_parameter_value().integer_value

        if self.sam_device_param == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.sam_device_param

        try:
            self.sam_model = FastSAM(self.sam_model_path)
            self.get_logger().info(
                f'Ultralytics FastSAM loaded: {self.sam_model_path} on {self.device}'
            )
        except Exception as e:
            self.sam_model = None
            self.get_logger().error(f'FastSAM 로드 실패: {e}')

        # YOLO tracking 설정 (detect+track)
        self.declare_parameter('tracking_mode', 'yolo')  # yolo | csrt
        self.declare_parameter('yolo_model_path', '/home/user/projects/Tomato_3DBoundingBox/3d_bb/src/tomato_3D/resource/best.pt')
        self.declare_parameter('yolo_tracker', 'bytetrack.yaml')  # bytetrack.yaml | botsort.yaml
        self.declare_parameter('yolo_conf', 0.25)
        self.declare_parameter('yolo_iou', 0.45)
        self.declare_parameter('yolo_classes', [])  # 예: [0] 처럼 클래스 필터. 비우면 전체.

        self.tracking_mode = self.get_parameter('tracking_mode').get_parameter_value().string_value
        self.yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_tracker = self.get_parameter('yolo_tracker').get_parameter_value().string_value
        self.yolo_conf = float(self.get_parameter('yolo_conf').get_parameter_value().double_value)
        self.yolo_iou = float(self.get_parameter('yolo_iou').get_parameter_value().double_value)
        # classes는 ParameterValue 타입 제약이 있어 string_array로 받는 경우가 많아서, 여기서는 단순하게 미사용 기본
        self.yolo_classes = None

        self.yolo_model = None
        self.target_track_id = None
        self.target_bbox_xyxy = None  # last bbox in xyxy

        if self.tracking_mode == 'yolo':
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info(f'YOLO loaded: {self.yolo_model_path} tracker={self.yolo_tracker}')
            except Exception as e:
                self.yolo_model = None
                self.get_logger().error(f'YOLO 로드 실패: {e}')

        # ZED 토픽에 맞는 QoS (센서 데이터)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.bridge = CvBridge()
        self.last_camera_info = None
        self.last_box2d = None  # 초기 bbox(1회 입력) 또는 재초기화용
        self.last_rgb = None
        self.last_depth = None

        # OpenCV CSRT tracker 상태
        self.tracker = None
        self.tracker_initialized = False
        self.tracker_bbox_xywh = None  # (x, y, w, h) in pixels
        self.declare_parameter('csrt_min_size', 80)
        self.declare_parameter('csrt_init_expand', 2.0)
        self.csrt_min_size = int(self.get_parameter('csrt_min_size').get_parameter_value().integer_value)
        self.csrt_init_expand = float(self.get_parameter('csrt_init_expand').get_parameter_value().double_value)

        # 파라미터: 기본값을 ZED Wrapper 기준으로 둠
        self.declare_parameter('rgb_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info')
        # 초기 bbox 입력(한 번만). 필요하면 재발행해서 tracker를 재초기화 가능
        self.declare_parameter('init_box2d_topic', '/tomato/box2d')
        self.declare_parameter('track_box2d_topic', '/tomato/track_box2d')
        self.declare_parameter('bbox3d_topic', '/tomato/box3d')
        self.declare_parameter('use_csrt_tracker', True)  # tracking_mode=csrt 일 때만 의미 있음

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        init_box2d_topic = self.get_parameter('init_box2d_topic').get_parameter_value().string_value
        track_box2d_topic = self.get_parameter('track_box2d_topic').get_parameter_value().string_value
        bbox3d_topic = self.get_parameter('bbox3d_topic').get_parameter_value().string_value
        self.use_csrt_tracker = self.get_parameter('use_csrt_tracker').get_parameter_value().bool_value

        # 구독
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, sensor_qos
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )
        self.init_box2d_sub = self.create_subscription(
            Polygon, init_box2d_topic, self.box2d_callback, 10
        )

        # 발행
        self.bbox3d_pub = self.create_publisher(Marker, bbox3d_topic, 10)
        self.track_box2d_pub = self.create_publisher(Polygon, track_box2d_topic, 10)

        # mask 발행
        self.declare_parameter('mask_topic', '/tomato/mask')
        self.declare_parameter('mask_overlay_topic', '/tomato/mask_overlay')

        mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        mask_overlay_topic = self.get_parameter('mask_overlay_topic').get_parameter_value().string_value

        self.mask_pub = self.create_publisher(Image, mask_topic, 10)
        self.mask_overlay_pub = self.create_publisher(Image, mask_overlay_topic, 10)

        # === [추가된 부분] 부드러운 3D 박스를 위한 스무딩 및 프레임 드랍 설정 ===
        self.declare_parameter('ema_alpha', 0.6) # 0.0 ~ 1.0 (낮을수록 부드럽지만 반응이 느림, 높을수록 반응이 빠름)
        self.ema_alpha = float(self.get_parameter('ema_alpha').get_parameter_value().double_value)
        self.smoothed_center = None
        self.smoothed_size = None
        self.is_processing = False  # 연산 밀림 방지용 락

        self.get_logger().info('FastSam3DNode 시작')

    # --- 콜백들 ---

    def camera_info_callback(self, msg: CameraInfo):
        self.last_camera_info = msg

    def box2d_callback(self, msg: Polygon):
        # Polygon 점 4개를 [x_min, y_min, x_max, y_max] 로 변환한다고 가정
        # 새 bbox가 들어오면 tracker를 재초기화할 수 있도록 저장
        self.last_box2d = msg
        self.tracker_initialized = False
        # YOLO tracking도 초기 bbox가 바뀌면 target을 다시 잡도록 리셋
        self.target_track_id = None
        self.target_bbox_xyxy = None

        # === [추가된 부분] 타겟이 바뀌면 스무딩 값도 리셋 ===
        self.smoothed_center = None
        self.smoothed_size = None

    def rgb_callback(self, msg: Image):
        if self.is_processing:
            return

        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f'RGB 변환 실패: {e}')
            return

        # RGB 도착 시마다 처리 시도
        self.is_processing = True
        try:
            self.try_process(msg.header, img_bgr)
        finally:
            self.is_processing = False  # 연산 종료 후 락 해제

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.last_depth = depth
        except Exception as e:
            self.get_logger().error(f'Depth 변환 실패: {e}')
            return

    # --- 메인 처리 ---

    def try_process(self, header, img_bgr):
        if self.last_rgb is None or self.last_depth is None:
            return
        if self.last_camera_info is None or self.last_box2d is None:
            return

        # 1) 현재 프레임에서 사용할 bbox를 결정 (YOLO track 우선, 아니면 CSRT, 아니면 초기 bbox)
        x_min, y_min, x_max, y_max = self.get_current_bbox(img_bgr)
        if x_min is None:
            return

        # 3-1) 추적 bbox 발행
        self.track_box2d_pub.publish(self.xyxy_to_polygon(x_min, y_min, x_max, y_max))

        # 4) SAM/FastSAM 으로 mask 생성 (bbox prompt)
        mask = self.run_fastsam(self.last_rgb, x_min, y_min, x_max, y_max)

        # 5) mask 시각화 토픽 발행
        self.publish_mask_topics(mask, self.last_rgb, header)

        # 6) mask + depth + CameraInfo 로 3D 포인트 & Bounding Box 계산 (카메라 프레임)
        bbox_center, bbox_size = self.compute_3d_bbox(mask, self.last_depth, self.last_camera_info)

        # === [추가된 부분] EMA(지수 이동 평균) 필터 적용 - 덜덜거림 완벽 방지 ===
        # 의미 있는 값이 나왔을 때만 필터 적용
        if np.any(bbox_size > 0):
            if self.smoothed_center is None:
                self.smoothed_center = bbox_center
                self.smoothed_size = bbox_size
            else:
                self.smoothed_center = self.ema_alpha * bbox_center + (1.0 - self.ema_alpha) * self.smoothed_center
                self.smoothed_size = self.ema_alpha * bbox_size + (1.0 - self.ema_alpha) * self.smoothed_size
            
            # 부드러워진 값으로 덮어쓰기
            bbox_center = self.smoothed_center
            bbox_size = self.smoothed_size
        else:
            return # 박스 계산 실패 시 프레임 스킵

        # 4) Marker 발행 (카메라 프레임 기준)
        marker = Marker()
        marker.header = header
        # camera_info.header.frame_id 를 그대로 사용 (예: zed_camera_center)
        marker.header.frame_id = "zed_left_camera_optical_frame"

        marker.ns = "tomato"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(bbox_center[0])
        marker.pose.position.y = float(bbox_center[1])
        marker.pose.position.z = float(bbox_center[2])
        marker.pose.orientation.w = 1.0  # 카메라 프레임 축 정렬

        marker.scale.x = float(bbox_size[0])
        marker.scale.y = float(bbox_size[1])
        marker.scale.z = float(bbox_size[2])

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        self.bbox3d_pub.publish(marker)
        # 디버그용 로그
        # self.get_logger().info(f'BBox center={bbox_center}, size={bbox_size}')

    def get_current_bbox(self, img_bgr):
        """
        tracking_mode에 따라 현재 프레임의 bbox(xyxy)를 반환.
        실패 시 None 반환.
        """
        img_h, img_w = img_bgr.shape[:2]

        # YOLO detect+track
        if self.tracking_mode == 'yolo' and self.yolo_model is not None:
            x_min0, y_min0, x_max0, y_max0 = self.parse_box2d(self.last_box2d)
            init_xyxy = np.array([x_min0, y_min0, x_max0, y_max0], dtype=float)

            results = self.yolo_model.track(
                source=img_bgr,
                persist=True,
                tracker=self.yolo_tracker,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                verbose=False,
                device=self.device,
            )
            if not results:
                return None, None, None, None

            r0 = results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                return None, None, None, None

            boxes_xyxy = r0.boxes.xyxy.cpu().numpy() if hasattr(r0.boxes.xyxy, "cpu") else np.asarray(r0.boxes.xyxy)
            ids = None
            if getattr(r0.boxes, "id", None) is not None:
                ids = r0.boxes.id.cpu().numpy().astype(int)

            # 1) 아직 target_track_id가 없으면, init bbox와 IoU가 가장 큰 detection을 선택
            if self.target_track_id is None:
                best_i = int(np.argmax([self.iou_xyxy(init_xyxy, b) for b in boxes_xyxy]))
                self.target_bbox_xyxy = boxes_xyxy[best_i].astype(float)
                if ids is not None and best_i < len(ids):
                    self.target_track_id = int(ids[best_i])
                return self.clamp_xyxy(*self.target_bbox_xyxy, img_w, img_h)

            # 2) target_track_id가 있으면 그 id를 찾고, 없으면 init bbox 근처 IoU로 재매칭
            if ids is not None:
                matches = np.where(ids == self.target_track_id)[0]
                if matches.size > 0:
                    b = boxes_xyxy[int(matches[0])].astype(float)
                    self.target_bbox_xyxy = b
                    return self.clamp_xyxy(*b, img_w, img_h)

            # id를 못 찾으면 IoU로 재선택
            best_i = int(np.argmax([self.iou_xyxy(init_xyxy, b) for b in boxes_xyxy]))
            b = boxes_xyxy[best_i].astype(float)
            self.target_bbox_xyxy = b
            if ids is not None and best_i < len(ids):
                self.target_track_id = int(ids[best_i])
            return self.clamp_xyxy(*b, img_w, img_h)

        # CSRT tracking (옵션)
        if self.tracking_mode == 'csrt' and self.use_csrt_tracker:
            if not self.tracker_initialized:
                x_min, y_min, x_max, y_max = self.parse_box2d(self.last_box2d)
                x, y, w, h = self.xyxy_to_xywh(x_min, y_min, x_max, y_max)
                x, y, w, h = self.clamp_xywh(x, y, w, h, img_w, img_h)
                ok = self.init_csrt_tracker(img_bgr, (x, y, w, h))
                if ok:
                    self.tracker_initialized = True
                    self.tracker_bbox_xywh = (x, y, w, h)
                else:
                    self.use_csrt_tracker = False
                    return self.parse_box2d(self.last_box2d)

            ok, bbox_xywh = self.tracker.update(img_bgr)
            if not ok:
                self.tracker_initialized = False
                return None, None, None, None
            x, y, w, h = [int(v) for v in bbox_xywh]
            x, y, w, h = self.clamp_xywh(x, y, w, h, img_w, img_h)
            self.tracker_bbox_xywh = (x, y, w, h)
            return self.xywh_to_xyxy(x, y, w, h)

        # fallback: 초기 bbox 그대로
        return self.parse_box2d(self.last_box2d)

    def clamp_xyxy(self, x_min, y_min, x_max, y_max, img_w, img_h):
        x_min = max(0, min(img_w - 1, int(round(x_min))))
        y_min = max(0, min(img_h - 1, int(round(y_min))))
        x_max = max(0, min(img_w - 1, int(round(x_max))))
        y_max = max(0, min(img_h - 1, int(round(y_max))))
        if x_max <= x_min:
            x_max = min(img_w - 1, x_min + 1)
        if y_max <= y_min:
            y_max = min(img_h - 1, y_min + 1)
        return x_min, y_min, x_max, y_max

    def iou_xyxy(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return float(inter / denom)

    def init_csrt_tracker(self, img_bgr, bbox_xywh):
        """
        OpenCV CSRT tracker 초기화.
        bbox_xywh: (x, y, w, h)
        """
        try:
            if hasattr(cv2, "TrackerCSRT_create"):
                self.tracker = cv2.TrackerCSRT_create()
            else:
                # OpenCV 빌드에 따라 legacy 네임스페이스를 쓰는 경우
                self.tracker = cv2.legacy.TrackerCSRT_create()
        except Exception as e:
            self.get_logger().error(f'CSRT tracker 생성 실패(opencv-contrib 필요): {e}')
            return False

        img_h, img_w = img_bgr.shape[:2]
        x, y, w, h = [int(v) for v in bbox_xywh]

        # CSRT는 너무 작은 박스에서 초기화가 잘 실패하므로, 1) 최소 크기 보장 2) 박스 확장
        x, y, w, h = self.expand_bbox_xywh(x, y, w, h, img_w, img_h, self.csrt_init_expand)
        w = max(self.csrt_min_size, w)
        h = max(self.csrt_min_size, h)
        x, y, w, h = self.clamp_xywh(x, y, w, h, img_w, img_h)

        self.get_logger().info(
            f'CSRT init image_size=({img_w}x{img_h}), bbox_xywh=({x},{y},{w},{h}), '
            f'min_size={self.csrt_min_size}, expand={self.csrt_init_expand}'
        )

        # OpenCV 빌드에 따라 bbox 파싱이 민감해서 순수 int 튜플로 전달
        ok = self.tracker.init(img_bgr, (int(x), int(y), int(w), int(h)))
        if not ok:
            # 한 번 더, 더 크게 확장해서 재시도
            x2, y2, w2, h2 = self.expand_bbox_xywh(x, y, w, h, img_w, img_h, 3.0)
            x2, y2, w2, h2 = self.clamp_xywh(x2, y2, w2, h2, img_w, img_h)
            self.get_logger().warn(f'CSRT init 재시도 bbox_xywh=({x2},{y2},{w2},{h2})')
            ok = self.tracker.init(img_bgr, (int(x2), int(y2), int(w2), int(h2)))
            if not ok:
                self.get_logger().error('CSRT tracker init 실패')
                return False

        self.get_logger().info(f'CSRT tracker initialized: ({x},{y},{w},{h})')
        return True

    def xyxy_to_xywh(self, x_min, y_min, x_max, y_max):
        x = int(x_min)
        y = int(y_min)
        w = int(max(1, x_max - x_min))
        h = int(max(1, y_max - y_min))
        return x, y, w, h

    def xywh_to_xyxy(self, x, y, w, h):
        x_min = int(x)
        y_min = int(y)
        x_max = int(x + w)
        y_max = int(y + h)
        return x_min, y_min, x_max, y_max

    def xyxy_to_polygon(self, x_min, y_min, x_max, y_max):
        poly = Polygon()
        poly.points.append(Point32(x=float(x_min), y=float(y_min), z=0.0))
        poly.points.append(Point32(x=float(x_max), y=float(y_min), z=0.0))
        poly.points.append(Point32(x=float(x_max), y=float(y_max), z=0.0))
        poly.points.append(Point32(x=float(x_min), y=float(y_max), z=0.0))
        return poly

    def clamp_xywh(self, x, y, w, h, img_w, img_h):
        x = max(0, min(img_w - 1, int(x)))
        y = max(0, min(img_h - 1, int(y)))
        w = max(1, int(w))
        h = max(1, int(h))
        if x + w > img_w:
            w = max(1, img_w - x)
        if y + h > img_h:
            h = max(1, img_h - y)
        return x, y, w, h

    def expand_bbox_xywh(self, x, y, w, h, img_w, img_h, scale):
        """
        (x,y,w,h)를 중심 기준으로 scale만큼 확장한다.
        """
        if scale is None or scale <= 1.0:
            return x, y, w, h
        cx = x + w / 2.0
        cy = y + h / 2.0
        nw = w * scale
        nh = h * scale
        nx = int(round(cx - nw / 2.0))
        ny = int(round(cy - nh / 2.0))
        return self.clamp_xywh(nx, ny, int(round(nw)), int(round(nh)), img_w, img_h)

    # --- 유틸 함수들 ---

    def parse_box2d(self, poly: Polygon):
        # Polygon의 point 4개에서 min/max 계산
        xs = [p.x for p in poly.points]
        ys = [p.y for p in poly.points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def run_fastsam(self, rgb_img, x_min, y_min, x_max, y_max):
        """
        Ultralytics SAM/SAM2 로 bbox prompt 기반 mask 생성
        반환: bool mask (H, W)
        """
        h, w = rgb_img.shape[:2]

        # SAM 모델이 없으면 더미 mask로 fallback
        if self.sam_model is None:
            self.get_logger().warn('SAM 모델이 없어 더미 mask 사용')
            mask = np.zeros((h, w), dtype=bool)
            x_min = max(0, min(w, int(x_min)))
            x_max = max(0, min(w, int(x_max)))
            y_min = max(0, min(h, int(y_min)))
            y_max = max(0, min(h, int(y_max)))
            mask[y_min:y_max, x_min:x_max] = True
            return mask

        try:
            bbox = [[int(x_min), int(y_min), int(x_max), int(y_max)]]

            results = self.sam_model.predict(
                source=rgb_img,
                bboxes=bbox,
                imgsz=self.sam_imgsz,
                device=self.device,
                verbose=False
            )

            if not results or results[0].masks is None:
                self.get_logger().warn('SAM 결과가 비어 있음')
                return np.zeros((h, w), dtype=bool)

            mask_data = results[0].masks.data  # shape: [N, H, W]

            if hasattr(mask_data, 'cpu'):
                mask_data = mask_data.cpu().numpy()

            if len(mask_data) == 0:
                self.get_logger().warn('SAM mask가 0개')
                return np.zeros((h, w), dtype=bool)

            mask = mask_data[0].astype(bool)

            # 혹시 해상도가 다르면 원본 크기로 맞춤
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            self.get_logger().info(f'SAM mask pixel count: {int(mask.sum())}')
            return mask

        except Exception as e:
            self.get_logger().error(f'SAM 실행 실패: {e}')
            return np.zeros((h, w), dtype=bool)

    def compute_3d_bbox(self, mask, depth, cam_info: CameraInfo):
        """
        mask: H x W bool (True 인 픽셀만 사용)
        depth: H x W float32 (m 단위 가정)
        cam_info: CameraInfo (K 사용)
        반환: (center[3], size[3])  모두 카메라 프레임 기준
        """
        h, w = depth.shape
        K = np.array(cam_info.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        valid = mask & np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        ys, xs = np.where(valid)
        zs = depth[ys, xs]

        xs_3d = (xs - cx) * zs / fx
        ys_3d = (ys - cy) * zs / fy

        points = np.stack([xs_3d, ys_3d, zs], axis=-1)
        return self.estimate_spherical_bbox(points)

    def estimate_spherical_bbox(self, points):
        """
        최소제곱법을 버리고 중앙값(Median) 기반으로 정확한 중심을 찾습니다.
        """
        if points.shape[0] < 6:
            return self.aabb_center_and_size(points)

        filtered = points
        if o3d is not None:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                nb_neighbors = min(20, max(3, points.shape[0] - 1))
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors,
                    std_ratio=2.0,
                )
                filtered = np.asarray(pcd.points)
            except Exception as e:
                self.get_logger().warn(f'Open3D outlier removal 실패, 원본 점 사용: {e}')
                filtered = points

        if filtered.shape[0] < 6:
            return self.aabb_center_and_size(filtered)

        # 1. [핵심 개선] 중앙값(Median)으로 오차 없는 중심점 찾기
        # 한쪽으로 치우친 줄기 데이터를 무시하고 덩어리의 정중앙을 타겟팅합니다.
        center = np.median(filtered, axis=0)

        # 2. 중심점에서 각 포인트까지의 거리 계산
        distances = np.linalg.norm(filtered - center, axis=1)

        # 3. 백분위수를 이용한 타이트한 반지름 추정
        # 거리가 먼 30%의 데이터(보통 줄기나 튀는 노이즈)를 잘라내고, 
        # 안쪽 70% 데이터를 감싸는 타이트한 크기를 반지름으로 설정합니다.
        radius = float(np.percentile(distances, 70.0))

        if radius <= 0.0:
            return self.aabb_center_and_size(filtered)

        diameter = 2.0 * radius
        size = np.array([diameter, diameter, diameter], dtype=float)
        return center.astype(float), size

    def robust_radius_from_distances(self, distances):
        """
        중심점으로부터의 거리 분포에서 이상치를 제거하고
        중앙값 기반으로 안정적인 반지름을 추정한다.
        """
        distances = distances[np.isfinite(distances) & (distances > 0.0)]
        if distances.size == 0:
            return None

        if distances.size >= 4:
            q1, q3 = np.percentile(distances, [25.0, 75.0])
            iqr = q3 - q1
            if iqr > 0.0:
                lower = max(0.0, q1 - 1.5 * iqr)
                upper = q3 + 1.5 * iqr
                inliers = distances[(distances >= lower) & (distances <= upper)]
                if inliers.size >= 3:
                    distances = inliers

        return float(np.median(distances))

    def aabb_center_and_size(self, points):
        """
        구 피팅이 실패했을 때 사용하는 안전한 AABB fallback.
        """
        if points.shape[0] == 0:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        p_min = points.min(axis=0)
        p_max = points.max(axis=0)
        center = (p_min + p_max) / 2.0
        size = (p_max - p_min)
        return center, size

    def publish_mask_topics(self, mask, rgb_img, header):
        """
        mask: bool (H, W)
        rgb_img: RGB numpy image (H, W, 3)
        header: 원본 이미지 header
        """
        try:
            # 1) mono8 mask 이미지
            mask_img = (mask.astype(np.uint8) * 255)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding='mono8')
            mask_msg.header.stamp = header.stamp
            mask_msg.header.frame_id = self.last_camera_info.header.frame_id
            self.mask_pub.publish(mask_msg)

            # 2) overlay 이미지
            overlay = rgb_img.copy()
            red = np.zeros_like(overlay)
            red[:, :, 0] = 255  # RGB 기준이면 R 채널, BGR이면 채널 주의

            alpha = 0.4
            overlay[mask] = (overlay[mask] * (1 - alpha) + red[mask] * alpha).astype(np.uint8)

            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding='bgr8')
            overlay_msg.header.stamp = header.stamp
            overlay_msg.header.frame_id = self.last_camera_info.header.frame_id
            self.mask_overlay_pub.publish(overlay_msg)

        except Exception as e:
            self.get_logger().error(f'Mask topic publish 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FastSam3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()