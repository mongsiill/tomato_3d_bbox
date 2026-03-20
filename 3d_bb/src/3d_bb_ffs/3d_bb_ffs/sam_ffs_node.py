#!/usr/bin/env python3

# SAM2 및 Fast-FoundationStereo 적용 버전
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Polygon, Point32
from cv_bridge import CvBridge
import numpy as np
import cv2
import message_filters  # === [추가됨] 좌/우 이미지 동기화용 ===

try:
    import open3d as o3d
except Exception:
    o3d = None

import sys
import torch
from ultralytics import SAM, YOLO
import torch.nn.functional as F  # 패딩(Padding) 처리를 위해 필요

# === Fast-FoundationStereo 폴더 경로 강제 추가 ===
sys.path.append('/home/user/projects/Fast-FoundationStereo')
# 원작자가 만든 패딩 도구 임포트
from core.utils.utils import InputPadder

# === [주의] Fast-FoundationStereo Repo를 클론한 경로에 맞춰 import 하세요 ===
# sys.path.append('/home/user/projects/Fast-FoundationStereo')
# from models.stereo_model import FastFoundationStereo  <-- (가상의 import 예시)


class Sam3DNode(Node):
    def __init__(self):
        super().__init__('sam_3d_node')

        # === 1. 모델 설정 (SAM2 + Fast-FoundationStereo) ===
        self.declare_parameter('sam_model_path', '/home/user/projects/Tomato_3DBoundingBox/3d_bb/src/tomato_3D/resource/sam2.1_t.pt')
        self.declare_parameter('sam_device', 'cuda')
        self.declare_parameter('sam_imgsz', 1024)
        
        # Fast-FoundationStereo 가중치 파일 경로
        self.declare_parameter('stereo_model_path', '/home/user/projects/Fast-FoundationStereo/weights/20-26-39/model_best_bp2_serialize.pth')
        # ZED 2i 카메라의 두 렌즈 사이 실제 거리 (Baseline, 단위: 미터)
        self.declare_parameter('camera_baseline', 0.12)

        self.sam_device_param = self.get_parameter('sam_device').get_parameter_value().string_value
        self.device = 'cuda' if self.sam_device_param == 'auto' and torch.cuda.is_available() else self.sam_device_param
        self.sam_imgsz = self.get_parameter('sam_imgsz').get_parameter_value().integer_value
        self.camera_baseline = self.get_parameter('camera_baseline').get_parameter_value().double_value

        # SAM2 로드
        try:
            self.sam_model = SAM(self.get_parameter('sam_model_path').get_parameter_value().string_value)
            self.get_logger().info('Ultralytics SAM 2 loaded')
        except Exception as e:
            self.sam_model = None
            self.get_logger().error(f'SAM 2 로드 실패: {e}')

        # Fast-FoundationStereo 로드
        try:
            stereo_weights = self.get_parameter('stereo_model_path').get_parameter_value().string_value
            
            # run_demo.py 방식대로 통째로 로드
            self.stereo_model = torch.load(stereo_weights, map_location='cpu', weights_only=False)
            
            # 파라미터 세팅 (run_demo.py 기본값 적용)
            self.stereo_model.args.valid_iters = 8
            self.stereo_model.args.max_disp = 192
            
            self.stereo_model = self.stereo_model.to(self.device).eval()
            self.get_logger().info('Fast-FoundationStereo 로드 완료!')
        except Exception as e:
            self.stereo_model = None
            self.get_logger().error(f'Stereo Model 로드 실패: {e}')

        # === 2. YOLO 설정 ===
        self.declare_parameter('tracking_mode', 'yolo')
        self.declare_parameter('yolo_model_path', '/home/user/projects/Tomato_3DBoundingBox/3d_bb/src/tomato_3D/resource/best.pt')
        self.declare_parameter('yolo_tracker', 'bytetrack.yaml')
        self.declare_parameter('yolo_conf', 0.25)
        self.declare_parameter('yolo_iou', 0.45)

        self.tracking_mode = self.get_parameter('tracking_mode').get_parameter_value().string_value
        self.yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.yolo_tracker = self.get_parameter('yolo_tracker').get_parameter_value().string_value
        self.yolo_conf = float(self.get_parameter('yolo_conf').get_parameter_value().double_value)
        self.yolo_iou = float(self.get_parameter('yolo_iou').get_parameter_value().double_value)

        self.yolo_model = None
        self.target_track_id = None
        self.target_bbox_xyxy = None

        if self.tracking_mode == 'yolo':
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
                self.get_logger().info('YOLO loaded')
            except Exception as e:
                self.yolo_model = None
                self.get_logger().error(f'YOLO 로드 실패: {e}')

        # === 3. 통신 및 상태 변수 ===
        self.bridge = CvBridge()
        self.last_box2d = None
        
        self.declare_parameter('ema_alpha', 0.6)
        self.ema_alpha = float(self.get_parameter('ema_alpha').get_parameter_value().double_value)
        self.smoothed_center = None
        self.smoothed_size = None
        self.is_processing = False

        # === 4. ROS 2 토픽 변경 (좌/우 이미지 구독) ===
        # ZED에서 뎁스를 받지 않고, Rectified 된 왼쪽/오른쪽 이미지를 직접 받습니다.
        self.declare_parameter('left_topic', '/zed/zed_node/left/color/rect/image')
        self.declare_parameter('right_topic', '/zed/zed_node/right/color/rect/image')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/left/color/rect/camera_info')
        self.declare_parameter('init_box2d_topic', '/tomato/box2d')
        self.declare_parameter('track_box2d_topic', '/tomato/track_box2d')
        self.declare_parameter('bbox3d_topic', '/tomato/box3d')

        left_topic = self.get_parameter('left_topic').get_parameter_value().string_value
        right_topic = self.get_parameter('right_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        init_box2d_topic = self.get_parameter('init_box2d_topic').get_parameter_value().string_value
        
        # Bbox 및 발행
        self.init_box2d_sub = self.create_subscription(Polygon, init_box2d_topic, self.box2d_callback, 10)
        self.bbox3d_pub = self.create_publisher(Marker, self.get_parameter('bbox3d_topic').get_parameter_value().string_value, 10)
        self.track_box2d_pub = self.create_publisher(Polygon, self.get_parameter('track_box2d_topic').get_parameter_value().string_value, 10)

        self.declare_parameter('mask_topic', '/tomato/mask')
        self.declare_parameter('mask_overlay_topic', '/tomato/mask_overlay')
        self.mask_pub = self.create_publisher(Image, self.get_parameter('mask_topic').get_parameter_value().string_value, 10)
        self.mask_overlay_pub = self.create_publisher(Image, self.get_parameter('mask_overlay_topic').get_parameter_value().string_value, 10)

        self.depth_vis_pub = self.create_publisher(Image, '/tomato/depth_vis', 10)

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        # === [핵심] 좌우 카메라 동기화 (TimeSynchronizer) ===
        self.left_sub = message_filters.Subscriber(self, Image, left_topic, qos_profile=sensor_qos)
        self.right_sub = message_filters.Subscriber(self, Image, right_topic, qos_profile=sensor_qos)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic)

        # 왼쪽, 오른쪽 이미지와 CameraInfo의 timestamp가 같은 것들만 묶어서 stereo_callback으로 보냅니다.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.info_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.stereo_callback)

        self.get_logger().info('Sam3DNode 시작 (Fast-FoundationStereo 적용 모드)')

    # --- 콜백들 ---

    def box2d_callback(self, msg: Polygon):
        self.last_box2d = msg
        self.target_track_id = None
        self.target_bbox_xyxy = None
        self.smoothed_center = None
        self.smoothed_size = None

    def stereo_callback(self, left_msg: Image, right_msg: Image, info_msg: CameraInfo):
        """
        좌/우 이미지와 카메라 정보가 동시에 들어올 때 실행됩니다.
        """
        if self.is_processing:
            return

        self.is_processing = True
        try:
            # 1. ROS 이미지를 CV2 (BGR -> RGB)로 변환
            left_bgr = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_bgr = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

            # 메인 처리 로직 호출
            self.try_process(left_msg.header, left_rgb, right_rgb, info_msg)

        except Exception as e:
            self.get_logger().error(f'Stereo callback 에러: {e}')
        finally:
            self.is_processing = False

    # --- 메인 처리 ---

    def try_process(self, header, left_rgb, right_rgb, cam_info: CameraInfo):
        if self.last_box2d is None:
            return

        # 1) YOLO 추적으로 현재 프레임(왼쪽 카메라 기준) 타겟 박스 찾기
        img_bgr_for_yolo = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        x_min, y_min, x_max, y_max = self.get_current_bbox(img_bgr_for_yolo)
        if x_min is None:
            return

        self.track_box2d_pub.publish(self.xyxy_to_polygon(x_min, y_min, x_max, y_max))

        # === 2) Fast-FoundationStereo 로 Depth Map 생성 ===
        depth_map = self.generate_depth_from_stereo(left_rgb, right_rgb, cam_info)

        if depth_map is None:
            self.get_logger().warn('Depth 맵 생성 실패')
            return

        # 3) SAM 2 로 mask 생성 (왼쪽 이미지 기준)
        mask = self.run_sam(left_rgb, x_min, y_min, x_max, y_max)
        self.publish_mask_topics(mask, left_rgb, header)

        # 4) mask + 딥러닝 depth + CameraInfo 로 3D Bounding Box 계산
        bbox_center, bbox_size = self.compute_3d_bbox(mask, depth_map, cam_info)

        self.get_logger().info(f'계산된 3D 박스 중심: {bbox_center}, 크기: {bbox_size}')

        # 5) EMA 스무딩 필터 적용
        if np.any(bbox_size > 0):
            if self.smoothed_center is None:
                self.smoothed_center = bbox_center
                self.smoothed_size = bbox_size
            else:
                self.smoothed_center = self.ema_alpha * bbox_center + (1.0 - self.ema_alpha) * self.smoothed_center
                self.smoothed_size = self.ema_alpha * bbox_size + (1.0 - self.ema_alpha) * self.smoothed_size
            
            bbox_center = self.smoothed_center
            bbox_size = self.smoothed_size
        else:
            return 

        # --- [여기에 디버깅/시각화 코드 추가!] ---
        # 뎁스맵의 평균 거리를 로그로 찍어봅니다 (0.1m ~ 10m 사이의 유효한 값만)
        valid_d = depth_map[(depth_map > 0.1) & (depth_map < 10.0)]
        mean_d = valid_d.mean() if len(valid_d) > 0 else 0.0
        self.get_logger().info(f'FFS 뎁스 생성 완료. 평균 거리: {mean_d:.2f}m')

        # RViz에서 눈으로 보기 위해 뎁스 맵을 정규화(0~255)하여 토픽으로 쏩니다
        disp_vis = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_vis_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
        depth_msg = self.bridge.cv2_to_imgmsg(disp_vis_color, encoding='bgr8')
        depth_msg.header.stamp = header.stamp
        depth_msg.header.frame_id = "zed_left_camera_optical_frame"
        self.depth_vis_pub.publish(depth_msg)
        # ----------------------------------------

        # 6) Marker 발행
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "zed_left_camera_optical_frame" # "zed_left_camera_optical_frame" 등

        marker.ns = "tomato"
        marker.id = 0
        marker.type = Marker.CUBE # 큐브 유지
        marker.action = Marker.ADD

        marker.pose.position.x = float(bbox_center[0])
        marker.pose.position.y = float(bbox_center[1])
        marker.pose.position.z = float(bbox_center[2])
        marker.pose.orientation.w = 1.0  

        marker.scale.x = float(bbox_size[0])
        marker.scale.y = float(bbox_size[1])
        marker.scale.z = float(bbox_size[2])

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        self.bbox3d_pub.publish(marker)

    def generate_depth_from_stereo(self, left_img, right_img, cam_info: CameraInfo):
        if self.stereo_model is None:
            return np.ones(left_img.shape[:2], dtype=np.float32)

        try:
            # 1. Numpy(H, W, C) -> PyTorch Tensor(B, C, H, W) 변환
            # run_demo.py를 보면 0~255 스케일을 그대로 float으로 캐스팅만 합니다.
            imgL = torch.as_tensor(left_img).float().to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
            imgR = torch.as_tensor(right_img).float().to(self.device).unsqueeze(0).permute(0, 3, 1, 2)

            # 2. 원작자의 InputPadder를 이용해 32의 배수로 자동 패딩
            padder = InputPadder(imgL.shape, divis_by=32, force_square=False)
            imgL, imgR = padder.pad(imgL, imgR)

            # 3. 모델 추론 (autocast로 속도 향상)
            # AMP_DTYPE은 보통 torch.float16 을 의미합니다.
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                disp = self.stereo_model.forward(
                    imgL, imgR, 
                    iters=8, 
                    test_mode=True, 
                    optimize_build_volume='pytorch1'
                )

            # 4. 언패딩(원래 사이즈로 복구) 및 Numpy로 가져오기
            disp = padder.unpad(disp.float())
            disparity_map = disp.data.cpu().numpy().squeeze().clip(0, None)
            H, W = left_img.shape[:2]
            disparity_map = disparity_map.reshape(H, W)

            # 5. Disparity(시차)를 Depth(미터)로 변환
            K = np.array(cam_info.k).reshape(3, 3)
            fx = K[0, 0] 
            
            disparity_map[disparity_map <= 0.01] = 0.01 
            depth_map = (fx * self.camera_baseline) / disparity_map
            
            return depth_map.astype(np.float32)

        except Exception as e:
            self.get_logger().error(f'Depth 생성 실패: {e}')
            return None


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

    def run_sam(self, rgb_img, x_min, y_min, x_max, y_max):
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
            mask_msg.header.frame_id = "zed_left_camera_optical_frame"
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
            overlay_msg.header.frame_id = "zed_left_camera_optical_frame"
            self.mask_overlay_pub.publish(overlay_msg)

        except Exception as e:
            self.get_logger().error(f'Mask topic publish 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = Sam3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()