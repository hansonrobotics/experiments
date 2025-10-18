import rospy, os, sys

import time, math, threading, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from deepface import DeepFace

try:
    from deepface.detectors import FaceDetector
    HAS_DEEPFACE_DETECTORS = True
except ImportError:
    HAS_DEEPFACE_DETECTORS = False
    rospy.logwarn("deepface.detectors not found, RetinaFace disabled.")
except Exception as e:
    HAS_DEEPFACE_DETECTORS = False
    rospy.logwarn(f"Error importing deepface.detectors: {e}, RetinaFace aligner disabled.")

from helper import _l2_normalize, _cosine_similarity, _median_embedding, _crop_is_valid

class VisionProcessor:
    """Handles Video I/O, face detection, tracking (IoU), and face recognition embeddings."""
    def __init__(self, config, parent_node):
        rospy.loginfo("Initializing Vision Subsystem...")
        self.parent = parent_node
        self.config = config

        #Hhardware
        self.rs_pipeline = None
        self.align = None

        #Models
        self.yolo_model = None
        self.face_recognizer_model_name = None
        self.face_aligner = None

    def initialize(self):
        self._initialize_yolo()
        self._initialize_vision_models()
        self._initialize_realsense()

    def _initialize_yolo(self):
        rospy.loginfo("Initializing YOLOv8 face detector...")
        try:
            project_root = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, 'models', 'yolov8n-face.pt')
            if not os.path.exists(model_path):
                rospy.logfatal(f"YOLO weights not found at: {model_path}. Exiting.")
                if self.parent: self.parent.stop_event.set()
                sys.exit(1)
            self.yolo_model = YOLO(model_path)
            rospy.loginfo("YOLO model loaded successfully.")
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLO model: {e}. Exiting.")
            if self.parent: self.parent.stop_event.set()
            sys.exit(1)

    def _initialize_vision_models(self):
        rospy.loginfo("Initializing Face Recognition models...")
        project_root = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            model_name = self.config["face_recognition_model"]
            _ = DeepFace.build_model(model_name) #Trigger build
            self.face_recognizer_model_name = model_name
            rospy.loginfo(f"Face recognition model ({model_name}) pre-loaded/checked.")
        except Exception as e:
            rospy.logerr(f"Failed to build/check DeepFace model '{model_name}': {e}")
            self.face_recognizer_model_name = None

        if HAS_DEEPFACE_DETECTORS and self.config.get("enable_retinaface_aligner", False):
            detector_name = self.config.get("retinaface_detector", "retinaface")
            try:
                self.face_aligner = FaceDetector.build_model(detector_name)
                rospy.loginfo(f"RetinaFace aligner ('{detector_name}') initialized.")
            except Exception as e:
                rospy.logwarn(f"RetinaFace aligner ('{detector_name}') init failed: {e}")
                self.face_aligner = None
        else:
             rospy.loginfo("RetinaFace aligner disabled by config or import error.")
             self.face_aligner = None

    def _initialize_realsense(self):
        rospy.loginfo("Initializing Intel RealSense camera...")
        try:
            self.rs_pipeline = rs.pipeline()
            config = rs.config()
            w, h, fps = self.config['img_width'], self.config['img_height'], self.config['img_fps']
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            profile = self.rs_pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            rospy.loginfo(f"RealSense streams enabled ({w}x{h} @ {fps}fps)")
        except RuntimeError as e:
             rospy.logfatal(f"RealSense runtime error: {e}. Check camera connection")
             if self.parent: self.parent.stop_event.set()
             sys.exit(1)
        except Exception as e:
            rospy.logfatal(f"Failed to initialize RealSense camera: {e}")
            if self.parent: self.parent.stop_event.set()
            sys.exit(1)

    def run_vision_loop(self):
        rospy.loginfo("Vision thread started.")
        while not self.parent.stop_event.is_set():
            color_image, depth_frame, depth_intrinsics = None, None, None
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                if not frames: continue

                #Align frames
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    rospy.logwarn_throttle(5, "Missing color or (aligned) depth frame.")
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                try:
                    depth_profile = depth_frame.profile.as_video_stream_profile()
                    depth_intrinsics = depth_profile.intrinsics
                except Exception as e:
                    rospy.logwarn_throttle(10, f"Could not get depth intrinsics: {e}")
                    continue

                #Face detection
                if self.yolo_model is None:
                     rospy.logerr_throttle(10, "YOLO model not initialized. Skipping")
                     time.sleep(1)
                     continue

                results = self.yolo_model(color_image, verbose=False, conf=self.config['yolo_confidence'])

                current_detections = []
                if results and hasattr(results[0], 'boxes') and results[0].boxes:
                    for box in results[0].boxes:
                        try:
                            confidence = float(box.conf.item())
                            xyxy = box.xyxy[0].cpu().numpy().astype(float)
                            current_detections.append({'bbox': xyxy, 'matched': False, 'conf': confidence})
                        except Exception as e:
                            rospy.logwarn(f"Error processing YOLO box: {e}")
                            continue

                #Generate embeddings
                embeddings_this_frame = {}
                for i, det in enumerate(current_detections):
                    face_emb = self._get_face_embedding(color_image, det['bbox'])
                    if face_emb is not None:
                        embeddings_this_frame[i] = face_emb

                #Update state
                with self.parent.lock:
                    self.parent._update_identities_with_detections(current_detections, embeddings_this_frame, color_image)
                    self.parent._calculate_angles(depth_frame, depth_intrinsics)

                vis_image = self._draw_visualizations(color_image)
                cv2.imshow("Head Tracking & Speaker ID", vis_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.parent.stop_event.set()

            except rs.error as e:
                rospy.logwarn_throttle(5, f"RealSense error in vision loop: {e}")
            except Exception as e:
                rospy.logerr(f"[Vision Thread] Unexpected Error: {e}", exc_info=True)
                time.sleep(0.1)

    def _draw_visualizations(self, color_image):
        #Bounding boxes
        vis_image = color_image.copy()
        with self.parent.lock:
             identity_items = list(self.parent.identities.items())

        for pid, data in identity_items:
             if data.get('on_screen', False) and 'bbox' in data:
                x1, y1, x2, y2 = data['bbox']
                prob = float(data.get('speaking_prob', 0.0))
                prob = max(0.0, min(1.0, prob))
                color = (0, int(255 * prob), int(255 * (1 - prob))) #Green=Speak, Red=Silent
                label = f"{pid} | SP:{prob:.2f}"
                if 'angle_deg' in data:
                     label += f" | {data['angle_deg']:.1f}d"

                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                label_y = max(y1 - 10, 15)
                cv2.putText(vis_image, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return vis_image

    def _calculate_iou(self, boxA, boxB):
        #IoU
        try:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)
            return iou
        except Exception as e:
            rospy.logwarn_throttle(10, f"Error calculating IoU: {e}")
            return 0.0

    def _align_face(self, color_image, bbox):
        #Face allignment with Retinaface
        if self.face_aligner is None: return None
        try:
            x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
            h, w = color_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1: return None
            crop = color_image[y1:y2, x1:x2]

            if not _crop_is_valid(crop, min_side=20, min_std=5.0): return None

            results = self.face_aligner.detect_faces(crop, align=True)
            aligned_face = None
            if isinstance(results, list) and len(results) > 0:
                 if isinstance(results[0], dict):
                      aligned_face = results[0].get("face")

            if _crop_is_valid(aligned_face):
                return aligned_face
            else:
                 return None
        except Exception as e:
            rospy.logwarn_throttle(10, f"Face alignment error: {e}")
            return None

    def _get_face_embedding(self, color_image, bbox):
        #Extract embeddings with Deepface
        if self.face_recognizer_model_name is None: return None

        face_img_to_process = None
        if self.face_aligner:
            aligned_face = self._align_face(color_image, bbox)
            if aligned_face is not None:
                 face_img_to_process = aligned_face

        #Fallback to simple crop allignment if disabled or failed
        if face_img_to_process is None:
            try:
                x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
                h, w = color_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1: return None
                crop = color_image[y1:y2, x1:x2]
                if _crop_is_valid(crop):
                     face_img_to_process = crop
                else: return None
            except Exception:
                 return None

        if face_img_to_process is None: return None

        try:
            representation = DeepFace.represent(
                img_path=face_img_to_process,
                model_name=self.face_recognizer_model_name,
                enforce_detection=False,
                detector_backend='skip'
            )
            if isinstance(representation, list) and len(representation) > 0:
                embedding = representation[0].get("embedding")
                if embedding is not None:
                    return _l2_normalize(np.array(embedding, dtype=np.float32))
            return None
        except ValueError as ve:
             if "Face could not be detected" not in str(ve):
                  rospy.logwarn_throttle(5, f"ValueError during face embedding: {ve}")
             return None
        except Exception as e:
            rospy.logwarn_throttle(5, f"Unexpected error during face embedding: {e}")
            return None

    def stop(self):
        rospy.loginfo("Stopping Vision Subsystem...")
        if self.rs_pipeline:
            try:
                self.rs_pipeline.stop()
                rospy.loginfo("RealSense pipeline stopped.")
            except Exception as e:
                rospy.logerr(f"Error stopping RealSense pipeline: {e}")
            finally:
                self.rs_pipeline = None