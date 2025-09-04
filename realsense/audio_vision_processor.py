import rospy, os, sys
# --- Set the model home directory for DeepFace & others ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.environ['DEEPFACE_HOME'] = project_root
os.environ['TORCH_HOME'] = project_root
# --- End of Fix ---

import time, math, threading
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import torch
from scipy.signal import butter, sosfilt, resample_poly
import pyaudio
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from std_msgs.msg import String
from deepface import DeepFace
from speechbrain.inference.speaker import SpeakerRecognition

from helper import _median_embedding, _l2_normalize, _cosine_similarity, _crop_is_valid

# Optional RetinaFace alignment (set model path in config to enable)
try:
    from deepface.detectors import FaceDetector
    HAS_DEEPFACE_DETECTORS = True
except Exception:
    HAS_DEEPFACE_DETECTORS = False


class AudioVisionProcessor:
    def __init__(self):
        # --- Centralized Configuration ---
        self.config: Dict[str, Any] = {
            "sample_rate": 44100, "vad_rate": 16000, "channels": 2,
            "window_size": 1024, "mic_distance": 0.143, "speed_of_sound": 343, "smoothing_factor": 0.9,
            "target_device_name": "M2S", "bandpass_lowcut": 300,
            "bandpass_highcut": 3800, "bandpass_order": 5,
            "img_width": 640, "img_height": 480, "img_fps": 30,
            "yolo_confidence": 0.6, "doa_sigma": 15.0,
            "vad_threshold": 0.5, "silence_reset_s": 2.0, "silero_frame_size": 512,
            "iou_threshold": 0.45, "inactive_threshold": 30,
            "movement_threshold_px": 15, "bbox_smoothing_fast": 0.9,
            "bbox_smoothing_slow": 0.2,
            "enrollment_audio_seconds": 2.5,
            "face_recognition_model": "SFace",
            "face_rec_threshold": 0.60,
            "voice_rec_threshold": 0.82,
            "speaking_prob_smoothing": 0.7,
            "max_embeddings_per_person": 30,
            "face_reid_min_conf": 0.65,
            "enable_retinaface_aligner": True,
            "retinaface_detector": "retinaface",
        }

        # --- ROS & Threading ---
        rospy.init_node('audio_vision_processor_node', anonymous=True)
        self.pub = rospy.Publisher('audio_doa_vad', String, queue_size=10)
        self.rate = rospy.Rate(50)
        self.vision_thread, self.stop_event, self.detection_lock = None, threading.Event(), threading.Lock()

        # --- State Variables ---
        self.p, self.stream, self.vad_model, self.yolo_model = None, None, None, None
        self.rs_pipeline, self.bandpass_sos, self.prev_doa, self.start_time = None, None, 0.0, None
        self.silero_buf, self.last_voice_time = np.array([], dtype=np.float32), 0.0
        self.face_aligner = None

        # Unified identity map: keys are "Person N" or "Unknown M"
        self.identities: Dict[str, Dict[str, Any]] = {}
        self.next_person_id = 1
        self.next_unknown_id = 1

        # Recognition models & buffers
        self.face_recognizer, self.speaker_recognizer = None, None
        self.audio_buffer_for_recognition = np.array([], dtype=np.int16)

    # ---------------- initialization ----------------
    def _initialize_all(self):
        rospy.loginfo("--- Initializing Subsystems ---")
        self._initialize_yolo()
        self._initialize_recognition_models()
        self._initialize_vad()
        self._initialize_realsense()
        self._initialize_audio_stream()
        rospy.loginfo("--- All Subsystems Initialized Successfully ---")

    def _initialize_yolo(self):
        rospy.loginfo("Initializing YOLOv8 face detector...")
        try:
            model_path = os.path.join(project_root, 'models', 'yolov8n-face.pt')
            if not os.path.exists(model_path):
                rospy.logfatal(f"YOLO weights not found at: {model_path}")
                sys.exit(1)
            self.yolo_model = YOLO(model_path)
        except Exception as e:
            rospy.logfatal(f"Failed to load YOLO model: {e}")
            sys.exit(1)

    def _initialize_recognition_models(self):
        rospy.loginfo("Initializing Face and Speaker Recognition models...")
        try:
            model_name = self.config["face_recognition_model"]
            self.face_recognizer = DeepFace.build_model(model_name)
            rospy.loginfo(f"Face recognition model ({model_name}) loaded.")
        except Exception as e:
            rospy.logerr(f"Failed to load DeepFace model: {e}")
            self.face_recognizer = None

        # optional aligner
        if HAS_DEEPFACE_DETECTORS and self.config.get("enable_retinaface_aligner", False):
            try:
                self.face_aligner = FaceDetector.build_model(self.config["retinaface_detector"])
                rospy.loginfo("RetinaFace aligner initialized.")
            except Exception as e:
                rospy.logwarn(f"RetinaFace aligner init failed: {e}")
                self.face_aligner = None

        try:
            self.speaker_recognizer = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(project_root, 'models')
            )
            rospy.loginfo("Speaker recognition model loaded.")
        except Exception as e:
            rospy.logerr(f"Failed to load SpeechBrain speaker model: {e}")
            self.speaker_recognizer = None

    def _initialize_vad(self):
        rospy.loginfo("Initializing Silero VAD...")
        try:
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            self.vad_model.reset_states()
        except Exception as e:
            rospy.logerr(f"Silero VAD load failed: {e}")
            self.vad_model = None
        self.silero_buf = np.array([], dtype=np.float32)
        self.last_voice_time = 0.0

    def _initialize_realsense(self):
        rospy.loginfo("Initializing Intel RealSense camera...")
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.config['img_width'], self.config['img_height'], rs.format.z16, self.config['img_fps'])
        config.enable_stream(rs.stream.color, self.config['img_width'], self.config['img_height'], rs.format.bgr8, self.config['img_fps'])
        self.rs_pipeline.start(config)

    def _initialize_audio_stream(self):
        rospy.loginfo("Initializing PyAudio...")
        self.p = pyaudio.PyAudio()
        device_idx, dev_name = self._find_audio_device()
        if device_idx is None:
            rospy.logerr("Audio device not found.")
            self.stop()
            sys.exit(1)
        rospy.loginfo(f"Using audio device: {dev_name} (index {device_idx})")
        self.bandpass_sos = self._design_bandpass_filter()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.config['channels'],
                                  rate=self.config['sample_rate'],
                                  input=True,
                                  input_device_index=device_idx,
                                  frames_per_buffer=self.config['window_size'])

    # ---------------- vision worker ----------------
    def _calculate_iou(self, boxA, boxB):
        try:
            xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
            xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        except Exception:
            return 0.0

    def _vision_worker(self):
        rospy.loginfo("Vision thread started.")
        while not self.stop_event.is_set():
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                results = self.yolo_model(color_image, verbose=False)

                detections = []
                for box in results[0].boxes:
                    try:
                        confidence = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                        if confidence < self.config['yolo_confidence']:
                            continue
                        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                        detections.append({'bbox': xyxy, 'matched': False, 'conf': confidence})
                    except Exception:
                        continue

                # mark all identities as potentially off-screen
                for pid, person_data in list(self.identities.items()):
                    person_data['on_screen'] = False
                    person_data['inactive_frames'] = person_data.get('inactive_frames', 0) + 1

                # Step 1: IoU association to existing identities (Person or Unknown)
                for pid, pdata in list(self.identities.items()):
                    best_match_idx, best_iou = -1, self.config['iou_threshold']
                    for i, det in enumerate(detections):
                        if det['matched']:
                            continue
                        iou = self._calculate_iou(pdata.get('bbox', [0,0,0,0]), det['bbox'])
                        if iou > best_iou:
                            best_iou, best_match_idx = iou, i
                    if best_match_idx != -1:
                        det = detections[best_match_idx]
                        det['matched'] = True
                        pdata['on_screen'] = True
                        pdata['inactive_frames'] = 0
                        prev_bbox = np.array(pdata.get('bbox', det['bbox']), dtype=float)
                        new_bbox = det['bbox']
                        move = np.linalg.norm(prev_bbox[:2] - new_bbox[:2])
                        alpha = self.config['bbox_smoothing_slow'] if move < self.config['movement_threshold_px'] else self.config['bbox_smoothing_fast']
                        smoothed = (alpha * new_bbox + (1 - alpha) * prev_bbox).astype(int)
                        pdata['bbox'] = smoothed

                # Step 2: Try face-based re-ID against enrolled Persons
                for i, det in enumerate(detections):
                    if det['matched']:
                        continue
                    face_emb = self._get_face_embedding(color_image, det['bbox'])
                    matched_person_id = self._recognize_enrolled_by_face(face_emb)
                    if matched_person_id is not None:
                        # assign detection to matched person
                        if matched_person_id not in self.identities:
                            self._create_person_identity(matched_person_id)
                        pdata = self.identities[matched_person_id]
                        pdata['bbox'] = det['bbox'].astype(int)
                        pdata['on_screen'] = True
                        pdata['inactive_frames'] = 0
                        if face_emb is not None:
                            self._update_identity(matched_person_id, face_embedding=face_emb)
                        det['matched'] = True

                # Step 3: Create Unknown tracks for unmatched detections
                for det in detections:
                    if det['matched']:
                        continue
                    face_emb = self._get_face_embedding(color_image, det['bbox'])
                    uid = f"Unknown {self.next_unknown_id}"
                    self.next_unknown_id += 1
                    self.identities[uid] = {
                        'id': uid,
                        'bbox': det['bbox'].astype(int),
                        'inactive_frames': 0,
                        'on_screen': True,
                        'speaking_prob': 0.0,
                        'audio_buffer': np.array([], dtype=np.int16),
                        'face_embeddings': deque(maxlen=self.config['max_embeddings_per_person']),
                        'voice_embeddings': deque(maxlen=self.config['max_embeddings_per_person'])
                    }
                    if face_emb is not None:
                        self.identities[uid]['face_embeddings'].append(face_emb)

                # Step 4: Try to promote on-screen Unknowns to Persons (face re-ID)
                for uid, data in list(self.identities.items()):
                    if not data.get('on_screen', False) or not uid.startswith("Unknown"):
                        continue
                    face_emb = self._get_face_embedding(color_image, data['bbox'])
                    matched_person_id = self._recognize_enrolled_by_face(face_emb)
                    if matched_person_id is not None:
                        rospy.loginfo(f"Merging {uid} into {matched_person_id} (face re-ID).")
                        if matched_person_id not in self.identities:
                            self._create_person_identity(matched_person_id)
                        self._merge_track_into_person(uid, matched_person_id, new_bbox=data['bbox'])
                        continue

                # Step 5: Auto-enrollment Unknown -> Person when enough data
                for uid, data in list(self.identities.items()):
                    if not uid.startswith("Unknown"):
                        continue
                    enough_face = len(data['face_embeddings']) > 0
                    enough_audio = len(data.get('audio_buffer', [])) >= int(self.config['enrollment_audio_seconds'] * self.config['vad_rate'])
                    if enough_face and enough_audio:
                        self._enroll_new_person_from_unknown(uid)

                # Step 6: Cleanup long-inactive unknowns
                to_delete = []
                for pid, pdata in list(self.identities.items()):
                    if pid.startswith("Unknown") and pdata.get('inactive_frames', 0) > self.config['inactive_threshold']:
                        to_delete.append(pid)
                for pid in to_delete:
                    rospy.loginfo(f"--- Removing inactive {pid} ---")
                    del self.identities[pid]

                # Prepare latest_detections for audio thread (angles)
                with self.detection_lock:
                    self.latest_detections = []
                    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    for pid, data in self.identities.items():
                        if data.get('on_screen', False):
                            try:
                                x1, y1, x2, y2 = [int(v) for v in data['bbox']]
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                depth = depth_frame.get_distance(cx, cy)
                                if 0.1 < depth < 10.0:
                                    point3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                                    angle = math.degrees(math.atan2(point3d[0], point3d[2]))
                                    data['angle_deg'] = angle
                                    self.latest_detections.append({"id": pid, "angle_deg": angle})
                            except Exception:
                                continue

                # Visualization overlay (smoothed speaking probability)
                vis_image = color_image.copy()
                for pid, data in self.identities.items():
                    if data.get('on_screen', False):
                        x1, y1, x2, y2 = [int(v) for v in data['bbox']]
                        prob = float(data.get('speaking_prob', 0.0))
                        prob = max(0.0, min(1.0, prob))
                        color = (0, int(255 * prob), int(255 * (1 - prob)))
                        label = f"{pid} | {prob:.2f}"
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis_image, label, (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("Head Tracking & Speaker ID", vis_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()

            except Exception as e:
                rospy.logerr(f"[Vision Thread] CRASH: {e}", exc_info=True)

    # ---------------- main loop & audio processing ----------------
    def run(self):
        try:
            self._initialize_all()
        except Exception as e:
            rospy.logfatal(f"Initialization failed: {e}. Shutting down.")
            self.stop()
            return
        self.vision_thread = threading.Thread(target=self._vision_worker, daemon=True)
        self.vision_thread.start()
        self.start_time = rospy.get_time()
        rospy.loginfo("--- Starting Main Audio Processing Loop ---")
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                self._process_audio_chunk()
            except IOError:
                rospy.logwarn("Audio input overflowed.")
            except Exception as e:
                rospy.logerr(f"Audio loop error: {e}", exc_info=True)
                break
            self.rate.sleep()

    def _process_audio_chunk(self):
        if not self.stream:
            return
        audio_chunk = self.stream.read(self.config['window_size'], exception_on_overflow=False)
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        if audio_array.size == 0:
            return

        # VAD (Silero) logic (resample & buffer)
        voice_prob = self._process_vad(audio_array)
        mono_audio_16k = self._resample_audio_for_rec(audio_array)

        # Always buffer recent audio for recognition attempts
        self.audio_buffer_for_recognition = np.concatenate((self.audio_buffer_for_recognition, mono_audio_16k))
        max_buffer_len = int(self.config['enrollment_audio_seconds'] * self.config['vad_rate'] * 1.5)
        if len(self.audio_buffer_for_recognition) > max_buffer_len:
            self.audio_buffer_for_recognition = self.audio_buffer_for_recognition[-max_buffer_len:]

        # If no speech, decay speaking probs and return
        if voice_prob < self.config['vad_threshold']:
            with self.detection_lock:
                for pdata in self.identities.values():
                    old_prob = float(pdata.get('speaking_prob', 0.0))
                    pdata['speaking_prob'] = old_prob * self.config['speaking_prob_smoothing']
            return

        # Speech detected: DOA calculation
        left, right = audio_array[0::2], audio_array[1::2]
        doa = self.process_doa(left, right)
        if doa is None:
            return

        # Attempt voice-only recognition (off-camera) using buffered audio
        recognized_voice_id = None
        if len(self.audio_buffer_for_recognition) >= int(self.config['enrollment_audio_seconds'] * self.config['vad_rate']):
            voice_emb = self._get_voice_embedding(self.audio_buffer_for_recognition)
            if voice_emb is not None:
                recognized_voice_id = self._recognize_person_by_voice(voice_emb)
                if recognized_voice_id:
                    # If recognized, ensure identity exists and update voice embedding
                    if recognized_voice_id not in self.identities:
                        # it's improbable but create identity container if missing
                        self._create_person_identity(recognized_voice_id)
                    self._update_identity(recognized_voice_id, voice_embedding=voice_emb)
            self.audio_buffer_for_recognition = np.array([], dtype=np.int16)

        # Update speaking probabilities for on-screen people using DOA + voice_prob
        with self.detection_lock:
            self._update_speaking_probs(doa, voice_prob)
            best_visual_id = self._find_best_visual_match()

            speaker_label = None
            # If we have a visual best candidate with high prob, prefer it
            if best_visual_id:
                pdata = self.identities.get(best_visual_id, None)
                # If visual candidate is Unknown but voice recognized someone, merge
                if best_visual_id.startswith("Unknown") and recognized_voice_id:
                    rospy.loginfo(f"Cross-modal merge: {best_visual_id} -> {recognized_voice_id}")
                    self._merge_track_into_person(best_visual_id, recognized_voice_id, new_bbox=pdata.get('bbox') if pdata else None)
                    speaker_label = recognized_voice_id
                else:
                    speaker_label = best_visual_id
                    # append audio to track (for later enrollment)
                    if pdata is not None and best_visual_id.startswith("Unknown"):
                        pdata['audio_buffer'] = np.concatenate((pdata.get('audio_buffer', np.array([], dtype=np.int16)), mono_audio_16k))

            # If no visual candidate but voice recognized someone, label them as off-screen speaker
            if speaker_label is None and recognized_voice_id:
                speaker_label = recognized_voice_id

            # Ensure we always produce a speaker_label (fall back to Unknown)
            if speaker_label is None:
                speaker_label = "Unknown"

        # Publish message (always include speaker)
        elapsed_time = rospy.get_time() - self.start_time
        msg_str = f"[{elapsed_time:8.3f}s] VAD: {voice_prob:.2f} | DOA: {doa:6.1f}"
        msg_str += f" -> Speaker: {speaker_label}"
        if speaker_label.startswith("Person") and best_visual_id is None:
            msg_str += " (Off-screen)"

        self.pub.publish(String(data=msg_str))
        rospy.loginfo(msg_str)

    # ---------------- speaking prob helpers ----------------
    def _update_speaking_probs(self, doa, voice_prob):
        alpha = self.config['speaking_prob_smoothing']
        for pid, pdata in self.identities.items():
            old_prob = float(pdata.get('speaking_prob', 0.0))
            # only compute DOA-based prob for on-screen entries
            if pdata.get('on_screen', False) and ('angle_deg' in pdata):
                prob_doa = math.exp(-0.5 * ((float(pdata['angle_deg']) - float(doa)) / self.config['doa_sigma'])**2)
                current_prob = 0.6 * prob_doa + 0.4 * voice_prob
                pdata['speaking_prob'] = alpha * old_prob + (1 - alpha) * current_prob
            else:
                # decay for off-screen identities
                pdata['speaking_prob'] = alpha * old_prob

    def _find_best_visual_match(self):
        best_id, max_prob = None, self.config['vad_threshold']
        for pid, pdata in self.identities.items():
            if pdata.get('on_screen', False):
                prob = float(pdata.get('speaking_prob', 0.0))
                if prob > max_prob:
                    max_prob, best_id = prob, pid
        return best_id

    # ---------------- enrollment/recognition helpers ----------------
    def _enroll_new_person_from_unknown(self, unknown_id: str):
        track = self.identities.get(unknown_id)
        if not track:
            return
        avg_face_emb = _median_embedding(list(track['face_embeddings']))
        voice_emb = self._get_voice_embedding(track.get('audio_buffer', np.array([], dtype=np.int16)))
        if avg_face_emb is None or voice_emb is None:
            return

        recognized_id = self._recognize_person(face_embedding=avg_face_emb, voice_embedding=voice_emb)
        if recognized_id != "Unknown":
            rospy.loginfo(f"{unknown_id} matches enrolled {recognized_id}; merging.")
            self._merge_track_into_person(unknown_id, recognized_id, new_bbox=track.get('bbox'))
            return

        new_person_id = f"Person {self.next_person_id}"
        self.next_person_id += 1
        rospy.loginfo(f"--- Enrolled new identity: {new_person_id} ---")

        self.identities[new_person_id] = {
            'id': new_person_id,
            'bbox': track.get('bbox', [0,0,0,0]),
            'inactive_frames': track.get('inactive_frames', 0),
            'on_screen': track.get('on_screen', False),
            'speaking_prob': track.get('speaking_prob', 0.0),
            'audio_buffer': np.array([], dtype=np.int16),
            'face_embeddings': deque(maxlen=self.config['max_embeddings_per_person']),
            'voice_embeddings': deque(maxlen=self.config['max_embeddings_per_person'])
        }
        for emb in list(track['face_embeddings']):
            if emb is not None:
                self.identities[new_person_id]['face_embeddings'].append(_l2_normalize(emb))
        if voice_emb is not None:
            self.identities[new_person_id]['voice_embeddings'].append(_l2_normalize(voice_emb))
        del self.identities[unknown_id]

    def _merge_track_into_person(self, track_id: str, person_id: str, new_bbox=None):
        if track_id not in self.identities:
            return
        track = self.identities[track_id]
        if person_id not in self.identities:
            self._create_person_identity(person_id)
        person = self.identities[person_id]

        if new_bbox is not None:
            person['bbox'] = np.array(new_bbox).astype(int)
        person['on_screen'] = track.get('on_screen', False)
        person['inactive_frames'] = 0

        for emb in list(track.get('face_embeddings', [])):
            if emb is not None:
                person['face_embeddings'].append(_l2_normalize(emb))
        for vemb in list(track.get('voice_embeddings', [])):
            if vemb is not None:
                person['voice_embeddings'].append(_l2_normalize(vemb))
        del self.identities[track_id]

    def _create_person_identity(self, person_id: str) -> Dict[str, Any]:
        if person_id in self.identities:
            return self.identities[person_id]
        self.identities[person_id] = {
            'id': person_id,
            'bbox': np.array([0, 0, 0, 0], dtype=int),
            'inactive_frames': self.config['inactive_threshold'] + 1,
            'on_screen': False,
            'speaking_prob': 0.0,
            'audio_buffer': np.array([], dtype=np.int16),
            'face_embeddings': deque(maxlen=self.config['max_embeddings_per_person']),
            'voice_embeddings': deque(maxlen=self.config['max_embeddings_per_person'])
        }
        return self.identities[person_id]

    def _recognize_enrolled_by_face(self, face_embedding: Optional[np.ndarray]) -> Optional[str]:
        if face_embedding is None:
            return None
        face_embedding = _l2_normalize(face_embedding)
        best_person, best_sim = None, -1.0
        for pid, pdata in self.identities.items():
            if not pid.startswith("Person"):
                continue
            if len(pdata['face_embeddings']) == 0:
                continue
            ref = _median_embedding(list(pdata['face_embeddings']))
            if ref is None:
                continue
            sim = _cosine_similarity(face_embedding, ref)
            if sim > best_sim:
                best_sim = sim
                best_person = pid
        if best_person is not None and best_sim >= self.config['face_reid_min_conf']:
            return best_person
        return None

    def _recognize_person_by_voice(self, voice_embedding: Optional[np.ndarray]) -> Optional[str]:
        if voice_embedding is None:
            return None
        voice_embedding = _l2_normalize(voice_embedding)
        best_person, best_sim = None, self.config['voice_rec_threshold']
        for pid, pdata in self.identities.items():
            if not pid.startswith("Person"):
                continue
            if len(pdata['voice_embeddings']) < 1:
                continue
            ref_voice = _median_embedding(list(pdata['voice_embeddings']))
            if ref_voice is None:
                continue
            sim = _cosine_similarity(voice_embedding, ref_voice)
            if sim > best_sim:
                best_sim = sim
                best_person = pid
        return best_person

    def _recognize_person(self, face_embedding=None, voice_embedding=None):
        if face_embedding is None and voice_embedding is None:
            return "Unknown"

        if face_embedding is not None:
            face_embedding = _l2_normalize(face_embedding)
        if voice_embedding is not None:
            voice_embedding = _l2_normalize(voice_embedding)

        best_match_id = "Unknown"
        highest_score = 0.0

        for person_id, identity in self.identities.items():
            if not person_id.startswith("Person"):
                continue

            face_sim, voice_sim = 0.0, 0.0
            if face_embedding is not None and len(identity['face_embeddings']) > 0:
                ref_face = _median_embedding(list(identity['face_embeddings']))
                if ref_face is not None:
                    face_sim = _cosine_similarity(face_embedding, ref_face)
            if voice_embedding is not None and len(identity['voice_embeddings']) > 0:
                ref_voice = _median_embedding(list(identity['voice_embeddings']))
                if ref_voice is not None:
                    voice_sim = _cosine_similarity(voice_embedding, ref_voice)

            score = max(face_sim, voice_sim)
            if score > highest_score:
                if (score == face_sim and face_sim > self.config['face_rec_threshold']) or \
                   (score == voice_sim and voice_sim > self.config['voice_rec_threshold']):
                    highest_score = score
                    best_match_id = person_id

        return best_match_id

    def _update_identity(self, person_id, face_embedding=None, voice_embedding=None):
        if person_id not in self.identities:
            return
        if face_embedding is not None:
            fe = _l2_normalize(face_embedding)
            if fe is not None:
                self.identities[person_id]['face_embeddings'].append(fe)
        if voice_embedding is not None:
            ve = _l2_normalize(voice_embedding)
            if ve is not None:
                self.identities[person_id]['voice_embeddings'].append(ve)

    # ---------------- face embedding / align ----------------
    def _align_face(self, color_image, bbox):
        if self.face_aligner is None:
            return None
        try:
            x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
            h, w = color_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = color_image[y1:y2, x1:x2]
            if not _crop_is_valid(crop):
                return None
            # Use face_aligner instance to detect and return aligned face image
            dets = self.face_aligner.detect_faces(crop, align=True)
            if isinstance(dets, list) and len(dets) > 0:
                face_img = dets[0].get("face", None)
                if _crop_is_valid(face_img):
                    return face_img
            return None
        except Exception:
            return None

    def _get_face_embedding(self, color_image, bbox):
        if self.face_recognizer is None:
            return None
        try:
            aligned = None
            if self.face_aligner is not None:
                aligned = self._align_face(color_image, bbox)
            if aligned is not None:
                face_img = aligned
            else:
                x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
                h, w = color_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    return None
                face_img = color_image[y1:y2, x1:x2]

            if not _crop_is_valid(face_img):
                return None

            # DeepFace.represent supports numpy arrays when detector_backend='skip'
            rep = DeepFace.represent(
                img_path=face_img,
                model_name=self.config["face_recognition_model"],
                enforce_detection=False,
                detector_backend='skip'
            )
            if not rep:
                return None
            embedding = np.array(rep[0]["embedding"], dtype=np.float32)
            return _l2_normalize(embedding)
        except Exception as e:
            # throttle logs to avoid flooding
            rospy.logdebug(f"Face embed error: {e}")
            return None

    # ---------------- voice embedding ----------------
    def _get_voice_embedding(self, audio_buffer):
        if self.speaker_recognizer is None:
            return None
        if audio_buffer is None or len(audio_buffer) < self.config['vad_rate']:
            return None
        try:
            wav = audio_buffer.astype(np.float32) / 32768.0
            # ensure shape (1, N)
            wav_t = torch.tensor(wav).unsqueeze(0)
            with torch.no_grad():
                emb = self.speaker_recognizer.encode_batch(wav_t)
            # encode_batch returns tensor or list; unify to numpy vector
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            if isinstance(emb, torch.Tensor):
                emb_np = emb.detach().cpu().numpy().reshape(-1)
            else:
                emb_np = np.asarray(emb).reshape(-1)
            return _l2_normalize(emb_np)
        except Exception as e:
            rospy.logwarn_throttle(5, f"Could not generate voice embedding: {e}")
            return None

    def _resample_audio_for_rec(self, audio_array: np.ndarray) -> np.ndarray:
        left = audio_array[0::2].astype(np.int32)
        right = audio_array[1::2].astype(np.int32)
        if left.size == 0 or right.size == 0:
            return np.array([], dtype=np.int16)
        mono_44k = ((left + right) // 2).astype(np.int16)
        resampled_f32 = resample_poly(mono_44k.astype(np.float32), self.config['vad_rate'], self.config['sample_rate'])
        clipped_f32 = np.clip(resampled_f32, -32768, 32767)
        return clipped_f32.astype(np.int16)

    # ---------------- VAD & DOA (kept logic / 2s reset) ----------------
    def _process_vad(self, audio_array: np.ndarray) -> float:
        mono_16k = self._resample_audio_for_rec(audio_array).astype(np.float32)
        self.silero_buf = np.concatenate([self.silero_buf, mono_16k])
        frame_size = self.config['silero_frame_size']
        max_voice_prob = 0.0

        with torch.no_grad():
            while len(self.silero_buf) >= frame_size:
                chunk = self.silero_buf[:frame_size]
                self.silero_buf = self.silero_buf[frame_size:]
                try:
                    tensor = torch.from_numpy(chunk / 32768.0).to(torch.float32)
                    new_prob = float(self.vad_model(tensor, self.config['vad_rate']).item())
                    if new_prob > max_voice_prob:
                        max_voice_prob = new_prob
                except Exception:
                    pass

        now = rospy.get_time() if rospy.core.is_initialized() else time.time()

        if max_voice_prob > self.config['vad_threshold']:
            self.last_voice_time = now
        if (now - self.last_voice_time) > self.config['silence_reset_s']:
            try:
                self.vad_model.reset_states()
            except Exception:
                pass

        return max_voice_prob

    def process_doa(self, left: np.ndarray, right: np.ndarray) -> Optional[float]:
        if self.bandpass_sos is None:
            return None
        try:
            s1_filt = sosfilt(self.bandpass_sos, left.astype(np.float32))
            s2_filt = sosfilt(self.bandpass_sos, right.astype(np.float32))
            n1, n2 = np.max(np.abs(s1_filt)) + 1e-6, np.max(np.abs(s2_filt)) + 1e-6
            s1_filt /= n1; s2_filt /= n2
            corr = np.correlate(s1_filt, s2_filt, mode='full')
            if np.max(corr) < 3 * np.mean(np.abs(corr) + 1e-6):
                return float(self.prev_doa)

            delay_sample = np.argmax(corr) - (len(s1_filt) - 1)
            max_delay = self.config['mic_distance'] / self.config['speed_of_sound']
            time_delay = np.clip(delay_sample / self.config['sample_rate'], -max_delay, max_delay)
            sin_theta = np.clip((time_delay * self.config['speed_of_sound']) / self.config['mic_distance'], -1.0, 1.0)

            doa = float(np.degrees(np.arcsin(sin_theta)))
            if self.prev_doa is not None:
                doa = self.config['smoothing_factor'] * self.prev_doa + (1 - self.config['smoothing_factor']) * doa
            self.prev_doa = doa
            return doa
        except Exception:
            return float(self.prev_doa)

    # ---------------- cleanup & audio helpers ----------------
    def stop(self):
        rospy.loginfo("--- Shutting Down ---")
        self.stop_event.set()
        if self.vision_thread:
            self.vision_thread.join(timeout=2)
        try:
            if self.rs_pipeline:
                self.rs_pipeline.stop()
        except Exception:
            pass
        try:
            if self.stream:
                self.stream.close()
        except Exception:
            pass
        try:
            if self.p:
                self.p.terminate()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        rospy.loginfo("Cleanup complete.")

    def _find_audio_device(self) -> Tuple[Optional[int], Optional[str]]:
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            name = dev.get('name', '')
            if self.config['target_device_name'] in name and dev.get('maxInputChannels', 0) >= self.config['channels']:
                return i, name
        rospy.logwarn(f"'{self.config['target_device_name']}' not found. Using default.")

        try:
            default_dev = self.p.get_default_input_device_info()
            return default_dev['index'], default_dev['name']
        except Exception as e:
            rospy.logerr(f"Default device check failed: {e}")
        return None, None

    def _design_bandpass_filter(self) -> np.ndarray:
        nyq = 0.5 * self.config['sample_rate']
        low = self.config['bandpass_lowcut'] / nyq
        high = min(self.config['bandpass_highcut'] / nyq, 0.99)
        return butter(self.config['bandpass_order'], [low, high], btype='band', output='sos')