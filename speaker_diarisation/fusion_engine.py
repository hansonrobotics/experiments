import rospy, os, sys
import threading, math, time
from collections import deque
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from std_msgs.msg import String
import cv2

from audio_processor import AudioProcessor
from vision_processor import VisionProcessor
from helper import _l2_normalize, _cosine_similarity, _median_embedding



class FusionEngine:
    def __init__(self):
        #ROS Initialization
        self.pub = rospy.Publisher('speaker_output', String, queue_size=10)
        self.rate = rospy.Rate(50)

        self.config = {
            "sample_rate": 44100, "vad_rate": 16000, "channels": 2, "window_size": 1024,
            "target_device_name": "M2S",
            "vad_threshold": 0.5, "silence_reset_s": 2.0, "silero_frame_size": 512,
            "mic_distance": 0.143, "speed_of_sound": 343, "doa_smoothing_factor": 0.9,
            "bandpass_lowcut": 300, "bandpass_highcut": 3800, "bandpass_order": 5,
            "doa_sigma": 15.0,
            "img_width": 640, "img_height": 480, "img_fps": 30,
            "yolo_confidence": 0.6,
            "iou_threshold": 0.45, "inactive_threshold": 30,
            "movement_threshold_px": 15, "bbox_smoothing_fast": 0.9, "bbox_smoothing_slow": 0.2,
            "enrollment_audio_seconds": 2.5,
            "face_recognition_model": "SFace",
            "face_rec_threshold": 0.60, "voice_rec_threshold": 0.82,
            "max_embeddings_per_person": 30, "face_reid_min_conf": 0.65,
            "enable_retinaface_aligner": True, "retinaface_detector": "retinaface",
            "speaking_prob_smoothing": 0.7,
        }

        #Threading
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.identities = {}
        self.latest_detections = [] #[{'id': pid, 'angle_deg': angle}]

        #Manage id
        self.next_person_id = 1
        self.next_unknown_id = 1

        self.audio_processor = AudioProcessor(self.config, self)
        self.vision_processor = VisionProcessor(self.config, self)
        self.vision_thread = None
        self.start_time = None

    def run(self):
        try:
            self.audio_processor.initialize()
            self.vision_processor.initialize()
            rospy.loginfo("All Subsystems Initialized Successfully")
        except Exception as e:
            rospy.logfatal(f"Initialization failed: {e}. Shutting down.")
            self.stop()
            return

        self.vision_thread = threading.Thread(target=self.vision_processor.run_vision_loop, daemon=True)
        self.vision_thread.start()

        self.start_time = rospy.get_time()
        rospy.loginfo("Starting Main Fusion Loop (Audio Triggered)")

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            voice_prob, doa, voice_emb = self.audio_processor.process_chunk()

            if voice_prob is not None:
                recognized_voice_id = None
                speaker_label = "Unknown"

                with self.lock:
                    if voice_emb is not None:
                         recognized_voice_id = self._recognize_person_by_voice(voice_emb)
                         if recognized_voice_id:
                              if recognized_voice_id not in self.identities:
                                   self._create_person_identity(recognized_voice_id)
                              self._update_identity(recognized_voice_id, voice_embedding=voice_emb)
                    if doa is not None:
                        self._update_speaking_probs(doa, voice_prob)
                    best_visual_id = self._find_best_visual_match()
                    speaker_label = self._determine_speaker_label(best_visual_id, recognized_voice_id)

                self._publish_result(voice_prob, doa, speaker_label, best_visual_id)

            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS interrupt received during sleep.")
                break

    def stop(self):
        if self.stop_event.is_set(): return
        rospy.loginfo("Shutting Down Fusion Engine")
        self.stop_event.set()

        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)

        if self.vision_processor: self.vision_processor.stop()
        if self.audio_processor: self.audio_processor.stop()

        try:
            cv2.destroyAllWindows()
        except Exception: pass
        rospy.loginfo("Fusion Engine Cleanup Complete")

    """Logic methods"""

    def _update_identities_with_detections(self, current_detections, embeddings_this_frame, color_image):
        identities = self.identities
        #Existing track marking
        for track_id, track_data in identities.items():
            track_data['on_screen'] = False
            track_data['inactive_frames'] = track_data.get('inactive_frames', 0) + 1

        #IoU to match new to old tracks
        matched_indices_iou = set()
        for track_id, track_data in identities.items():
            best_match_idx, best_iou = -1, self.config['iou_threshold']
            for i, det in enumerate(current_detections):
                if i in matched_indices_iou or det.get('matched', False): continue

                iou = self.vision_processor._calculate_iou(track_data.get('bbox', [0,0,0,0]), det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_match_idx != -1:
                det = current_detections[best_match_idx]
                det['matched'] = True
                matched_indices_iou.add(best_match_idx)

                track_data['on_screen'] = True
                track_data['inactive_frames'] = 0
                prev_bbox = np.array(track_data.get('bbox', det['bbox']), dtype=float)
                new_bbox = det['bbox']
                move = np.linalg.norm(prev_bbox[:2] - new_bbox[:2])
                alpha = self.config['bbox_smoothing_slow'] if move < self.config['movement_threshold_px'] else self.config['bbox_smoothing_fast']
                smoothed = (alpha * new_bbox + (1 - alpha) * prev_bbox).astype(int)
                track_data['bbox'] = smoothed
                if best_match_idx in embeddings_this_frame:
                     self._update_identity(track_id, face_embedding=embeddings_this_frame[best_match_idx])

        #Reidentify unmatched detections
        matched_indices_reid = set()
        for i, det in enumerate(current_detections):
            if det.get('matched', False): continue

            if i in embeddings_this_frame:
                face_emb = embeddings_this_frame[i]
                matched_person_id = self._recognize_enrolled_by_face(face_emb)
                if matched_person_id:
                    det['matched'] = True
                    matched_indices_reid.add(i)
                    if matched_person_id not in identities: 
                         self._create_person_identity(matched_person_id)
                    pdata = identities[matched_person_id]
                    pdata['bbox'] = det['bbox'].astype(int)
                    pdata['on_screen'] = True
                    pdata['inactive_frames'] = 0
                    self._update_identity(matched_person_id, face_embedding=face_emb)

        #New unknown track creation
        newly_created_unknowns = []
        for i, det in enumerate(current_detections):
            if not det.get('matched', False):
                new_track_id = self._create_unknown_identity(det['bbox'])
                newly_created_unknowns.append(new_track_id)
                if i in embeddings_this_frame:
                    identities[new_track_id]['face_embeddings'].append(embeddings_this_frame[i])


        processed_for_promotion = set()
        for track_id, track_data in list(identities.items()):
            if not track_id.startswith("Unknown") or track_id in processed_for_promotion: continue

            #Reidentify against "persons"
            current_face_emb = None
            if track_data['on_screen']:
                 found_emb = False
                 for idx, det_idx in enumerate(matched_indices_iou.union(matched_indices_reid)): #Check matches
                      pass
                 #Get new embedding
                 current_face_emb = self.vision_processor._get_face_embedding(color_image, track_data['bbox'])


            if current_face_emb is None and len(track_data['face_embeddings']) > 0:
                 current_face_emb = _median_embedding(list(track_data['face_embeddings']))

            if current_face_emb is not None:
                matched_person_id = self._recognize_enrolled_by_face(current_face_emb)
                if matched_person_id:
                    rospy.loginfo(f"Merging Unknown {track_id} into {matched_person_id} (face re-ID).")
                    self._merge_track_into_person(track_id, matched_person_id, new_bbox=track_data['bbox'])
                    processed_for_promotion.add(track_id)
                    continue

            #Check enrollment
            if track_id not in processed_for_promotion:
                 enough_face = len(track_data['face_embeddings']) > 0
                 audio_buffer = track_data.get('audio_buffer', np.array([], dtype=np.int16))
                 enough_audio = len(audio_buffer) >= int(self.config['enrollment_audio_seconds'] * self.config['vad_rate'])

                 if enough_face and enough_audio:
                      rospy.loginfo(f"Attempting to enroll {track_id} based on collected data.")
                      self._enroll_new_person_from_unknown(track_id)
                      processed_for_promotion.add(track_id)


        #Get rid of unused unknowns
        inactive_threshold = self.config['inactive_threshold']
        to_delete = [
            track_id for track_id, track_data in identities.items()
            if track_id.startswith("Unknown") and track_data.get('inactive_frames', 0) > inactive_threshold
        ]
        for track_id in to_delete:
            rospy.loginfo(f"--- Removing inactive {track_id} ---")
            del identities[track_id]


    def _calculate_angles(self, depth_frame, depth_intrinsics):

        self.latest_detections = []
        h, w = self.config['img_height'], self.config['img_width']

        for pid, data in self.identities.items():
            if data.get('on_screen', False) and 'bbox' in data:
                try:
                    x1, y1, x2, y2 = data['bbox']
                    cx = np.clip(int((x1 + x2) / 2), 0, w - 1)
                    cy = np.clip(int((y1 + y2) / 2), 0, h - 1)

                    depth = depth_frame.get_distance(cx, cy)
                    if 0.1 < depth < 10.0: #Validate depth
                        point3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                        angle = math.degrees(math.atan2(point3d[0], point3d[2])) #XZ plane
                        data['angle_deg'] = angle
                        self.latest_detections.append({"id": pid, "angle_deg": angle})
                    else:
                         data.pop('angle_deg', None)
                except Exception as e:
                    data.pop('angle_deg', None)
                    rospy.logwarn_throttle(10, f"Could not calculate angle for {pid}: {e}")
            else:
                 data.pop('angle_deg', None)


    def _determine_speaker_label(self, best_visual_id, recognized_voice_id):
         speaker_label = "Unknown" # Default

         #Strong voice recongition
         if recognized_voice_id:
              speaker_label = recognized_voice_id
              if best_visual_id and best_visual_id.startswith("Unknown"):
                   rospy.loginfo(f"Cross-modal merge (Audio prio): {best_visual_id} -> {recognized_voice_id}")
                   pdata = self.identities.get(best_visual_id)
                   self._merge_track_into_person(best_visual_id, recognized_voice_id, new_bbox=pdata.get('bbox') if pdata else None)

         #Strong visual match
         elif best_visual_id:
              speak_prob = self.identities[best_visual_id].get('speaking_prob', 0.0)
              if speak_prob > 0.6: 
                  speaker_label = best_visual_id
                  if best_visual_id.startswith("Unknown"):
                       pass
              else:
                   speaker_label = "Unknown (Silent?)"

         #No voice or visual matches
         else:
              speaker_label = "Unknown"

         return speaker_label

    def _publish_result(self, voice_prob, doa, speaker_label, best_visual_id):
         elapsed_time = rospy.get_time() - self.start_time
         doa_str = f"{doa:6.1f}" if doa is not None else " N/A"
         msg_str = f"[{elapsed_time:8.3f}s] VAD: {voice_prob:.2f} | DOA: {doa_str}"
         msg_str += f" -> Speaker: {speaker_label}"

         if speaker_label.startswith("Person"):
             with self.lock:
                  is_on_screen = self.identities.get(speaker_label, {}).get('on_screen', False)
             if not is_on_screen:
                  msg_str += " (Off-screen)"

         try:
              self.pub.publish(String(data=msg_str))
              rospy.loginfo(msg_str)
         except Exception as e:
              rospy.logwarn(f"Failed to publish message: {e}")


    """ID Management and Recognition"""

    def _create_unknown_identity(self, bbox):
        uid = f"Unknown {self.next_unknown_id}"
        self.next_unknown_id += 1
        self.identities[uid] = {
            'id': uid,
            'bbox': np.array(bbox).astype(int), 
            'inactive_frames': 0,
            'on_screen': True, 
            'speaking_prob': 0.0,
            'audio_buffer': np.array([], dtype=np.int16),
            'face_embeddings': deque(maxlen=self.config['max_embeddings_per_person']),
            'voice_embeddings': deque(maxlen=self.config['max_embeddings_per_person'])
        }
        rospy.logdebug(f"Created new track: {uid}")
        return uid

    def _update_speaking_probs(self, doa, voice_prob):
        """Updates speaking probability based on DOA and VAD."""
        alpha = self.config['speaking_prob_smoothing']

        doa_sigma = self.config.get('doa_sigma', 15.0)

        for pid, pdata in self.identities.items():
            old_prob = float(pdata.get('speaking_prob', 0.0))
            new_prob = 0.0

            if pdata.get('on_screen', False) and ('angle_deg' in pdata):
                angle_diff = float(pdata['angle_deg']) - float(doa)
                #Gaussian probability
                prob_doa = math.exp(-0.5 * (angle_diff / doa_sigma)**2)
                current_prob_estimate = 0.6 * prob_doa + 0.4 * voice_prob
                new_prob = alpha * old_prob + (1 - alpha) * current_prob_estimate
            else:
                new_prob = alpha * old_prob

            pdata['speaking_prob'] = max(0.0, min(1.0, new_prob))

    def _find_best_visual_match(self):
        best_id, max_prob = None, self.config['vad_threshold']
        for pid, pdata in self.identities.items():
            if pdata.get('on_screen', False):
                prob = float(pdata.get('speaking_prob', 0.0))
                if prob > max_prob:
                    max_prob = prob
                    best_id = pid
        return best_id

    def _enroll_new_person_from_unknown(self, unknown_id):
        if unknown_id not in self.identities: return
        track = self.identities[unknown_id]
        avg_face_emb = _median_embedding(list(track['face_embeddings']))
        audio_buffer = track.get('audio_buffer', np.array([], dtype=np.int16))
        voice_emb = self.audio_processor._get_voice_embedding(audio_buffer)

        if avg_face_emb is None or voice_emb is None:
             rospy.logdebug(f"Enrollment {unknown_id}: Missing face/voice median.")
             return

        recognized_id = self._recognize_person(face_embedding=avg_face_emb, voice_embedding=voice_emb)
        if recognized_id != "Unknown":
            rospy.loginfo(f"Enrollment check: {unknown_id} matches existing {recognized_id}. Merging.")
            self._merge_track_into_person(unknown_id, recognized_id, new_bbox=track.get('bbox'))
            return

        new_person_id = f"Person {self.next_person_id}"
        self.next_person_id += 1
        rospy.loginfo(f"--- Enrolling {unknown_id} as new identity: {new_person_id} ---")

        self.identities[new_person_id] = {
            'id': new_person_id,
            'bbox': track.get('bbox', np.array([0,0,0,0], dtype=int)),
            'inactive_frames': track.get('inactive_frames', 0),
            'on_screen': track.get('on_screen', False),
            'speaking_prob': track.get('speaking_prob', 0.0),
            'audio_buffer': np.array([], dtype=np.int16), #Clear buffer
            'face_embeddings': deque([avg_face_emb], maxlen=self.config['max_embeddings_per_person']),
            'voice_embeddings': deque([voice_emb], maxlen=self.config['max_embeddings_per_person'])
        }

        del self.identities[unknown_id]

    def _merge_track_into_person(self, track_id, person_id, new_bbox=None):
        if track_id not in self.identities: return
        if person_id not in self.identities: self._create_person_identity(person_id)

        track = self.identities[track_id]
        person = self.identities[person_id]

        if new_bbox is not None: person['bbox'] = np.array(new_bbox).astype(int)
        person['on_screen'] = track.get('on_screen', person.get('on_screen', False))
        person['inactive_frames'] = 0 #Reset inactivity

        merged_face_count, merged_voice_count = 0, 0
        for emb in list(track.get('face_embeddings', [])):
            if emb is not None:
                #Helper functions for normalising
                norm_emb = _l2_normalize(emb)
                if norm_emb is not None:
                    person['face_embeddings'].append(norm_emb)
                    merged_face_count += 1
        for vemb in list(track.get('voice_embeddings', [])):
            if vemb is not None:

                norm_vemb = _l2_normalize(vemb)
                if norm_vemb is not None:
                    person['voice_embeddings'].append(norm_vemb)
                    merged_voice_count += 1
        rospy.logdebug(f"Merged {merged_face_count}/{merged_voice_count} face/voice emb from {track_id} into {person_id}.")
        del self.identities[track_id]

    def _create_person_identity(self, person_id):
        if person_id in self.identities:
            return self.identities[person_id]
        rospy.loginfo(f"Creating new identity container for {person_id}")
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

    def _recognize_enrolled_by_face(self, face_embedding):
        if face_embedding is None: return None

        face_emb_norm = _l2_normalize(face_embedding)
        if face_emb_norm is None: return None

        best_person_id, best_sim = None, self.config['face_reid_min_conf']

        for pid, pdata in self.identities.items():
            if not pid.startswith("Person") or len(pdata['face_embeddings']) == 0: continue
            ref_face = _median_embedding(list(pdata['face_embeddings']))
            if ref_face is None: continue
            sim = _cosine_similarity(face_emb_norm, ref_face)
            if sim > best_sim:
                best_sim = sim
                best_person_id = pid
        return best_person_id

    def _recognize_person_by_voice(self, voice_embedding):
        if voice_embedding is None: return None

        voice_emb_norm = _l2_normalize(voice_embedding)
        if voice_emb_norm is None: return None

        best_person_id, best_sim = None, self.config['voice_rec_threshold']

        for pid, pdata in self.identities.items():
            if not pid.startswith("Person") or len(pdata['voice_embeddings']) == 0: continue
            ref_voice = _median_embedding(list(pdata['voice_embeddings']))
            if ref_voice is None: continue
            sim = _cosine_similarity(voice_emb_norm, ref_voice)
            if sim > best_sim:
                best_sim = sim
                best_person_id = pid
        return best_person_id

    def _recognize_person(self, face_embedding=None, voice_embedding=None):
        if face_embedding is None and voice_embedding is None: return "Unknown"

        face_emb_norm = _l2_normalize(face_embedding) if face_embedding is not None else None
        voice_emb_norm = _l2_normalize(voice_embedding) if voice_embedding is not None else None

        best_match_id = "Unknown"
        #High threshold for combined check
        highest_score = max(self.config['face_rec_threshold'], self.config['voice_rec_threshold']) - 0.01

        for person_id, identity in self.identities.items():
            if not person_id.startswith("Person"): continue
            face_sim, voice_sim = 0.0, 0.0

            if face_emb_norm is not None and len(identity['face_embeddings']) > 0:
                ref_face = _median_embedding(list(identity['face_embeddings']))
                if ref_face is not None: face_sim = _cosine_similarity(face_emb_norm, ref_face)
            if voice_emb_norm is not None and len(identity['voice_embeddings']) > 0:
                ref_voice = _median_embedding(list(identity['voice_embeddings']))
                if ref_voice is not None: voice_sim = _cosine_similarity(voice_emb_norm, ref_voice)

            current_max_sim = max(face_sim, voice_sim)
            if current_max_sim > highest_score:
                 #Check against individual threshold
                 if (current_max_sim == face_sim and face_sim >= self.config['face_rec_threshold']) or \
                    (current_max_sim == voice_sim and voice_sim >= self.config['voice_rec_threshold']):
                      highest_score, best_match_id = current_max_sim, person_id
        return best_match_id

    def _update_identity(self, person_id, face_embedding=None, voice_embedding=None):
        if person_id not in self.identities:
            rospy.logwarn(f"Attempted to update non-existent identity: {person_id}")
            return

        identity = self.identities[person_id]
        if face_embedding is not None:
            fe_norm = _l2_normalize(face_embedding)
            if fe_norm is not None: identity['face_embeddings'].append(fe_norm)
        if voice_embedding is not None:
            ve_norm = _l2_normalize(voice_embedding)
            if ve_norm is not None: identity['voice_embeddings'].append(ve_norm)