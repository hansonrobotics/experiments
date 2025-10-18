import rospy, os, sys
import time, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.signal import butter, sosfilt, resample_poly
import pyaudio
from speechbrain.inference.speaker import SpeakerRecognition

from helper import _l2_normalize

class AudioProcessor:
    def __init__(self, config, parent_node):
        rospy.loginfo("Initializing Audio Processing")
        self.parent = parent_node
        self.config = config

        # Hardware
        self.p = None
        self.stream = None
        self.bandpass_sos = None

        # Models
        self.vad_model = None
        self.speaker_recognizer = None
       
        # States and Buffer
        self.prev_doa = 0.0
        self.silero_buf = np.array([], dtype=np.float32)
        self.last_voice_time = 0.0
        self.audio_buffer_for_recognition = np.array([], dtype=np.int16)

    def initialize(self):
        self.p = pyaudio.PyAudio() 
        self._initialize_audio_models()
        self._initialize_vad()
        self._initialize_audio_stream()

    def _initialize_audio_models(self):
        try:
            project_root = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    def _initialize_audio_stream(self):
        rospy.loginfo("Initializing PyAudio...")
        device_idx, dev_name = self._find_audio_device()
        if device_idx is None:
            rospy.logerr("Audio device not found.")
            if self.parent:
                self.parent.stop_even.set()
            sys.exit(1)

        rospy.loginfo(f"Using audio device: {dev_name} (index {device_idx})")
        self.bandpass_sos = self._design_bandpass_filter()
            
        self.stream = self.p.open(format=pyaudio.paInt16,
                                 channels=self.config['channels'],
                                 rate=self.config['sample_rate'],
                                 input=True,
                                 input_device_index=device_idx,
                                 frames_per_buffer=self.config['window_size'])

    def process_chunk(self):
        """Processes one chunk of audio. Called repeatedly by the parent node."""
        if not self.stream or self.parent.stop_event.is_set():
            return None, None, None 
        
        try:
            audio_chunk = self.stream.read(self.config['window_size'], exception_on_overflow=False)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        except IOError:
             rospy.logwarn("Audio input overflowed.")
             return None, None, None
        except Exception as e:
            rospy.logerr(f"Error reading audio stream: {e}")
            self.parent.stop_event.set()
            return None, None, None

        if audio_array.size == 0:
            return None, None, None

        voice_prob = self._process_vad(audio_array)
        mono_audio_16k = self._resample_audio_for_rec(audio_array)

        # Buffer
        self.audio_buffer_for_recognition = np.concatenate((self.audio_buffer_for_recognition, mono_audio_16k))
        max_buffer_len = int(self.config['enrollment_audio_seconds'] * self.config['vad_rate'] * 1.5)
        if len(self.audio_buffer_for_recognition) > max_buffer_len:
            self.audio_buffer_for_recognition = self.audio_buffer_for_recognition[-max_buffer_len:]

        doa = None
        voice_emb = None

        if voice_prob >= self.config['vad_threshold']:
            #DOA if speech detected
            left, right = audio_array[0::2], audio_array[1::2]
            doa = self._calculate_doa(left, right)

            # Is audio buffered enough for recognition
            if len(self.audio_buffer_for_recognition) >= int(self.config['enrollment_audio_seconds'] * self.config['vad_rate']):
                voice_emb = self._get_voice_embedding(self.audio_buffer_for_recognition)
                self.audio_buffer_for_recognition = np.array([], dtype=np.int16)
        else:
             pass

        return voice_prob, doa, voice_emb


    def _get_voice_embedding(self, audio_buffer):
       #Generates a voice embedding from buffer
        if self.speaker_recognizer is None: return None
        if audio_buffer is None or len(audio_buffer) < self.config['vad_rate'] * 0.5: # Need at least 0.5 sec?
            return None
        try:
            wav = audio_buffer.astype(np.float32) / 32768.0
            wav = np.clip(wav, -1.0, 1.0)
            wav_t = torch.tensor(wav).unsqueeze(0)
            with torch.no_grad():
                emb = self.speaker_recognizer.encode_batch(wav_t)
            if isinstance(emb, torch.Tensor):
                 emb_np = emb.squeeze().cpu().numpy()
            else:
                rospy.logwarn("Unexpected embedding type from SpeechBrain")
                return None

            if emb_np is None or emb_np.size == 0: return None

        except Exception as e:
            rospy.logwarn_throttle(5, f"Could not generate voice embedding: {e}")
            return None


    def _resample_audio_for_rec(self, audio_array):
        #Resample stereo 44.1kHz to mono 16kHz for VAD and Recognition
        if audio_array.size < 2: return np.array([], dtype=np.int16)
        left = audio_array[0::2].astype(np.float32)
        right = audio_array[1::2].astype(np.float32)

        mono_44k = (left + right) / 2.0
        target_rate = int(self.config['vad_rate'])
        current_rate = int(self.config['sample_rate'])
        if target_rate == current_rate:
             resampled_f32 = mono_44k
        else:
            resampled_f32 = resample_poly(mono_44k, target_rate, current_rate)

        # Convert to int16
        clipped_f32 = np.clip(resampled_f32, -32768.0, 32767.0)
        return clipped_f32.astype(np.int16)

    def _process_vad(self, audio_array):
        """Processes audio through Silero VAD, returns voice probability."""
        if self.vad_model is None: return 0.0

    
        mono_16k_f32 = self._resample_audio_for_rec(audio_array).astype(np.float32) / 32768.0
        mono_16k_f32 = np.clip(mono_16k_f32, -1.0, 1.0)

        self.silero_buf = np.concatenate([self.silero_buf, mono_16k_f32])
        frame_size = int(self.config['silero_frame_size']) # Ensure int
        max_voice_prob = 0.0

        with torch.no_grad():
            while len(self.silero_buf) >= frame_size:
                chunk = self.silero_buf[:frame_size]
                self.silero_buf = self.silero_buf[frame_size:]
                try:
                    vad_input = torch.from_numpy(chunk)
                    # Check expected input format (maybe needs batch dim?)
                    new_prob = self.vad_model(vad_input, self.config['vad_rate']).item()
                    result_tensor = self.vad_model(vad_input, int(self.config['vad_rate']))
                    new_prob = result_tensor.item() # Extract scalar probability

                    max_voice_prob = max(max_voice_prob, new_prob)
                except Exception as e:
                     rospy.logwarn_throttle(10, f"VAD processing error: {e}")
                     pass # Continue processing other chunks


        # Reset VAD state
        now = rospy.get_time() if rospy.core.is_initialized() else time.time()
        if max_voice_prob >= self.config['vad_threshold']:
            self.last_voice_time = now
        elif (now - self.last_voice_time) > self.config['silence_reset_s']:
            try:
                if hasattr(self.vad_model, 'reset_states'):
                    self.vad_model.reset_states()
                    rospy.loginfo("VAD state reset due to silence.")
            except Exception as e:
                 rospy.logwarn(f"Could not reset VAD state: {e}")


        return max_voice_prob


    def _calculate_doa(self, left, right):
        """Calculates Direction of Arrival using GCC-PHAT."""
        if self.bandpass_sos is None or left.size != right.size or left.size == 0:
            return None

        try:
            #Filter
            s1_filt = sosfilt(self.bandpass_sos, left.astype(np.float32))
            s2_filt = sosfilt(self.bandpass_sos, right.astype(np.float32))

            #Normalize
            n1 = np.max(np.abs(s1_filt)) + 1e-8
            n2 = np.max(np.abs(s2_filt)) + 1e-8
            s1_norm = s1_filt / n1
            s2_norm = s2_filt / n2

            #Compute cross-correlation
            corr = np.correlate(s1_norm, s2_norm, mode='full')
            delay_axis = np.arange(-(s1_norm.size - 1), s1_norm.size)

            #Basic GCC-PHAT
            #Phase transform
            S1 = np.fft.fft(s1_norm)
            S2 = np.fft.fft(s2_norm)

            R = S1 * np.conj(S2)
            
            denom = np.abs(R) + 1e-10
            G_phat = R / denom
            r_phat = np.fft.ifft(G_phat)
            corr_phat = np.abs(np.fft.fftshift(r_phat))

            delay_sample_phat = delay_axis[np.argmax(corr_phat)]

            #Check correlation peak quality (optional but good)
            peak_val = np.max(corr_phat)
            mean_val = np.mean(corr_phat)
            if peak_val < 3 * mean_val: # Threshold for reliable peak
               return float(self.prev_doa) # Return previous if peak is weak

            #Calculate time delay and angle
            max_delay_samples = int((self.config['mic_distance'] / self.config['speed_of_sound']) * self.config['sample_rate'])
            #Check if delay is possible
            if abs(delay_sample_phat) > max_delay_samples:
                 return float(self.prev_doa)

            time_delay = delay_sample_phat / self.config['sample_rate']

            sin_theta_arg = (time_delay * self.config['speed_of_sound']) / self.config['mic_distance']
            sin_theta = np.clip(sin_theta_arg, -1.0, 1.0)

            doa = float(np.degrees(np.arcsin(sin_theta)))

            #Smoothing
            current_prev_doa = self.prev_doa if self.prev_doa is not None else 0.0
            smoothed_doa = self.config['smoothing_factor'] * current_prev_doa + (1 - self.config['smoothing_factor']) * doa
            self.prev_doa = smoothed_doa
            return smoothed_doa

        except Exception as e:
            rospy.logwarn_throttle(5, f"DOA calculation error: {e}")
            #Return previous doa
            return float(self.prev_doa) if self.prev_doa is not None else 0.0


    def _find_audio_device(self):
        target_name = self.config.get('target_device_name', '')
        num_devices = self.p.get_device_count()
        for i in range(num_devices):
            try:
                dev = self.p.get_device_info_by_index(i)
                name = dev.get('name', '')
                max_channels = dev.get('maxInputChannels', 0)
                if target_name in name and max_channels >= self.config['channels']:
                    return i, name
            except Exception as e:
                 rospy.logerr(f"Error getting info for device {i}: {e}")

        #If not found, use default
        rospy.logwarn(f"Target audio device '{target_name}' not found. Trying default input device.")
        try:
            default_dev_info = self.p.get_default_input_device_info()
            default_idx = default_dev_info['index']
            default_name = default_dev_info['name']
            default_channels = default_dev_info.get('maxInputChannels', 0)
            if default_channels >= self.config['channels']:
                rospy.loginfo(f"Using default input device: {default_name} (index {default_idx})")
                return default_idx, default_name
            else:
                 rospy.logerr(f"Default input device '{default_name}' only has {default_channels} channels, need {self.config['channels']}.")
                 return None, None
        except Exception as e:
            rospy.logerr(f"Could not get default input device info: {e}")
            return None, None

    def _design_bandpass_filter(self):
        try:
            nyq = 0.5 * self.config['sample_rate']
            low = max(0.01, self.config['bandpass_lowcut'] / nyq)
            high = min(0.99, self.config['bandpass_highcut'] / nyq)
            if low >= high:
                 rospy.logerr(f"Bandpass lowcut ({low*nyq} Hz) must be less than highcut ({high*nyq} Hz).")
                 return None
            order = int(self.config['bandpass_order'])
            return butter(order, [low, high], btype='band', output='sos')
        except Exception as e:
            rospy.logerr(f"Error designing bandpass filter: {e}")
            return None


    def stop(self):
        rospy.loginfo("Stopping Audio Subsystem...")
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                rospy.loginfo("Audio stream closed.")
            except Exception as e:
                rospy.logerr(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        if self.p:
            try:
                self.p.terminate()
                rospy.loginfo("PyAudio terminated.")
            except Exception as e:
                rospy.logerr(f"Error terminating PyAudio: {e}")
            finally:
                self.p = None