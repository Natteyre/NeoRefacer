import cv2
import onnxruntime as rt
import sys
sys.path.insert(1, './recognition')
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import os.path as osp
import os
import requests
from tqdm import tqdm
import ffmpeg
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from insightface.model_zoo.inswapper import INSwapper
import psutil
from enum import Enum
from insightface.app.common import Face
from insightface.utils.storage import ensure_available
import re
import subprocess
from PIL import Image
import numpy as np
import time
from codeformer_wrapper import enhance_image, enhance_image_memory
import tempfile

gc = __import__('gc')

# Preload NVIDIA DLLs if Windows
if sys.platform in ("win32", "win64"):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin")
            os.add_dll_directory(r"C:\\Program Files\\NVIDIA\\CUDNN\\v9.4\\bin\\12.6")
        except Exception as e:
            print(f"[INFO] Failed to add CUDA or CUDNN DLL directory: {e}")
            print("[INFO] This error can be ignored if running in CPU mode. Otherwise, make sure the paths are correct.")

    if hasattr(rt, "preload_dlls"):
        rt.preload_dlls()

class RefacerMode(Enum):
    CPU, CUDA, COREML, TENSORRT = range(1, 5)

class Refacer:
    def __init__(self, force_cpu=False, colab_performance=False):
        self.disable_similarity = False
        self.multiple_faces_mode = False
        self.first_face = False
        self.force_cpu = force_cpu
        self.colab_performance = colab_performance
        self.use_num_cpus = mp.cpu_count()
        self.__check_encoders()
        self.__check_providers()
        self.total_mem = psutil.virtual_memory().total
        self.__init_apps()
       
    def _partial_face_blend(self, original_frame, swapped_frame, face):
        h_frame, w_frame = original_frame.shape[:2]
   
        x1, y1, x2, y2 = map(int, face.bbox)
        x1 = max(0, min(x1, w_frame-1))
        y1 = max(0, min(y1, h_frame-1))
        x2 = max(0, min(x2, w_frame))
        y2 = max(0, min(y2, h_frame))
   
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid bbox: {x1},{y1},{x2},{y2}")
            return swapped_frame
   
        w = x2 - x1
        h = y2 - y1
        cutoff = int(h * (1.0 - self.blend_height_ratio))
   
        swap_crop = swapped_frame[y1:y2, x1:x2].copy()
        orig_crop = original_frame[y1:y2, x1:x2].copy()
   
        mask = np.ones((h, w, 3), dtype=np.float32)
        transition = 40
   
        if cutoff < h:
            blend_start = max(cutoff - transition // 2, 0)
            blend_end = min(cutoff + transition // 2, h)
   
            if blend_end > blend_start:
                alpha = np.linspace(1.0, 0.0, blend_end - blend_start)[:, np.newaxis, np.newaxis]
                mask[blend_start:blend_end, :, :] = alpha
            mask[blend_end:, :, :] = 0.0
   
        blended_crop = (swap_crop.astype(np.float32) * mask + orig_crop.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
   
        blended_frame = swapped_frame.copy()
        blended_frame[y1:y2, x1:x2] = blended_crop
   
        return blended_frame
   
    def __download_with_progress(self, url, output_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}")

        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("ERROR, something went wrong downloading the model!")

    def __check_providers(self):
        available_providers = rt.get_available_providers()
        print(f"[Refacer] Available ONNX Runtime providers: {available_providers}")

        if self.force_cpu:
            self.providers = ['CPUExecutionProvider']
            print("[Refacer] 🔴  force_cpu=True → using CPU only")
        else:
            # Prefer faster execution providers in order – **exclude TensorRT** to avoid interference
            self.providers = []
            for p in ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']:
                if p in available_providers:
                    self.providers.append(p)
            print(f"[Refacer] 🎯  Provider priority list: {self.providers}")

        self.sess_options = rt.SessionOptions()
        self.sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        self.sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable silent CPU fallback via SessionOptions if available (ORT >=1.22)
        if hasattr(self.sess_options, 'add_config_entry'):
            self.sess_options.add_config_entry('session.disable_cpu_ep_fallback', '1')

        # ----- decide mode WITHOUT creating a test session (model not downloaded yet) -----
        cuda_available = 'CUDAExecutionProvider' in self.providers
        coreml_available = 'CoreMLExecutionProvider' in self.providers

        if cuda_available:
            self.mode = RefacerMode.CUDA
            self.use_num_cpus = 2
            self.sess_options.intra_op_num_threads = 1
            print("[Refacer] 🟢  GPU mode selected (CUDAExecutionProvider available)")
        elif coreml_available:
            self.mode = RefacerMode.COREML
            self.use_num_cpus = max(mp.cpu_count() - 1, 1)
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus / 2)
            print("[Refacer] 🟢  CoreML mode selected")
        elif self.colab_performance:
            self.mode = RefacerMode.TENSORRT
            self.use_num_cpus = max(mp.cpu_count() - 1, 1)
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus / 2)
            print("[Refacer] 🟡  TensorRT/colab_performance mode selected")
        else:
            self.mode = RefacerMode.CPU
            self.use_num_cpus = max(mp.cpu_count() - 1, 1)
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus / 2)
            print("[Refacer] 🔴  CPU mode selected")

        print(f"[Refacer] Mode: {self.mode}")

    def _make_session(self, model_path: str, name: str):
        """Create InferenceSession with CUDA provider options; on failure, print error and fallback to CPU."""
        # CUDA provider options – minimal set that works on Colab T4
        cuda_opts = {
            'device_id': 0,
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,   # 4 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        providers = []
        if 'CUDAExecutionProvider' in self.providers:
            providers.append(('CUDAExecutionProvider', cuda_opts))
        providers.append('CPUExecutionProvider')

        try:
            sess = rt.InferenceSession(model_path, self.sess_options, providers=providers)
            actual = sess.get_providers()
            gpu_used = any('CUDA' in p or 'Tensorrt' in p for p in actual)
            print(f"[Refacer]   {name:15s} → actual providers: {actual}  {'🟢 GPU' if gpu_used else '🔴 CPU only'}")
            if not gpu_used and 'CUDAExecutionProvider' in self.providers:
                print(f"[Refacer]   ⚠️  {name}: CUDA EP was requested but session fell back to CPU. "
                      "This usually means CUDA EP initialization failed silently (cuDNN version mismatch).")
            return sess
        except Exception as e:
            print(f"[Refacer]   ❌  {name}: Failed to create session with providers {providers}: {e}")
            print(f"[Refacer]   🔄  Falling back to CPUExecutionProvider only.")
            sess = rt.InferenceSession(model_path, self.sess_options, providers=['CPUExecutionProvider'])
            actual = sess.get_providers()
            print(f"[Refacer]   {name:15s} → actual providers: {actual}  🔴 CPU only (fallback)")
            return sess

    def __init_apps(self):
        assets_dir = ensure_available('models', 'buffalo_l', root='~/.insightface')

        model_path = os.path.join(assets_dir, 'det_10g.onnx')
        sess_face = self._make_session(model_path, 'FaceDetector')
        self.face_detector = SCRFD(model_path, sess_face)
        self.face_detector.prepare(0, input_size=(640, 640))

        model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
        sess_rec = self._make_session(model_path, 'FaceRecognizer')
        self.rec_app = ArcFaceONNX(model_path, sess_rec)
        self.rec_app.prepare(0)

        model_dir = os.path.join('weights', 'inswapper')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'inswapper_128.onnx')

        if not os.path.exists(model_path):
            print(f"Model {model_path} not found. Downloading from HuggingFace...")
            url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
            try:
                self.__download_with_progress(url, model_path)
                print(f"Downloaded {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download {model_path}. Error: {e}")

        sess_swap = self._make_session(model_path, 'FaceSwapper')
        self.face_swapper = INSwapper(model_path, sess_swap)

    # ----- the rest of the original methods (unchanged) -----
    def prepare_faces(self, faces, disable_similarity=False, multiple_faces_mode=False):
        self.replacement_faces = []
        self.disable_similarity = disable_similarity
        self.multiple_faces_mode = multiple_faces_mode

        for face in faces:
            if "destination" not in face or face["destination"] is None:
                print("Skipping face config: No destination face provided.")
                continue

            _faces = self.__get_faces(face['destination'], max_num=1)
            if len(_faces) < 1:
                raise Exception('No face detected on "Destination face" image')

            if multiple_faces_mode:
                self.replacement_faces.append((None, _faces[0], 0.0))
            else:
                if "origin" in face and face["origin"] is not None and not disable_similarity:
                    face_threshold = face['threshold']
                    bboxes1, kpss1 = self.face_detector.autodetect(face['origin'], max_num=1)
                    if len(kpss1) < 1:
                        raise Exception('No face detected on "Face to replace" image')
                    feat_original = self.rec_app.get(face['origin'], kpss1[0])
                else:
                    face_threshold = 0
                    self.first_face = True
                    feat_original = None

                self.replacement_faces.append((feat_original, _faces[0], face_threshold))

    def __get_faces(self, frame, max_num=0):
        bboxes, kpss = self.face_detector.detect(frame, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            face.embedding = self.rec_app.get(frame, kps)
            ret.append(face)
        return ret

    def process_first_face(self, frame):
        faces = self.__get_faces(frame, max_num=0)
        if not faces:
            return frame
   
        if self.disable_similarity:
            for face in faces:
                swapped = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=True)
                if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                    self.blend_height_ratio = self.partial_reface_ratio
                    frame = self._partial_face_blend(frame, swapped, face)
                else:
                    frame = swapped
        return frame

    def process_faces(self, frame):
        faces = self.__get_faces(frame, max_num=0)
        if not faces:
            return frame
 
        faces = sorted(faces, key=lambda face: face.bbox[0])
 
        if self.multiple_faces_mode:
            for idx, face in enumerate(faces):
                if idx >= len(self.replacement_faces):
                    break
                swapped = self.face_swapper.get(frame, face, self.replacement_faces[idx][1], paste_back=True)
                if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                    self.blend_height_ratio = self.partial_reface_ratio
                    frame = self._partial_face_blend(frame, swapped, face)
                else:
                    frame = swapped
        elif self.disable_similarity:
            for face in faces:
                swapped = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=True)
                if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                    self.blend_height_ratio = self.partial_reface_ratio
                    frame = self._partial_face_blend(frame, swapped, face)
                else:
                    frame = swapped
        else:
            for rep_face in self.replacement_faces:
                for i in range(len(faces) - 1, -1, -1):
                    sim = self.rec_app.compute_sim(rep_face[0], faces[i].embedding)
                    if sim >= rep_face[2]:
                        swapped = self.face_swapper.get(frame, faces[i], rep_face[1], paste_back=True)
                        if hasattr(self, 'partial_reface_ratio') and self.partial_reface_ratio > 0.0:
                            self.blend_height_ratio = self.partial_reface_ratio
                            frame = self._partial_face_blend(frame, swapped, faces[i])
                        else:
                            frame = swapped
                        del faces[i]
                        break
        return frame

    def reface_group(self, faces, frames, output):
        with ThreadPoolExecutor(max_workers=self.use_num_cpus) as executor:
            if self.first_face:
                results = list(tqdm(executor.map(self.process_first_face, frames), total=len(frames), desc="Processing frames"))
            else:
                results = list(tqdm(executor.map(self.process_faces, frames), total=len(frames), desc="Processing frames"))
            for result in results:
                output.write(result)

    def __check_video_has_audio(self, video_path):
        self.video_has_audio = False
        probe = ffmpeg.probe(video_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream is not None:
            self.video_has_audio = True

    def reface(self, video_path, faces, preview=False, disable_similarity=False, multiple_faces_mode=False, partial_reface_ratio=0.0):
        original_name = osp.splitext(osp.basename(video_path))[0]
        timestamp = str(int(time.time()))
        filename = f"{original_name}_preview.mp4" if preview else f"{original_name}_{timestamp}.mp4"
   
        self.__check_video_has_audio(video_path)
   
        if preview:
            os.makedirs("output/preview", exist_ok=True)
            output_video_path = os.path.join('output', 'preview', filename)
        else:
            os.makedirs("output", exist_ok=True)
            output_video_path = os.path.join('output', filename)
   
        self.prepare_faces(faces, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode)
        self.first_face = False if multiple_faces_mode else (faces[0].get("origin") is None or disable_similarity)
        self.partial_reface_ratio = partial_reface_ratio
   
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
   
        frames = []
        frame_index = 0
        skip_rate = 10 if preview else 1
   
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break
                if frame_index % skip_rate == 0:
                    frames.append(frame)
                    if len(frames) > 300:
                        self.reface_group(faces, frames, output)
                        frames = []
                        gc.collect()
                frame_index += 1
                pbar.update()
   
        cap.release()
        if frames:
            self.reface_group(faces, frames, output)
        output.release()
   
        converted_path = self.__convert_video(video_path, output_video_path, preview=preview)
   
        if video_path.lower().endswith(".gif"):
            if preview:
                gif_output_path = os.path.join("output", "preview", os.path.basename(converted_path).replace(".mp4", ".gif"))
            else:
                gif_output_path = os.path.join("output", "gifs", os.path.basename(converted_path).replace(".mp4", ".gif"))
   
            self.__generate_gif(converted_path, gif_output_path)
            return converted_path, gif_output_path
   
        return converted_path, None
   
 
 
    def __generate_gif(self, video_path, gif_output_path):
        os.makedirs(os.path.dirname(gif_output_path), exist_ok=True)
        print(f"Generating GIF at {gif_output_path}")
        (
            ffmpeg
            .input(video_path)
            .output(gif_output_path, vf='fps=10,scale=512:-1:flags=lanczos', loop=0)
            .overwrite_output()
            .run(quiet=True)
        )

    def __convert_video(self, video_path, output_video_path, preview=False):
        if self.video_has_audio and not preview:
            new_path = output_video_path + str(random.randint(0, 999)) + "_c.mp4"
            in1 = ffmpeg.input(output_video_path)
            in2 = ffmpeg.input(video_path)
            out = ffmpeg.output(in1.video, in2.audio, new_path, video_bitrate=self.ffmpeg_video_bitrate, vcodec=self.ffmpeg_video_encoder)
            out.run(overwrite_output=True, quiet=True)
        else:
            new_path = output_video_path
        print(f"Refaced video saved at: {os.path.abspath(new_path)}")
        return new_path

    def reface_image(self, image_path, faces, disable_similarity=False, multiple_faces_mode=False, partial_reface_ratio=0.0):
         self.prepare_faces(faces, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode)
         self.first_face = False if multiple_faces_mode else (faces[0].get("origin") is None or disable_similarity)
         self.partial_reface_ratio = partial_reface_ratio
 
         ext = osp.splitext(image_path)[1].lower()
         os.makedirs("output", exist_ok=True)
         original_name = osp.splitext(osp.basename(image_path))[0]
         timestamp = str(int(time.time()))
 
         if ext in ['.tif', '.tiff']:
             pil_img = Image.open(image_path)
             frames = []
 
             page_count = 0
             try:
                 while True:
                     pil_img.seek(page_count)
                     page_count += 1
             except EOFError:
                 pass
 
             pil_img = Image.open(image_path)
 
             with tqdm(total=page_count, desc="Processing TIFF pages") as pbar:
                 for page in range(page_count):
                     pil_img.seek(page)
                     bgr_image = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                     refaced_bgr = self.process_first_face(bgr_image.copy()) if self.first_face else self.process_faces(bgr_image.copy())
                     enhanced_bgr = enhance_image_memory(refaced_bgr)
                     enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                     enhanced_pil = Image.fromarray(enhanced_rgb)
                     frames.append(enhanced_pil)
                     pbar.update(1)
 
             output_path = os.path.join("output", f"{original_name}_{timestamp}.tif")
             frames[0].save(output_path, save_all=True, append_images=frames[1:], compression="tiff_deflate")
             print(f"Saved multipage refaced TIFF to {output_path}")
             return output_path
 
         else:
             bgr_image = cv2.imread(image_path)
             if bgr_image is None:
                 raise ValueError("Failed to read input image")
 
             refaced_bgr = self.process_first_face(bgr_image.copy()) if self.first_face else self.process_faces(bgr_image.copy())
             refaced_rgb = cv2.cvtColor(refaced_bgr, cv2.COLOR_BGR2RGB)
             pil_img = Image.fromarray(refaced_rgb)
             filename = f"{original_name}_{timestamp}.jpg"
             output_path = os.path.join("output", filename)
             pil_img.save(output_path, format='JPEG', quality=100, subsampling=0)
             output_path = enhance_image(output_path)
             print(f"Saved refaced image to {output_path}")
             return output_path


    def extract_faces_from_image(self, image_path, max_faces=5):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Failed to read input image for face extraction.")

        faces = self.__get_faces(frame, max_num=max_faces)
        cropped_faces = []

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])

            cropped = frame[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
            pil_img.save(temp_file.name)
            cropped_faces.append(temp_file.name)

            if len(cropped_faces) >= max_faces:
                break

        return cropped_faces

    def __try_ffmpeg_encoder(self, vcodec):
        command = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=1280x720:rate=30', '-vcodec', vcodec, 'testsrc.mp4']
        try:
            subprocess.run(command, check=True, capture_output=True).stderr
        except subprocess.CalledProcessError:
            return False
        return True

    def __check_encoders(self):
        self.ffmpeg_video_encoder = 'libx264'
        self.ffmpeg_video_bitrate = '0'
        pattern = r"encoders: ([a-zA-Z0-9_]+(?: [a-zA-Z0-9_]+)*)"
        command = ['ffmpeg', '-codecs', '--list-encoders']
        commandout = subprocess.run(command, check=True, capture_output=True).stdout
        result = commandout.decode('utf-8').split('\n')
        for r in result:
            if "264" in r:
                encoders = re.search(pattern, r)
                if encoders:
                    for v_c in Refacer.VIDEO_CODECS:
                        for v_k in encoders.group(1).split(' '):
                            if v_c == v_k and self.__try_ffmpeg_encoder(v_k):
                                self.ffmpeg_video_encoder = v_k
                                self.ffmpeg_video_bitrate = Refacer.VIDEO_CODECS[v_k]
                                return

    VIDEO_CODECS = {
        'h264_videotoolbox': '0',
        'h264_nvenc': '0',
        'libx264': '0'
    }