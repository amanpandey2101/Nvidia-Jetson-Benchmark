import os
import time
import ctypes
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger("benchmark.trt")

# Monkey-patch np.bool which was removed in NumPy 1.24+ but is required by TensorRT 8.x
if not hasattr(np, "bool"):
    np.bool = bool

import tensorrt as trt

class CudaManager:
    def __init__(self):
        self.lib = None
        # Try loading standard paths
        libs_to_try = [
            "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
            "libcudart.so.11",
            "libcudart.so.12",
            "cudart64_110.dll",
            "cudart64_120.dll",
            "cudart.dll"
        ]
        for libname in libs_to_try:
            try:
                self.lib = ctypes.CDLL(libname)
                logger.info(f"Loaded CUDA runtime: {libname}")
                break
            except Exception:
                continue
                
        if self.lib is None:
            raise RuntimeError("CUDA Runtime Library (libcudart) not found. Cannot run native TensorRT.")
            
        # Define prototypes
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        
        self.lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        self.lib.cudaHostAlloc.restype = ctypes.c_int
        
        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cudaFreeHost.restype = ctypes.c_int
        
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        
        self.lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

    def malloc(self, size: int) -> int:
        ptr = ctypes.c_void_p()
        res = self.lib.cudaMalloc(ctypes.byref(ptr), size)
        if res != 0:
            raise RuntimeError(f"cudaMalloc failed: {res}")
        return ptr.value

    def free(self, ptr: int):
        if ptr:
            self.lib.cudaFree(ctypes.c_void_p(ptr))

    def host_alloc(self, size: int) -> int:
        ptr = ctypes.c_void_p()
        res = self.lib.cudaHostAlloc(ctypes.byref(ptr), size, 0)
        if res != 0:
            raise RuntimeError(f"cudaHostAlloc failed: {res}")
        return ptr.value

    def host_free(self, ptr: int):
        if ptr:
            self.lib.cudaFreeHost(ctypes.c_void_p(ptr))

    def memcpy(self, dst: int, src: int, size: int, kind: int):
        res = self.lib.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), size, kind)
        if res != 0:
            raise RuntimeError(f"cudaMemcpy failed: {res}")

    def memcpy_async(self, dst: int, src: int, size: int, kind: int, stream: int):
        res = self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), size, kind, ctypes.c_void_p(stream))
        if res != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed: {res}")

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        res = self.lib.cudaStreamCreate(ctypes.byref(stream))
        if res != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {res}")
        return stream.value

    def destroy_stream(self, stream: int):
        if stream:
            self.lib.cudaStreamDestroy(ctypes.c_void_p(stream))

    def synchronize_stream(self, stream: int):
        if stream:
            self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def preprocess_image(image_source, target_size=(640, 640)) -> np.ndarray:
    """
    Loads, letterboxes, normalizes, and reshapes an image to target_size float32.
    image_source can be a file path, file-like object, or PIL Image.
    """
    if isinstance(image_source, Image.Image):
        img = image_source.convert('RGB')
    else:
        img = Image.open(image_source).convert('RGB')
        
    iw, ih = img.size
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    
    # Image.BILINEAR is compatible with both older and newer versions of Pillow
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new('RGB', target_size, (114, 114, 114))
    new_img.paste(img_resized, ((tw - nw) // 2, (th - nh) // 2))
    
    arr = np.array(new_img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
    return np.ascontiguousarray(arr)


class TensorRTEngine:
    def __init__(self, engine_path: str, model_type: str = "yolov5s"):
        self.engine_path = engine_path
        self.model_type = model_type.lower()
        self.cuda = CudaManager()
        
        # Load engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        logger.info(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")
            
        self.trt_version = int(trt.__version__.split('.')[0])
        logger.info(f"Using TensorRT Version: {trt.__version__}")
        
        # Extract input shape from the engine dynamically
        self.input_shape = None
        if self.trt_version < 10:
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_shape = list(self.engine.get_binding_shape(i))
                    break
        else:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_shape = list(self.engine.get_tensor_shape(name))
                    break
        
        if self.input_shape is None:
            raise RuntimeError("Could not find any input tensor in the TensorRT engine.")
        logger.info(f"Engine input shape: {self.input_shape}")

        # Extract height and width from input shape
        if len(self.input_shape) == 4:
            # Layout is typically NCHW (e.g., [1, 3, 544, 960]) or NHWC (e.g., [1, 544, 960, 3])
            if self.input_shape[1] == 3: # NCHW
                self.input_height = self.input_shape[2]
                self.input_width = self.input_shape[3]
            elif self.input_shape[3] == 3: # NHWC
                self.input_height = self.input_shape[1]
                self.input_width = self.input_shape[2]
            else:
                self.input_height = self.input_shape[2]
                self.input_width = self.input_shape[3]
        elif len(self.input_shape) == 3:
            # Layout is CHW or HWC
            if self.input_shape[0] == 3: # CHW
                self.input_height = self.input_shape[1]
                self.input_width = self.input_shape[2]
            elif self.input_shape[2] == 3: # HWC
                self.input_height = self.input_shape[0]
                self.input_width = self.input_shape[1]
            else:
                self.input_height = self.input_shape[1]
                self.input_width = self.input_shape[2]
        else:
            # Fallback
            self.input_height = 640
            self.input_width = 640
        logger.info(f"Parsed dynamic input dimensions: height={self.input_height}, width={self.input_width}")

    def create_execution_context(self):
        return TensorRTContext(self)


class TensorRTContext:
    def __init__(self, parent: TensorRTEngine):
        self.parent = parent
        self.engine = parent.engine
        self.cuda = parent.cuda
        self.model_type = parent.model_type
        
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.create_stream()
        
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        
        self._allocate_buffers()

    def _allocate_buffers(self):
        if self.parent.trt_version < 10:
            # TensorRT 8.x binding-based allocation
            num_bindings = self.engine.num_bindings
            self.bindings = [0] * num_bindings
            
            for i in range(num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
                
                size = int(np.prod(shape))
                nbytes = size * np.dtype(dtype).itemsize
                
                # Allocate device memory
                d_ptr = self.cuda.malloc(nbytes)
                self.bindings[i] = d_ptr
                
                buffer_info = {
                    "index": i,
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "nbytes": nbytes,
                    "d_ptr": d_ptr,
                    "h_buf": None
                }
                
                if is_input:
                    # Input: Allocate host pinned memory
                    h_ptr = self.cuda.host_alloc(nbytes)
                    buffer_info["h_ptr"] = h_ptr
                    # Map input buffer
                    buffer_info["h_buf"] = np.ctypeslib.as_array(
                        (ctypes.c_byte * nbytes).from_address(h_ptr)
                    ).view(dtype).reshape(shape)
                    self.inputs[name] = buffer_info
                else:
                    # Output: Allocate host pinned memory
                    h_ptr = self.cuda.host_alloc(nbytes)
                    buffer_info["h_ptr"] = h_ptr
                    buffer_info["h_buf"] = np.ctypeslib.as_array(
                        (ctypes.c_byte * nbytes).from_address(h_ptr)
                    ).view(dtype).reshape(shape)
                    self.outputs[name] = buffer_info
        else:
            # TensorRT 10.x name-based tensor allocation
            num_io_tensors = self.engine.num_io_tensors
            
            for i in range(num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                
                is_input = (mode == trt.TensorIOMode.INPUT)
                size = int(np.prod(shape))
                nbytes = size * np.dtype(dtype).itemsize
                
                d_ptr = self.cuda.malloc(nbytes)
                h_ptr = self.cuda.host_alloc(nbytes)
                h_buf = np.ctypeslib.as_array(
                    (ctypes.c_byte * nbytes).from_address(h_ptr)
                ).view(dtype).reshape(shape)
                
                buffer_info = {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "nbytes": nbytes,
                    "d_ptr": d_ptr,
                    "h_ptr": h_ptr,
                    "h_buf": h_buf
                }
                
                if is_input:
                    self.inputs[name] = buffer_info
                    # Set execution shape
                    self.context.set_input_shape(name, shape)
                else:
                    self.outputs[name] = buffer_info
                
                # Bind tensor address
                self.context.set_tensor_address(name, d_ptr)

    def infer(self, preprocessed_image: np.ndarray, conf_threshold: float = 0.10, iou_threshold: float = 0.45) -> list:
        """
        Runs inference on the preprocessed image.
        Returns a list of detections: [{"class_id": int, "confidence": float, "box": [x1, y1, x2, y2]}]
        """
        # 1. Copy input to host pinned buffer
        input_name = list(self.inputs.keys())[0]
        input_info = self.inputs[input_name]
        
        # Ensure correct batch layout
        batch_size = input_info["shape"][0]
        # In this loader, we replicate the image to match the engine's batch size
        batch_data = np.repeat(preprocessed_image[np.newaxis, ...], batch_size, axis=0)
        
        # Copy to pinned host buffer
        np.copyto(input_info["h_buf"], batch_data)
        
        # 2. Transfer host input to device GPU
        self.cuda.memcpy_async(input_info["d_ptr"], input_info["h_ptr"], input_info["nbytes"], 1, self.stream)
        
        # 3. Enqueue execution
        if self.parent.trt_version < 10:
            self.context.execute_async_v2(self.bindings, self.stream)
        else:
            self.context.enqueue_v3(self.stream)
            
        # 4. Transfer outputs from device back to host pinned buffers
        for output_info in self.outputs.values():
            self.cuda.memcpy_async(output_info["h_ptr"], output_info["d_ptr"], output_info["nbytes"], 2, self.stream)
            
        # 5. Synchronize stream
        self.cuda.synchronize_stream(self.stream)
        
        # 6. Parse and decode detections (only return first batch item for camera stream logic)
        return self._postprocess(conf_threshold, iou_threshold)[0]

    def _postprocess(self, conf_threshold: float, iou_threshold: float) -> list:
        # Extract numpy arrays from host buffers
        output_tensors = {info["name"]: info["h_buf"] for info in self.outputs.values()}
        
        if self.model_type == "yolov5s":
            # YOLOv5 outputs a single tensor usually named 'output' or first output tensor
            output_name = list(output_tensors.keys())[0]
            output_data = output_tensors[output_name]
            shape = self.outputs[output_name]["shape"]
            return self._decode_yolov5(output_data, shape, conf_threshold, iou_threshold)
        elif self.model_type == "rtdetr":
            return self._decode_rtdetr(output_tensors, conf_threshold)
        else:
            raise ValueError(f"Unknown model type {self.model_type}")

    def _decode_yolov5(self, data: np.ndarray, shape: tuple, conf_threshold: float, iou_threshold: float) -> list:
        # data shape is typically (B, N, 5 + classes)
        # Flattened layout is passed, reshape just in case
        data = data.reshape(shape)
        B, N, C = data.shape
        batch_detections = []
        
        for b in range(B):
            frame_data = data[b]
            # obj_conf is column 4
            obj_conf = frame_data[:, 4]
            keep_indices = obj_conf > conf_threshold
            filtered = frame_data[keep_indices]
            
            if len(filtered) == 0:
                batch_detections.append([])
                continue
                
            cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            # Convert to [x1, y1, x2, y2]
            # Often coordinates are already at model scale or relative (0 to 1).
            # Let's check: if coords are all < 1.01, scale them by input_width and input_height.
            is_relative = np.max(cx) <= 1.01 if len(cx) > 0 else True
            x_mult = float(self.parent.input_width) if is_relative else 1.0
            y_mult = float(self.parent.input_height) if is_relative else 1.0
            
            x1 = (cx - w / 2.0) * x_mult
            y1 = (cy - h / 2.0) * y_mult
            x2 = (cx + w / 2.0) * x_mult
            y2 = (cy + h / 2.0) * y_mult
            
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            obj_conf = filtered[:, 4]
            class_probs = filtered[:, 5:]
            
            detections = []
            for class_id in range(class_probs.shape[1]):
                scores = obj_conf * class_probs[:, class_id]
                valid_mask = scores >= conf_threshold
                if not np.any(valid_mask):
                    continue
                    
                cls_boxes = boxes[valid_mask]
                cls_scores = scores[valid_mask]
                
                keep = nms(cls_boxes, cls_scores, iou_threshold)
                for idx in keep:
                    detections.append({
                        "class_id": class_id,
                        "confidence": float(cls_scores[idx]),
                        "box": [float(c) for c in cls_boxes[idx]]
                    })
            batch_detections.append(detections)
        return batch_detections

    def _decode_rtdetr(self, output_tensors: dict, conf_threshold: float) -> list:
        # RT-DETR has output tensors for boxes (B, Q, 4) and scores (B, Q, classes)
        boxes_tensor = None
        scores_tensor = None
        
        for name, arr in output_tensors.items():
            if len(arr.shape) == 3:
                if arr.shape[2] == 4:
                    boxes_tensor = arr
                elif arr.shape[2] > 4:
                    scores_tensor = arr
                    
        if boxes_tensor is None or scores_tensor is None:
            # Check if there is one combined tensor (e.g. B, Q, 4+classes)
            for name, arr in output_tensors.items():
                if len(arr.shape) == 3 and arr.shape[2] >= 6:
                    return self._decode_rtdetr_combined(arr, conf_threshold)
            return [[]]
            
        B, Q, C = scores_tensor.shape
        batch_detections = []
        
        for b in range(B):
            detections = []
            b_boxes = boxes_tensor[b]
            b_scores = scores_tensor[b]
            
            for q in range(Q):
                scores = b_scores[q]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence >= conf_threshold:
                    cx, cy, w, h = b_boxes[q]
                    # RT-DETR boxes are normalized [cx, cy, w, h]
                    x1 = (cx - w / 2.0) * float(self.parent.input_width)
                    y1 = (cy - h / 2.0) * float(self.parent.input_height)
                    x2 = (cx + w / 2.0) * float(self.parent.input_width)
                    y2 = (cy + h / 2.0) * float(self.parent.input_height)
                    detections.append({
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "box": [x1, y1, x2, y2]
                    })
            batch_detections.append(detections)
        return batch_detections

    def _decode_rtdetr_combined(self, data: np.ndarray, conf_threshold: float) -> list:
        B, Q, C = data.shape
        batch_detections = []
        for b in range(B):
            detections = []
            frame_data = data[b]
            for q in range(Q):
                row = frame_data[q]
                # Format: [cx, cy, w, h, class0_score, class1_score, ...] or [x1, y1, x2, y2, conf, class_id]
                if C == 6:
                    x1, y1, x2, y2, conf, class_id = row
                    if conf >= conf_threshold:
                        detections.append({
                            "class_id": int(class_id),
                            "confidence": float(conf),
                            "box": [float(x1), float(y1), float(x2), float(y2)]
                        })
                else:
                    cx, cy, w, h = row[:4]
                    scores = row[4:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence >= conf_threshold:
                        x1 = (cx - w / 2.0) * float(self.parent.input_width)
                        y1 = (cy - h / 2.0) * float(self.parent.input_height)
                        x2 = (cx + w / 2.0) * float(self.parent.input_width)
                        y2 = (cy + h / 2.0) * float(self.parent.input_height)
                        detections.append({
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "box": [x1, y1, x2, y2]
                        })
            batch_detections.append(detections)
        return batch_detections

    def __del__(self):
        # Free CUDA allocations
        try:
            for name, info in self.inputs.items():
                self.cuda.free(info["d_ptr"])
                self.cuda.host_free(info["h_ptr"])
            for name, info in self.outputs.items():
                self.cuda.free(info["d_ptr"])
                self.cuda.host_free(info["h_ptr"])
            self.cuda.destroy_stream(self.stream)
        except Exception:
            pass



