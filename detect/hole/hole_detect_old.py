import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode, time_sync



class HoleDetectModel:

    def __init__(self, hole_detect_config : dict) -> None:

        self._model = None
        self._config = hole_detect_config

        self._load_model()
        self._detect_img = None

    def _load_model(self) -> None:

        self._device = select_device('')
        self._model = DetectMultiBackend(weights=self._config['model_weight_path'],
                                         device=self._device, dnn=False,
                                         data=self._config['model_data_path'])

        self._stride = self._model.stride
        self._name = self._model.names
        self._pt = self._model.pt
        self._jit = self._model.jit
        self._onnx = self._model.onnx
        self._engine = self._model.engine

        self._imgsz = (416, 416)
        self._imgsz = check_img_size(self._imgsz, s=self._stride)  # check image size
        self._half = False
        self._half &= (self._pt or self._jit or self._onnx or self._engine) and self._device.type != 'cpu'  # FP16 supported on limited backends with CUDA

        if self._pt or self._jit:

            self._model.model.half() if self._half else self._model.model.float()

        print("\nHole Detect Model Initial Done.\n")

    def detect(self, frame : np.ndarray, focus_center_shape : tuple = None) -> (list, np.ndarray):

        self._detect_img = frame.copy()

        frame = self._detect_img

        # Focus center
        cut_start_x, cut_start_y = 0, 0

        if focus_center_shape is not None:

            focus_w = focus_center_shape[0]
            focus_h = focus_center_shape[1]

            frame_x = frame.shape[0]
            frame_y = frame.shape[0]

            cut_start_x = int(frame_x / 2) - int(focus_w / 2)
            cut_start_y = int(frame_y / 2) - int(focus_h / 2)

            frame = frame[cut_start_x : cut_start_x + focus_w, cut_start_y : cut_start_y + focus_h]

        # Padded resize
        letter_frame = letterbox(frame, 640, stride=32, auto=True)[0]

        # Convert
        convert_frame = letter_frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        convert_frame = np.ascontiguousarray(convert_frame)

        # Predict before process
        self._model.warmup(imgsz=(1, 3, *self._imgsz))  # warmup

        process_frame = convert_frame

        process_frame = torch.from_numpy(process_frame).to(self._device)
        process_frame = process_frame.half() if self._half else process_frame.float()  # uint8 to fp16/32
        process_frame /= 255  # 0 - 255 to 0.0 - 1.0

        if len(process_frame.shape) == 3:

            process_frame = process_frame[None]  # expand for batch dim

        # Predict -> NMS
        pred = self._model(process_frame, augment=False, visualize=False)
        # pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.1, classes=None, agnostic=False, max_det=1000)

        # Get predict result
        hole_detect = []

        det, detect_frame = None, frame

        # Process predictions
        for i, det in enumerate(pred):  # per image

            gn = torch.tensor(detect_frame.shape)[[1, 0, 1, 0]]
            annotator = Annotator(detect_frame, line_width=3, example=str(self._name))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(process_frame.shape[2:], det[:, :4], detect_frame.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    xyxy[0] = xyxy[0] + cut_start_y
                    xyxy[2] = xyxy[2] + cut_start_y
                    xyxy[1] = xyxy[1] + cut_start_x
                    xyxy[3] = xyxy[3] + cut_start_x

                    middle = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))

                    xyxy[0] = int(xyxy[0])
                    xyxy[1] = int(xyxy[1])
                    xyxy[2] = int(xyxy[2])
                    xyxy[3] = int(xyxy[3])

                    c = int(cls)  # integer class


                    print(c)
                    if c == 1:

                        hole_detect.append([[(min(xyxy[0], xyxy[2]), min(xyxy[1], xyxy[3])), (max(xyxy[0], xyxy[2]), max(xyxy[1], xyxy[3]))], middle])

                    # Draw detect object coordinate
                    c = int(cls)  # integer class
                    # label = (f'{self.names[c]} {conf:.2f}')
                    label = str(c)
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            detect_frame = self._draw_hole_center(hole_detect, annotator.result())

        return hole_detect, detect_frame

    def _draw_hole_center(self, hole_detect : list, detect_frame : np.ndarray) -> np.ndarray:

        for _, middle in hole_detect:

            cv2.circle(detect_frame, (middle[0], middle[1]), 3, (0, 0, 255), 4)

        return detect_frame
