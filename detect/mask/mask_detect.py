import sys
from pathlib import Path

FILE = Path(__file__).resolve()

ROOT = FILE.parents[2]  # YOLOv5 root directory

ROOT_YOLACT = Path.joinpath(ROOT, 'yolact')

if str(ROOT) not in sys.path:

    sys.path.append(str(ROOT))  # add ROOT to PATH

if str(ROOT_YOLACT) not in sys.path:

    sys.path.append(str(ROOT_YOLACT))  # add ROOT to PATH

from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import cv2

class MaskDetectModel:

    def __init__(self, mask_detect_config : dict) -> None:

        self._net = None
        self._config = mask_detect_config

        self._load_yolact_model()

        self._detect_mask = None

    def _load_yolact_model(self, model_config_name : str ="yolact_base_config" ) -> None:

        set_cfg(model_config_name)

        with torch.no_grad():

            if self._config['cuda']:

                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')

            else:

                torch.set_default_tensor_type('torch.FloatTensor')

            self._dataset = None

        print('Loading model_weight...', end='')
        self._net = Yolact()
        self._net.load_weights(self._config['model_weight_path'])
        self._net.eval()
        print(' Done.')

        if self._config['cuda']:

            self._net = self._net.cuda()

        self._net.detect.use_fast_nms = self._config['fast_nms']
        self._net.detect.use_cross_class_nms = self._config['cross_class_nms']
        cfg.mask_proto_debug = self._config['mask_proto_debug']

        print("\nMask Detect Model Initial Done.\n")


    def change_weight(self, new_weight_path : str) -> None:

        self._config['model_weight_path'] = new_weight_path

        self._load_yolact_model()


    def detect(self, frame : np.ndarray) -> np.ndarray:

        yoloframe = frame.copy()

        frame = torch.from_numpy(yoloframe).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self._net(batch)

        # ----執行 yolact 辨識----
        h, w, _ = frame.shape
        classes, scores, boxes, masks = postprocess( preds, w, h,
                                                     visualize_lincomb = self._config['display_lincomb'],
                                                     crop_masks = self._config['crop'],
                                                     score_threshold = float(self._config['score_threshold']) )

        # ----輸出 mask 圖片----
        msk = masks[0, :, :, None]
        mask = msk.view(1, msk.shape[0], msk.shape[1], 1)
        img_gpu_masked = (mask.sum(dim=0) >= 1).float().expand(-1, -1, 3)  # img_gpu *
        img_numpy = (img_gpu_masked * 255).byte().cpu().numpy()
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2GRAY)
        cnts, _ = cv2.findContours(img_numpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:

            out = np.zeros(img_numpy.shape, np.uint8)
            self._detect_mask = cv2.bitwise_and(img_numpy, out)

        else:

            cnt = max(cnts, key=cv2.contourArea)
            out = np.zeros(img_numpy.shape, np.uint8)
            cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
            self._detect_mask = cv2.bitwise_and(img_numpy, out)

        return self._detect_mask


    def get_vertex(self, detect_frame : np.ndarray = None) -> list:

        if self._detect_mask is None and detect_frame is None :

            return []

        if detect_frame is None:

            detect_frame = self._detect_mask

        contours_com, _ = cv2.findContours(detect_frame, 3, 2)
        contours_com_tmp = contours_com[0]
        minAreaRect_com = cv2.minAreaRect(contours_com_tmp)
        boxs_com = cv2.boxPoints(minAreaRect_com)

        vertex = [(int(boxs_com[0][0]), int(boxs_com[0][1])), (int(boxs_com[1][0]), int(boxs_com[1][1])),
                  (int(boxs_com[2][0]), int(boxs_com[2][1])), (int(boxs_com[3][0]), int(boxs_com[3][1]))]
        
        return vertex

    def get_mask_length(self, vertex : list = None) -> (list, float):

        if vertex is None:

            vertex = self.get_vertex()

        four_length = [((vertex[0][0] - vertex[1][0]) ** 2 + (vertex[0][1] - vertex[1][1]) ** 2) ** 0.5,
                       ((vertex[1][0] - vertex[2][0]) ** 2 + (vertex[1][1] - vertex[2][1]) ** 2) ** 0.5,
                       ((vertex[2][0] - vertex[3][0]) ** 2 + (vertex[2][1] - vertex[3][1]) ** 2) ** 0.5,
                       ((vertex[3][0] - vertex[0][0]) ** 2 + (vertex[3][1] - vertex[0][1]) ** 2) ** 0.5]

        return four_length, sum(four_length)

    def get_stand_mask(self, detect_frame : np.ndarray = None):

        if self._detect_mask is None and detect_frame is None :

            return []

        if detect_frame is None:

            detect_frame = self._detect_mask

        contours_com, _ = cv2.findContours(detect_frame, 3, 2)
        contours_com_tmp = contours_com[0]
        minAreaRect_com = cv2.minAreaRect(contours_com_tmp)
        boxs_com = cv2.boxPoints(minAreaRect_com)

        mask = np.zeros((detect_frame.shape[0], detect_frame.shape[1]), np.uint8)  # 16:9
        mask = cv2.fillPoly(mask, np.array([boxs_com], dtype=np.int32), 255)

        return mask


if str(ROOT) in sys.path:

    sys.path.remove(str(ROOT))  # remove ROOT to PATH

if str(ROOT_YOLACT) in sys.path:

    sys.path.remove(str(ROOT_YOLACT))  # remove ROOT to PATH