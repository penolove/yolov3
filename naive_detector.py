import argparse

import torch
import numpy as np
import arrow
from bistiming import Stopwatch
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.config import BoundedBoxObject
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import (ImageHandler, Image, resize_and_stack_image_objs)

from utils.utils import (
    torch_utils,
    load_classes,
    non_max_suppression,
    scale_coords
)
from utils.datasets import letterbox
from utils.parse_config import parse_data_cfg
from models import Darknet, load_darknet_weights, save_weights


# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
parser.add_argument(
    '--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
parser.add_argument(
    '--weight_file', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
parser.add_argument(
    '--conf-thres', type=float, default=0.08, help='object confidence threshold')
parser.add_argument(
    '--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
parser.add_argument(
    '--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument(
    '--store_weight', action='store_true', help='stores the model to darknet weight',
    default=False)


class YoloV3DetectorWrapper(ObjectDetector):
    def __init__(self, cfg, img_size, weight_file, data_cfg, conf_thres=0.5, nms_thres=0.5,
                 resize_with_padding=True, use_fuse=True):
        self.device = torch_utils.select_device()
        self.max_img_side = img_size
        self.model = Darknet(cfg, img_size)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.resize_with_padding = resize_with_padding

        # Load weights
        if weight_file.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weight_file, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, weight_file)

        # Fuse Conv2d + BatchNorm2d layers
        if use_fuse:
            self.model.fuse()

        # set Eval mode
        self.model.to(self.device).eval()

        self.classes = load_classes(parse_data_cfg(data_cfg)['names'])

    def get_max_image_side(self, image_size):
        max_side = max(image_size)
        ratio = self.max_img_side / max_side

        def find_closet_32_base(x):
            return max(round(x / 32.0), 1) * 32

        return tuple(find_closet_32_base(i * ratio) for i in image_size)

    def detect(self, image_obj) -> DetectionResult:
        ori_w, ori_h = image_obj.pil_image_obj.size
        detected_objects = []

        target_size = self.get_max_image_side((ori_w, ori_h))  # resize w, h
        if self.resize_with_padding:
            img_array, _, _, _ = letterbox(
                np.array(image_obj.pil_image_obj), new_shape=self.max_img_side)
            img = img_array.transpose(2, 0, 1).astype(np.float32)  # BGR to RGB
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        else:
            # not padding will slightly faster
            # N, H, W, C
            processed_image = resize_and_stack_image_objs(target_size, [image_obj.pil_image_obj])
            # N, C(RGB), H, W
            img_array = processed_image.transpose(0, 3, 1, 2).astype(np.float32)
            img_array /= 255.0
            img = torch.from_numpy(img_array).cuda().to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]
        if det is not None and len(det) > 0:
            ori_size = (ori_h, ori_w)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ori_size).round()
            for x1, y1, x2, y2, score, _, label_idx in det:
                label = self.classes[int(label_idx)]
                # make sure not directly reference to the cuda tensor
                detected_objects.append(BoundedBoxObject(
                    int(x1), int(y1), int(x2), int(y2), label, float(score), ''))
        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    @property
    def valid_labels(self):
        return set(self.classes)


if __name__ == '__main__':
    opt = parser.parse_args()
    use_fuse = not opt.store_weight
    object_detector = YoloV3DetectorWrapper(opt.cfg, opt.img_size, opt.weight_file, opt.data_cfg,
                                            conf_thres=opt.conf_thres, use_fuse=use_fuse)
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    with Stopwatch('Running inference on image {}...'.format(raw_image_path)):
        detection_result = object_detector.detect(image_obj)
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")

    # stores models to darkent weight
    if opt.store_weight:
        save_weights(object_detector.model, path='converted_darknet.weights')
