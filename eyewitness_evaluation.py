import argparse
import logging

from eyewitness.dataset_util import BboxDataSet
from eyewitness.evaluation import BboxMAPEvaluator

from naive_detector import YoloV3DetectorWrapper


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
    '--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument(
    '--conf-thres', type=float, default=0.1, help='object confidence threshold')
parser.add_argument(
    '--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
parser.add_argument(
    '--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument(
    '--img-size', type=int, default=416, help='inference size (pixels)')


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    dataset_folder = 'data/VOCdevkit/VOC2007'
    dataset_VOC_2007 = BboxDataSet(dataset_folder, 'VOC2007')
    opt = parser.parse_args()
    object_detector = YoloV3DetectorWrapper(
        opt.cfg, opt.img_size, opt.weight_file, opt.data_cfg, conf_thres=opt.conf_thres)
    bbox_map_evaluator = BboxMAPEvaluator()
    # which will lead to ~0.79
    print(bbox_map_evaluator.evaluate(object_detector, dataset_VOC_2007)['mAP'])
