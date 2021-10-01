import numpy as np
import pickle
import torch
# import timm
import cv2
import time

# pytorch related imports
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from src.dataset import ICPDataset

# import pytorch_lightning as pl
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from time import time
from pathlib import Path
# import caffe2.python.onnx.backend as backend

import onnx

from onnx import ModelProto
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import onnxruntime as rt

from utils_cv.detection.model import DetectionLearner, get_pretrained_keypointrcnn
from utils_cv.detection.dataset import DetectionDataset
from classes import person_keypoint_meta
from my_utils import get_all_files_in_folder


def convert_model_to_onnx(onnx_path, engine_name, model, device, batch_size):
    input_shape = (batch_size, 3, 256, 256)
    inputs = torch.ones(*input_shape)
    inputs = inputs.to('cpu')

    torch.onnx.export(model, inputs, onnx_path, opset_version=11, export_params=True,
                      do_constant_folding=True,
                      input_names=None, output_names=None,
                      dynamic_axes=None)  # ['box_predictor', 'keypoint_predictor']


def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    batch_size = 1

    model_test = get_pretrained_keypointrcnn(
        num_classes=2,  # __background__ and milk_bottle
        num_keypoints=len(person_keypoint_meta["labels"]),
    )

    checkpoint = torch.load('logs/checkpoint_exp_5_e0_precision_0.334_recall_0.435.pth', map_location="cpu")
    model_test.load_state_dict(checkpoint['model'])
    model_test.to('cpu')
    model_test.eval()

    # print(next(model_test.parameters()).is_cuda)
    # print(model_test)

    onnx_path = "onnx/keypointrcnn_resnet50_fpn.onnx"
    engine_name = "onnx/keypointrcnn_resnet50_fpn.plan"
    # convert_model_to_onnx(
    #     onnx_path=onnx_path, engine_name=engine_name, model=model_test, device=device, batch_size=batch_size)

    # one_img = torch.randn(1, 3, 256, 256)

    format = 'opencv_dnn'

    if format == 'onnxruntime':
        print('rt.get_device()', rt.get_device())

        test_images = get_all_files_in_folder(Path('data/test/images'), ['*.jpg'])

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(onnx_path)

        detection_time = 0
        for im in tqdm(test_images):
            img = cv2.imread(str(im), cv2.IMREAD_COLOR)
            one_img = img.copy()
            one_img = np.moveaxis(one_img, -1, 0)
            one_img = torch.tensor(one_img[np.newaxis, ...].astype(np.float32))

            start_time = time.time()
            onnx_results = sess.run(None,
                                    {net_feed_input[0]: one_img.detach().numpy()})
            detection_time += time.time() - start_time

            # cv2.imshow('prediction', img)
            # cv2.waitKey()

        one_frame = detection_time / len(test_images)
        print('1 frame prediction time, ms', round(one_frame * 1000, 1))
        print('FPS:', round(1 / one_frame, 2))

    elif format == 'opencv_dnn':
        net = cv2.dnn.readNetFromONNX(onnx_path)
        print("[INFO] Model is loaded")


if __name__ == '__main__':
    main()
