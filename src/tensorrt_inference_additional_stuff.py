# polygraphy surgeon sanitize onnx/keypointrcnn_resnet50_fpn_simple.onnx --fold-constants --output onnx/keypointrcnn_resnet50_fpn_simple_folded.onnx
# python -m onnxsim onnx/keypointrcnn_resnet50_fpn.onnx onnx/keypointrcnn_resnet50_fpn_simple.onnx

import numpy as np
import pickle
import torch
# import timm
import cv2
import time

# pytorch related imports
from torch.utils.data import DataLoader
from torch import nn

from torch2trt import torch2trt

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
    input_shape = (1, 3, 256, 256)
    inputs = torch.ones(*input_shape)
    inputs = inputs.to('cpu')

    torch.onnx.export(model, inputs, onnx_path,
                      opset_version=11,  # 11,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=None,
                      output_names=['output'],
                      dynamic_axes=None)  # ['box_predictor', 'keypoint_predictor']


# def build_engine(TRT_LOGGER, onnx_path, shape, MAX_BATCH=100):
#     with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network(
#             1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#         # builder.fp16_mode = True
#         # shape = [3, 256, 256]
#         builder.max_workspace_size = 1 << 30  # 1GB
#         builder.max_batch_size = MAX_BATCH
#         profile = builder.create_optimization_profile()
#         config.max_workspace_size = (3072 << 20)
#         config.add_optimization_profile(profile)
#         with open(onnx_path, 'rb') as model:
#             parser.parse(model.read())
#
#             for index in range(parser.num_errors):
#                 print(parser.get_error(index))
#
#         network.get_input(0).shape = shape
#         engine = builder.build_cuda_engine(network)
#         return engine


# def save_engine(engine, file_name):
#     buf = engine.serialize()
#     with open(file_name, 'wb') as f:
#         f.write(buf)


def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    batch_size = 1

    model_test = get_pretrained_keypointrcnn(
        num_classes=2,
        num_keypoints=len(person_keypoint_meta["labels"]),
    )

    checkpoint = torch.load('logs/checkpoint_exp_8_e0_precision_0.159_recall_0.313.pth', map_location="cpu")
    model_test.load_state_dict(checkpoint['model'])
    model_test.to('cpu')
    model_test.eval()


    # model_test.cuda()
    # x = torch.ones((1, 3, 256, 256)).cuda()
    #
    # # convert to TensorRT feeding sample data as input
    # model_trt = torch2trt(model_test, [x])

    #
    # BATCH_SIZE = 1
    #
    # dummy_input = torch.randn(1, 3, 256, 256)
    #
    # torch.onnx.export(model_test,
    #                   dummy_input,
    #                   "onnx/resnet50_pytorch.onnx",
    #                   opset_version=11,
    #                   export_params=True,
    #                   keep_initializers_as_inputs=True,
    #                   verbose=True)

    # last_layer = model_test.get_layer(model_test.num_layers - 1)
    # model_test.mark_output(last_layer.get_output(0))

    # print(next(model_test.parameters()).is_cuda)
    # print(model_test)

    onnx_path = "onnx/keypointrcnn_resnet50_fpn.onnx"
    engine_name = "onnx/keypointrcnn_resnet50_fpn.plan"
    convert_model_to_onnx(
        onnx_path=onnx_path, engine_name=engine_name, model=model_test, device=device, batch_size=batch_size)


    format = 'onnxruddntime'

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
        for i, im in enumerate(test_images):
            img = cv2.imread(str(im), cv2.IMREAD_COLOR)
            one_img = img.copy()
            one_img = np.moveaxis(one_img, -1, 0)
            one_img = torch.tensor(one_img[np.newaxis, ...].astype(np.float32))

            print(f'frame {i}, detection_time {detection_time}')
            start_time = time.time()
            onnx_results = sess.run(None,
                                    {net_feed_input[0]: one_img.detach().numpy()})
            if i != 0:
                detection_time += time.time() - start_time

            # cv2.imshow('prediction', img)
            # cv2.waitKey()

        one_frame = detection_time / (len(test_images) - 1)
        print('1 frame prediction time, ms', round(one_frame * 1000, 1))
        print('FPS:', round(1 / one_frame, 2))

    elif format == 'opencv_dnn':
        net = cv2.dnn.readNetFromONNX(onnx_path)
        print("[INFO] Model is loaded")
    elif format == 'other':
        # TensorRT flow

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        verbose = True
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if verbose else trt.Logger()
        trt_runtime = trt.Runtime(TRT_LOGGER)

        model = ModelProto()
        with open(onnx_path, "rb") as f:
            model.ParseFromString(f.read())

        d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
        d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
        shape = [1, d0, d1, d2]

        print('shape', shape)

        builder = trt.Builder(TRT_LOGGER)
        # print('builder', builder)

        ONNX_build_engine(onnx_path, '1.trt')

        # engine = build_engine(TRT_LOGGER=TRT_LOGGER, onnx_path=onnx_path, shape=shape)
        # save_engine(engine, engine_name)


def ONNX_build_engine(onnx_file_path, engine_file_path):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network,
                                                                                                             G_LOGGER) as parser:
        builder.max_batch_size = 16
        builder.max_workspace_size = 1 << 20

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())

            for index in range(parser.num_errors):
                print(parser.get_error(index))

        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        # with open(engine_file_path, "wb") as f:
        #     f.write(engine.serialize())
        return engine

    # explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # network = builder.create_network(explicit_batch)


if __name__ == '__main__':
    main()
