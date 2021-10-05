# polygraphy surgeon sanitize onnx/keypointrcnn_resnet50_fpn_simple.onnx --fold-constants --output onnx/keypointrcnn_resnet50_fpn_simple_folded.onnx
# python -m onnxsim onnx/keypointrcnn_resnet50_fpn.onnx onnx/keypointrcnn_resnet50_fpn_simple.onnx

import numpy as np
import torch
import cv2
import time

from pathlib import Path

import onnx

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

    onnx_path = "onnx/keypointrcnn_resnet50_fpn.onnx"
    engine_name = "onnx/keypointrcnn_resnet50_fpn.plan"
    convert_model_to_onnx(
        onnx_path=onnx_path, engine_name=engine_name, model=model_test, device=device, batch_size=batch_size)

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


if __name__ == '__main__':
    main()
