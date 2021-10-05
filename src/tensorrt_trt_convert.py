import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt

engine_name = "hrnet_w32_coco_384x288.plan"
onnx_path = "onnx/keypointrcnn_resnet50_fpn_simple_folded.onnx"
batch_size = 1

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
shape = [batch_size, d0, d1, d2]
print(shape)
engine = eng.build_engine(onnx_path, shape=shape)
# eng.save_engine(engine, engine_name)
