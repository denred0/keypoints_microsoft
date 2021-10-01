import cv2
import torch
import numpy as np
import shutil
import time

from pathlib import Path
from tqdm import tqdm

from utils_cv.detection.dataset import DetectionDataset
from utils_cv.detection.model import DetectionLearner, get_pretrained_keypointrcnn
from utils_cv.detection.plot import plot_detections, PlotSettings

from my_utils import get_all_files_in_folder
from classes import person_keypoint_meta

DATA_PATH = 'data/train'
THRESHOLD = 0.5

data = DetectionDataset(
    root=DATA_PATH,
    keypoint_meta=person_keypoint_meta,
    train_pct=0.8,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

model = get_pretrained_keypointrcnn(
    num_classes=2,  # __background__ and milk_bottle
    num_keypoints=len(person_keypoint_meta["labels"]),
)
checkpoint = torch.load('logs/checkpoint_exp_4_e21_precision_0.83_recall_0.883.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

detector = DetectionLearner(dataset=data, model=model, device=device)

test_images = get_all_files_in_folder(Path('data/test/images'), ['*.jpg'])

dirpath_images = Path('data/test/images_result')
if dirpath_images.exists() and dirpath_images.is_dir():
    shutil.rmtree(dirpath_images)
Path(dirpath_images).mkdir(parents=True, exist_ok=True)

detection_time = 0

for im in tqdm(test_images):
    start_time = time.time()
    detections = detector.predict(str(im), threshold=THRESHOLD)
    detection_time += time.time() - start_time
    image = plot_detections(detections, keypoint_meta=person_keypoint_meta)
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(Path.joinpath(dirpath_images, im.name)), img)

    # cv2.imshow('prediction', img)
    # cv2.waitKey()

one_frame = detection_time / len(test_images)
print('1 frame prediction time, ms', round(one_frame * 1000, 1))
print('FPS:', round(1 / one_frame, 2))
