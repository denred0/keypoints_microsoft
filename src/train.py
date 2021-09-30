import torch

from utils_cv.detection.dataset import DetectionDataset
from utils_cv.detection.model import DetectionLearner, get_pretrained_keypointrcnn

from classes import person_keypoint_meta

DATA_PATH = 'data/train'
EPOCHS = 30
LEARNING_RATE = 1e-3

data = DetectionDataset(
    root=DATA_PATH,
    keypoint_meta=person_keypoint_meta,
    train_pct=0.8,
)
print(f"There are {len(data)} images, of which {len(data.train_ds)} are for training.")
# data.show_ims(rows=3, seed=54)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using torch device: {device}")

model = get_pretrained_keypointrcnn(
    num_classes=2,  # __background__ and milk_bottle
    num_keypoints=len(person_keypoint_meta["labels"]),
)

detector = DetectionLearner(dataset=data, model=model, device=device)
detector.fit(epochs=EPOCHS, lr=LEARNING_RATE, print_freq=100, skip_evaluation=False)
