import numpy as np
import torch

from json import load
from os import path
from random import Random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from generator import CHARSET, generate_code, render
from model import INPUT_HEIGHT, INPUT_WIDTH

DATASET_DIR = path.join(path.dirname(path.abspath(__file__)), "dataset")
RAW_DIR = path.join(DATASET_DIR, "raw")
LABELS_PATH = path.join(DATASET_DIR, "labels.json")


def prepare(image):
    return image.convert("L").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)


def augment(image, random):
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.9)))
    array = np.asarray(image, dtype=np.float32)
    if random.random() < 0.7:
        array = array * random.uniform(0.85, 1.15) + random.uniform(-18, 18)
    if random.random() < 0.5:
        noise = np.random.default_rng(random.getrandbits(32))
        array = array + noise.normal(0, random.uniform(1, 7), array.shape)
    return np.clip(array, 0, 255)


def to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) / 0.5).unsqueeze(0)


def encode(code):
    return torch.tensor([CHARSET.index(letter) for letter in code], dtype=torch.long)


class SyntheticCaptchas(Dataset):
    def __init__(self, length, seed=0, train=True):
        self.length = length
        self.seed = seed
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        random = Random(self.seed * 1000003 + index)
        code = generate_code(random)
        image = prepare(render(code, random))
        if self.train:
            image = augment(image, random)
        return to_tensor(image), encode(code)


class LabelledCaptchas(Dataset):
    def __init__(self, train=False, minimum_confidence=0.0):
        with open(LABELS_PATH) as handle:
            entries = load(handle)
        self.train = train
        self.items = [
            (name, entry["label"])
            for name, entry in entries.items()
            if entry.get("label") and entry.get("confidence", 1.0) >= minimum_confidence
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        name, code = self.items[index]
        image = prepare(Image.open(path.join(RAW_DIR, name)))
        if self.train:
            image = augment(image, Random(index))
        return to_tensor(image), encode(code)
