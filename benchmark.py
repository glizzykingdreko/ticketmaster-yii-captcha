import numpy as np
import onnxruntime

from sys import exit, argv
from glob import glob
from os import path
from random import Random
from time import time
from PIL import Image

from dataset import RAW_DIR, prepare, to_tensor
from generator import CHARSET, CODE_LENGTH, generate_code, render

# Must match RELEASE in trainer.py
RELEASE = argv[1] if len(argv) > 1 else "nano"
ONNX_PATH = path.join(path.dirname(path.abspath(__file__)), "output", f"{RELEASE}.onnx")
SAMPLES = 20000
# Fixed so runs compare like for like
SEED = 20260916
GATE = 0.999


def to_input(image):
    return to_tensor(prepare(image)).numpy()


def run_model(session, inputs, chunk_size=1000):
    batch = np.stack(inputs)
    logits = [
        session.run(None, {"input1": batch[start:start + chunk_size]})[0]
        for start in range(0, len(batch), chunk_size)
    ]
    return np.concatenate(logits).reshape(len(batch), CODE_LENGTH, len(CHARSET))


def confidence(logits):
    # Softmax per slot, then the weakest slot decides the whole read
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
    return probabilities.max(axis=2).min(axis=1)


if __name__ == "__main__":
    # [1] Load the exported reader
    try:
        session = onnxruntime.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except:
        print(f"Path {ONNX_PATH} not found.\nPlease trian the model first using trainer.py\nor download it from https://github.com/glizzykingdreko/ticketmaster-yii-captcha-solver/releases")
        exit(1)
    
    print(f"[1] Loaded {ONNX_PATH}")
    print(f"    size={path.getsize(ONNX_PATH) / 1e6:.3f}MB\n")

    # [2] Score on fresh synthetic
    print(f"[2] Scoring {SAMPLES} synthetic captchas")
    random = Random(SEED)
    inputs, targets = [], []
    for _ in range(SAMPLES):
        code = generate_code(random)
        inputs.append(to_input(render(code, random)))
        targets.append([CHARSET.index(letter) for letter in code])
    logits = run_model(session, inputs)
    matches = logits.argmax(axis=2) == np.array(targets)
    correct = matches.all(axis=1)
    scores = confidence(logits)
    print(f"    full={correct.mean() * 100:.3f}%  per_letter={matches.mean() * 100:.4f}%  misses={(~correct).sum()}\n")

    # [3] Testing the confidence gate
    print("[3] Testing the confidence gate")
    for threshold in (0.0, 0.9, 0.99, GATE, 0.9999):
        kept = scores >= threshold
        if not kept.any():
            continue
        print(f"    >= {threshold:<8} accepted {kept.mean() * 100:5.1f}%  accuracy {correct[kept].mean() * 100:8.4f}%  misses {(~correct[kept]).sum():3}  pulls per solve {1 / kept.mean():.2f}")
    print()

    # [4] Measure a single image
    print("[4] Measuring single image inference")
    single = {"input1": inputs[0][None]}
    session.run(None, single)
    started = time()
    for _ in range(300):
        session.run(None, single)
    each = (time() - started) / 300
    print(f"    {each * 1000:.2f}ms per image, {1 / each:.0f}/s single threaded\n")

    # [5] Real captchas have no labels
    real_paths = sorted(glob(path.join(RAW_DIR, "*.png")))
    if not real_paths:
        print(f"[5] Skipping real captchas, none found in {RAW_DIR}")
        raise SystemExit(0)
    sample = Random(0).sample(real_paths, min(5000, len(real_paths)))
    print(f"[5] Scoring {len(sample)} real captchas")
    real_scores = confidence(run_model(session, [to_input(Image.open(image_path)) for image_path in sample]))
    passing = (real_scores >= GATE).mean()
    pulls = 1 / passing if passing else float("inf")
    print(f"[=] median confidence {np.median(real_scores):.4f}, {passing * 100:.1f}% clear the {GATE} gate, {pulls:.2f} pulls per solve")
