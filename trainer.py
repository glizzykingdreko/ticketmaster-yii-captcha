
import torch

from sys import argv, exit
from os import makedirs, path
from time import time
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from dataset import LabelledCaptchas, SyntheticCaptchas
from generator import CHARSET, CODE_LENGTH
from model import INPUT_HEIGHT, INPUT_WIDTH, CaptchaNet, RELEASES, pick_device

# Which reader to build, one of pico, nano, micro
RELEASE = argv[1] if len(argv) > 1 else "nano"
STEPS = 4000
BATCH_SIZE = 128
WORKERS = 6
LOG_EVERY = 250
# Mix in the labelled real images from dataset/labels.json
USE_REAL = False
# Start from the saved weights when they exist
RESUME = False

OUTPUT_DIR = path.join(path.dirname(path.abspath(__file__)), "output")
OUT_PATH = path.join(OUTPUT_DIR, f"{RELEASE}.pt")
ONNX_PATH = path.join(OUTPUT_DIR, f"{RELEASE}.onnx")


def evaluate(model, images, targets, device, batch_size=256):
    model.eval()
    whole, letters = 0, 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            prediction = model(images[start:start + batch_size].to(device)).argmax(dim=2).cpu()
            matches = prediction == targets[start:start + batch_size]
            letters += matches.sum().item()
            whole += matches.all(dim=1).sum().item()
    model.train()
    return whole / len(images) * 100, letters / (len(images) * CODE_LENGTH) * 100


if __name__ == "__main__":
    try:
        channels = RELEASES[RELEASE]
    except:
        print(f"{RELEASE=} not found")
        exit(0)
    
    device = pick_device()

    # [1] Build the datasets
    print(f"[1] Building {RELEASE} at {channels} channels on {device}")
    train_set = SyntheticCaptchas(STEPS * BATCH_SIZE, seed=1, train=True)
    if USE_REAL:
        real_set = LabelledCaptchas(train=True)
        repeats = max(len(train_set) // max(len(real_set), 1) // 8, 1)
        train_set = ConcatDataset([train_set] + [real_set] * repeats)
        print(f"    real={len(real_set)} repeats={repeats}")
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=USE_REAL,
        num_workers=WORKERS,
        persistent_workers=WORKERS > 0,
    )

    # Held in memory
    valid_set = SyntheticCaptchas(3000, seed=99, train=False)
    valid_samples = [valid_set[index] for index in range(len(valid_set))]
    valid_images = torch.stack([image for image, _ in valid_samples])
    valid_targets = torch.stack([target for _, target in valid_samples])
    print(f"    validation={len(valid_images)}\n")

    # [2] Build the model
    model = CaptchaNet(channels).to(device)
    if RESUME and path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH, map_location=device)["state"])
        print(f"[2] Resuming from {OUT_PATH}")
    else:
        print("[2] Starting from scratch")
    print(f"    parameters={sum(parameter.numel() for parameter in model.parameters())}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=STEPS)
    loss_function = nn.CrossEntropyLoss()

    # [3] Train
    print(f"[3] Training for {STEPS} steps")
    model.train()
    started = time()
    running_loss = 0.0
    logged_steps = 0
    for step, (images, targets) in enumerate(train_loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        loss = loss_function(model(images).reshape(-1, len(CHARSET)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        running_loss += loss.item()
        logged_steps += 1
        if step % LOG_EVERY == 0 or step == STEPS:
            full, per_letter = evaluate(model, valid_images, valid_targets, device)
            print(f"    step {step:5}/{STEPS}  loss {running_loss / logged_steps:.4f}  full {full:6.2f}%  per letter {per_letter:6.2f}%  {time() - started:4.0f}s")
            running_loss = 0.0
            logged_steps = 0
        if step >= STEPS:
            break

    # [4] Save the weights
    makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "charset": CHARSET,
            "length": CODE_LENGTH,
            "channels": channels,
            "release": RELEASE,
        },
        OUT_PATH,
    )
    print(f"\n[4] Saved {OUT_PATH}")

    # [5] Export to ONNX for benchmark.py and the solver, the input name is what they feed
    model.cpu().eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 1, INPUT_HEIGHT, INPUT_WIDTH),
        ONNX_PATH,
        input_names=["input1"],
        dynamic_axes={
            "input1": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"[=] Exported {ONNX_PATH}")
