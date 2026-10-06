from base64 import b64decode, b64encode
from io import BytesIO
from os import path
from random import Random
from time import perf_counter

import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont

from generator import CHARSET, CODE_LENGTH, HEIGHT, WIDTH, generate_code, render
from model import DEFAULT_RELEASE

ONNX_PATH = path.join(path.dirname(path.abspath(__file__)), "output", f"{DEFAULT_RELEASE}.onnx")
SCALE = 3
PANEL_WIDTH = 360


def solve_captcha(session, image):
    # image is a file path or a base64 string, with or without the data:image/...;base64, prefix
    if path.exists(image):
        captcha = Image.open(image)
    else:
        captcha = Image.open(BytesIO(b64decode(image.split(",")[-1])))
    # Grayscale, resized to the 96x80 model input, pixels scaled to [-1, 1]
    resized = captcha.convert("L").resize((96, 80), Image.BILINEAR)
    pixels = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
    logits = session.run(None, {"input1": pixels[None, None]})[0]
    logits = logits.reshape(CODE_LENGTH, len(CHARSET))
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = (exponent / exponent.sum(axis=-1, keepdims=True)).max(axis=-1)
    answer = "".join(CHARSET[index] for index in logits.argmax(axis=-1))
    return answer, probabilities


if __name__ == "__main__":
    # [1] Load the default model
    if not path.exists(ONNX_PATH):
        raise SystemExit(f"{ONNX_PATH} not found, train it first with RELEASE = \"{DEFAULT_RELEASE}\" in trainer.py")
    session = onnxruntime.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    # The first run allocates buffers, so it would inflate the solve time
    session.run(None, {"input1": np.zeros((1, 1, 80, 96), dtype=np.float32)})
    print(f"[1] Loaded {ONNX_PATH}\n")

    for x in range(6):
        # [2] Generate a captcha
        random = Random()
        code = generate_code(random)
        captcha = render(code, random)
        # Sent as base64, the same shape a captcha usually comes back in from an API
        buffer = BytesIO()
        captcha.save(buffer, format="PNG")
        captcha_base64 = b64encode(buffer.getvalue()).decode()
        print("[2] Generated captcha")
        print(f"    code={code}\n")

        # [3] Solve it, timing decoding, preprocessing and inference together
        started = perf_counter()
        answer, probabilities = solve_captcha(session, captcha_base64)
        solve_time = (perf_counter() - started) * 1000
        letters_right = sum(expected == solved for expected, solved in zip(code, answer))
        # The weakest letter decides the confidence of the whole read
        confidence = probabilities.min()
        verdict = "PASS" if answer == code else "FAIL"
        print("[3] Solved captcha")
        print(f"    answer={answer} time={solve_time:.2f}ms confidence={confidence * 100:.2f}%\n")

        # [4] Draw the captcha on the left and the solution on the right
        canvas = Image.new("RGB", (WIDTH * SCALE + PANEL_WIDTH, HEIGHT * SCALE), (24, 24, 24))
        canvas.paste(captcha.resize((WIDTH * SCALE, HEIGHT * SCALE), Image.NEAREST), (0, 0))
        draw = ImageDraw.Draw(canvas)
        title_font = ImageFont.load_default(size=26)
        font = ImageFont.load_default(size=18)
        left = WIDTH * SCALE + 24
        verdict_color = (80, 200, 120) if verdict == "PASS" else (230, 80, 80)
        draw.text((left, 20), f"{verdict} {answer}", font=title_font, fill=verdict_color)
        rows = [
            ("expected", code),
            ("letters", f"{letters_right}/{CODE_LENGTH}"),
            ("confidence", f"{confidence * 100:.2f}%"),
            ("solve time", f"{solve_time:.2f}ms"),
            ("model", DEFAULT_RELEASE),
        ]
        for index, (label, value) in enumerate(rows):
            draw.text((left, 70 + index * 26), label, font=font, fill=(150, 150, 150))
            draw.text((left + 130, 70 + index * 26), value, font=font, fill=(230, 230, 230))
        # One row per letter, green when it matches the generated code
        for index, (solved, probability) in enumerate(zip(answer, probabilities)):
            color = (80, 200, 120) if solved == code[index] else (230, 80, 80)
            draw.text((left, 210 + index * 20), f"letter {index + 1}", font=font, fill=(150, 150, 150))
            draw.text((left + 130, 210 + index * 20), solved, font=font, fill=color)
            draw.text((left + 170, 210 + index * 20), f"{probability * 100:.2f}%", font=font, fill=color)
        canvas.save(f"captcha_solved_{x}.jpeg")
        #canvas.show()
        print(f"{verdict=} expected {code}, solved {answer} in {solve_time:.2f}ms")
