from functools import lru_cache
from math import cos, radians, sin
from random import Random
from PIL import Image, ImageDraw, ImageFont

# Yii captcha font
FONT_PATH = "./assets/SpicyRice.ttf"

# Yii constants
CHARSET = "bcdfghjklmnpqrstvwyz"
CODE_LENGTH = 4

# Ticketmaster sg / ph config
WIDTH = 120
HEIGHT = 100
PADDING = 2
OFFSET = -2
BACK_COLOR = (0x00, 0x46, 0x7F)
FORE_COLOR = (0xFF, 0xFF, 0xFF)

POINTS_TO_PIXELS = 96 / 72

@lru_cache(maxsize=256)
def load_font(point_size):
    return ImageFont.truetype(FONT_PATH, max(int(round(point_size * POINTS_TO_PIXELS)), 1))


@lru_cache(maxsize=8192)
def glyph_box(point_size, letter):
    return load_font(point_size).getbbox(letter, anchor="ls")


def generate_code(random):
    return "".join(random.choice(CHARSET) for _ in range(CODE_LENGTH))


def render(code, random):
    # CaptchaAction::renderImageByGD ported from PhP
    image = Image.new("RGB", (WIDTH, HEIGHT), BACK_COLOR)
    font = load_font(30)
    left, top, right, bottom = font.getbbox(code)
    box_width = (right - left) + OFFSET * (len(code) - 1)
    box_height = bottom - top
    scale = min((WIDTH - PADDING * 2) / box_width, (HEIGHT - PADDING * 2) / box_height)
    x = 10
    y = round(HEIGHT * 27 / 40)
    for letter in code:
        point_size = int(random.randint(26, 32) * scale * 0.8)
        angle = random.randint(-10, 10)
        pad = 80
        layer = Image.new("L", (pad * 2, pad * 2), 0)
        ImageDraw.Draw(layer).text((pad, pad), letter, font=load_font(point_size), fill=255, anchor="ls")
        if angle:
            layer = layer.rotate(angle, resample=Image.BICUBIC, center=(pad, pad))
        image.paste(FORE_COLOR, (x - pad, y - pad), layer)
        left, top, right, bottom = glyph_box(point_size, letter)
        radian = radians(angle)
        x = x + int(round(right * cos(radian) + bottom * sin(radian))) + OFFSET
    return image


def sample(count, seed=0):
    random = Random(seed)
    for _ in range(count):
        code = generate_code(random)
        yield code, render(code, random)


if __name__ == '__main__':
    random = Random()
    code = generate_code(random)
    image = render(code, random)
    image.show()
    #image.save("test.jpeg")