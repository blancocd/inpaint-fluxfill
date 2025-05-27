import textwrap
from PIL import Image, ImageDraw, ImageFont

def add_text_below_image_wrapped(
    img, text, font_size=30, fill=(0,0,0),
    padding=10, bg_color=(255,255,255), max_width=None, line_spacing=4
):
    # load font
    font = ImageFont.load_default(size=font_size)

    # determine max text width
    if max_width is None:
        max_width = img.width - 2 * padding

    # prepare a drawing context on a dummy image to measure text
    dummy_draw = ImageDraw.Draw(img)

    # wrap text into lines fitting max_width
    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=1000)  # initial break: no limit
        # refine by measuring
        refined = []
        for line in wrapped:
            words = line.split(" ")
            current = ""
            for word in words:
                test = current + (" " if current else "") + word
                left, top, right, bottom = dummy_draw.textbbox((0,0), test, font=font)
                if right - left <= max_width:
                    current = test
                else:
                    if current:
                        refined.append(current)
                    current = word
            if current:
                refined.append(current)
        lines.extend(refined)

    # measure total text block height
    heights = []
    widths = []
    for line in lines:
        l, t, r, b = dummy_draw.textbbox((0,0), line, font=font)
        widths.append(r - l)
        heights.append(b - t)
    text_block_height = sum(heights) + line_spacing * (len(lines)-1)
    text_block_width = max(widths) if widths else 0

    # new canvas size
    new_width = max(img.width, text_block_width + 2*padding)
    new_height = img.height + padding + text_block_height + padding

    # create new image
    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    # paste original centered horizontally
    x_offset = (new_width - img.width) // 2
    new_img.paste(img, (x_offset, 0))

    # draw each line centered below
    draw = ImageDraw.Draw(new_img)
    y = img.height + padding
    for i, line in enumerate(lines):
        left, top, right, bottom = draw.textbbox((0,0), line, font=font)
        line_width = right - left
        x = (new_width - line_width) / 2 - left
        draw.text((x, y - top), line, font=font, fill=fill)
        y += (bottom - top) + line_spacing

    return new_img