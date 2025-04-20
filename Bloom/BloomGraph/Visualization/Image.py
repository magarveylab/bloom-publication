import os
from typing import List

import cairosvg
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from Bloom.BloomGraph.DataStructs import ImageDataDict


def create_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def prepare_mol_image(svg_data: str, mol_png_fp: str):
    # export svg
    svg_fp = mol_png_fp.replace(".png", ".svg")
    with open(svg_fp, "w") as svg:
        svg.write(svg_data)
    svg.close()
    # convert to png
    cairosvg.svg2png(
        url=svg_fp,
        write_to=mol_png_fp,
        dpi=300,
        parent_height=1500,
        parent_width=1500,
    )
    os.remove(svg_fp)
    # load png
    return Image.open(mol_png_fp)


def prepare_legend(colour_map: dict, legend_png_fp: str):
    # prepare legend
    legend_data = [(colour, unit) for unit, colour in colour_map.items()]
    rows = [legend_data[i : i + 3] for i in range(0, len(legend_data), 3)]
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.gca()
    # dimensions and creating circle tooltips
    y = 0.95
    for row in rows:
        x = 0.05
        for entry in row:
            circle = plt.Circle((x, y), 0.03, color=entry[0])
            ax.add_patch(circle)
            plt.text(x + 0.05, y - 0.0075, entry[1], fontsize=12)
            x += 0.35
        y -= 0.1
    max_y = (0.1 + (len(rows) - 1) * 0.1 + 0.06) * 3000
    # plot
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(legend_png_fp, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    # export
    with Image.open(legend_png_fp) as im:
        im_crop = im.crop((0, 0, 3000, max_y))
        im_crop.save(legend_png_fp)
    # load png
    return Image.open(legend_png_fp)


def merge_imgs(mol_img, legend_img, combined_png_fp):
    # Modify mol
    width, height = mol_img.size
    new_height = height + 400
    # Create background
    background = Image.new(mol_img.mode, (width, new_height), "white")
    background.paste(mol_img, (0, 0))
    # Resize legend
    basewidth = int(width * 1.1)
    h_perc = float(basewidth / legend_img.size[0])
    new_height = int(float(legend_img.size[1]) * h_perc)
    legend_img = legend_img.resize((basewidth, new_height), Image.ANTIALIAS)
    legend_width, legend_height = legend_img.size
    background.paste(legend_img, (0, height - 50))
    background = background.crop((0, 0, width, height + legend_height))
    # save
    background.save(combined_png_fp)
    return background


def get_image(
    data: ImageDataDict,
    tmp_dir: str,
    img_id: str = "tmp",
    add_legend: bool = True,
):
    # prep local cache dir
    create_dir(tmp_dir)
    # export svg
    mol_png_fp = "{}/mol_{}.png".format(tmp_dir, img_id)
    mol_img = prepare_mol_image(data["svg"], mol_png_fp)
    # prepare legend
    if add_legend == True:
        legend_png_fp = "{}/legend_{}.png".format(tmp_dir, img_id)
        legend_img = prepare_legend(data["colour_map"], legend_png_fp)
        combined_png_fp = "{}/combined_{}.png".format(tmp_dir, img_id)
        img = merge_imgs(mol_img, legend_img, combined_png_fp)
        return img
    else:
        return mol_img


def combine_images(images):
    mode = images[0].mode
    widths = []
    heights = []
    for i in images:
        w, h = i.size
        widths.append(w)
        heights.append(h)
    # new dimension to image
    new_width = sum(widths)
    new_height = max(heights)
    # Create background
    background = Image.new(mode, (new_width, new_height), "white")
    # paste images
    for idx, i in enumerate(images):
        if idx == 0:
            background.paste(i, (0, 0))
        else:
            background.paste(i, (widths[idx - 1], 0))
    return background
