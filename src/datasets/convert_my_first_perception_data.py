from PIL import Image
import itertools
import numpy as np
import json
from glob import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", help="dataset path", type=str)
parser.add_argument("-o", "--output_dir",
                    default="./my_first_perception", help="output directory", type=str)
args = parser.parse_args()


def load_palette(specs):
  colors = [[0, 0, 0]]
  for s in specs:
    v = s['pixel_value']
    colors.append([int(v['r']*0xff), int(v['g']*0xff), int(v['b']*0xff)])
  NUM_ENTRIES_IN_PILLOW_PALETTE = 256
  palette_arr = list(itertools.chain.from_iterable(colors))
  palette_arr.extend(
      [0, 0, 0] * (NUM_ENTRIES_IN_PILLOW_PALETTE - len(colors)))
  palette_img = Image.new('P', (0, 0))
  palette_img.getdata().putpalette('RGB', bytes(palette_arr))
  return palette_img


dataset_dir = args.dataset_dir
if dataset_dir[-1] == "/":
  dataset_dir = dataset_dir[:-1]
output_dir = args.output_dir
if output_dir[-1] == "/":
  output_dir = output_dir[:-1]
dataset = os.path.join(args.output_dir, os.path.basename(dataset_dir))
os.makedirs(os.path.join(dataset, 'ImageSets'), exist_ok=True)
os.makedirs(os.path.join(dataset, 'JPEGImages'), exist_ok=True)
os.makedirs(os.path.join(dataset, 'SegmentationClass'), exist_ok=True)

for path in glob(os.path.join(dataset_dir, '*')):
  variant = os.path.basename(path)
  if variant in ['ImageSets', 'JPEGImages', 'SegmentationClass']:
    continue
  for path in glob(os.path.join(path, 'Dataset*')):
    with open(os.path.join(path, 'annotation_definitions.json')) as f:
      d = json.load(f)
      assert d['version'] == '0.0.1'
  for definition in d['annotation_definitions']:
    if definition['name'] == 'semantic segmentation':
      palette = load_palette(definition['spec'])
      break
  file_ids = []
  img_size = None
  for path in glob(os.path.join(dataset_dir, variant, 'SemanticSegmentation*')):
    for src in glob(os.path.join(path, '*.png')):
      name, _ = os.path.splitext(os.path.basename(src))
      _, img_id = name.split('_')
      file_id = f"{variant}_{img_id}"
      img = Image.open(src).convert(
          'RGB').quantize(palette=palette, dither=0)
      annotation = Image.fromarray(
          np.array(img, dtype=np.uint8), mode='L')
      if img_size is None:
        img_size = img.size
      annotation.save(os.path.join(
          dataset, 'SegmentationClass', f"{file_id}.png"))
  for path in glob(os.path.join(dataset_dir, variant, 'RGB*')):
    for src in glob(os.path.join(path, '*.png')):
      name, _ = os.path.splitext(os.path.basename(src))
      _, img_id = name.split('_')
      file_id = f"{variant}_{img_id}"
      file_ids.append(file_id)
      img = Image.open(src).convert('RGB')
      img = img.resize(img_size)
      img.save(os.path.join(dataset, 'JPEGImages', f"{file_id}.jpg"))

  with open(os.path.join(dataset, 'ImageSets', f"{variant}.txt"), 'w') as w:
    w.write("\n".join(file_ids))
