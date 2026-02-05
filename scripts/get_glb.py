import sys
import os
import torch
import torch.nn.functional as F
import contextlib
import spconv.pytorch as spconv
import imageio
from PIL import Image
import numpy as np
import glob
from pathlib import Path
import argparse


def create_transparent_background(original_image_path, mask_image_path, output_path, threshold=127):
    original_img = Image.open(original_image_path).convert('RGB')
    mask_img = Image.open(mask_image_path).convert('L') 
    
    if original_img.size != mask_img.size:
        mask_img = mask_img.resize(original_img.size, Image.Resampling.LANCZOS)
    
    mask_array = np.array(mask_img)
    
    foreground_mask = mask_array > threshold
    
    alpha = np.zeros(mask_array.shape, dtype=np.uint8)
    alpha[foreground_mask] = 255 
    
    rgb_array = np.array(original_img)
    rgba_array = np.dstack((rgb_array, alpha))
    
    rgba_img = Image.fromarray(rgba_array, 'RGBA')
    rgba_img.save(output_path, 'PNG', optimize=True)
    
    return True
        


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_img', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    return args 


def main():
    args = options()
    input_dir = args.best_img
    output_dir = args.out_dir

    os.makedirs(output_dir, exist_ok=True)

    sys.path.append("notebook")
    from inference import Inference, load_image, load_single_mask, render_video, ready_gaussian_for_video_rendering, load_mask

    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    orig_files = glob.glob(os.path.join(input_dir, "*_ori.png"))

    for i, orig_path in enumerate(orig_files):
    
        filename = os.path.basename(orig_path)
        file_id = filename.replace("_ori.png", "")
        
        print(f"\n[{i+1}/{len(orig_files)}] 正在处理: {file_id}")
        
        seg_path = os.path.join(input_dir, f"{file_id}_mask.png")
        white_path = os.path.join(input_dir, f"{file_id}_white.png")  # 临时生成的透明mask
        output_glb_name = os.path.join(output_dir, f"{file_id}.glb")
        
        if not os.path.exists(seg_path):
            print(f"  [跳过] 找不到对应的 seg 文件: {seg_path}")
            continue

        success = create_transparent_background(orig_path, seg_path, white_path)
        if not success:
            continue

        image = load_image(orig_path)
        mask = load_mask(white_path)


        output = inference(image, mask, seed=42)

        output["glb"].export(output_glb_name)
        print(f"  ✅ 成功保存: {output_glb_name}")
    
if __name__ == "__main__":
    main()
    print("=== done", flush=True)