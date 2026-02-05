import os 
import argparse 
import cv2 
import math
import json 
import open3d as o3d 
import numpy as np 
from PIL import Image
import torch
from tqdm import tqdm, trange

from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    HardPhongShader,
    SoftPhongShader
)


def render_pcd_pt3d(points, colors, intri_mat, extri_mat, image_size, no_rasterize=False):
    device = points.device
    point_cloud = Pointclouds(points=[points], features=[colors])

    R = extri_mat[:3, :3].clone().to(device).unsqueeze(0)
    T = extri_mat[:3, 3].clone().to(device).unsqueeze(0)
    K = intri_mat[:3, :3].clone().to(device).unsqueeze(0)
    image_size_pt = torch.tensor(image_size).to(device).unsqueeze(0)

    cameras = cameras_from_opencv_projection( R=R, tvec=T, camera_matrix=K, image_size=image_size_pt )

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003, 
        points_per_pixel=10,
        bin_size=0
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        compositor=AlphaCompositor()  # 对于密集点云，Alpha合成效果更好
    )
    image = renderer(point_cloud, cameras=cameras)[0]
    # out_img = image.clone().cpu().detach().numpy() * 255.0
    # out_img = cv2.cvtColor(out_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if no_rasterize:
        return image 

    fragments = renderer.rasterizer(point_cloud)
    # depth_map = fragments.zbuf[0, ..., 0]  
    mask = fragments.idx[0, ..., 0] >= 0
    
    # depth_map_np = depth_map.clone().cpu().detach().numpy()
    # mask_np = mask.clone().cpu().detach().numpy().astype(np.uint8) * 255 

    idx_tensor = fragments.idx[0] 
    flat_indices = idx_tensor.flatten()
    valid_indices = flat_indices[flat_indices >= 0]
    visible_indices = torch.unique(valid_indices).long()
    return image, mask, visible_indices

  
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--recon_res_dir', type=str, required=True)
    parser.add_argument('--lift_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    return args 

def main():
    args = options()
    
    data_dir = args.data_dir
    recon_res_dir = args.recon_res_dir
    lift_dir = args.lift_dir 

    if args.out_dir is None:
        args.out_dir = args.lift_dir
    
    out_root = args.out_dir
    os.makedirs(out_root , exist_ok=True)

    labels_info_path = os.path.join(lift_dir, "info/labels.json")
    with open(labels_info_path, 'r') as fin:
        labels_dict = json.load(fin)

    instance_info_path = os.path.join(lift_dir, "instances/instances_info.json")
    with open(instance_info_path, 'r') as fin:
       instances_info_dict = json.load(fin)
    
    out_dir = os.path.join(out_root, "instances_vis")
    if os.path.exists(out_dir):
        os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    mask_dir = os.path.join(data_dir, "masks/raw")
    mask_filenames = sorted([x for x in os.listdir(mask_dir) if x.endswith(".png")])

    detailed_label_path = os.path.join(data_dir, "masks/label_mapping.json")
    with open(detailed_label_path, 'r') as fin:
        detailed_label_dict = json.load(fin)

    raw_img_dir = os.path.join(data_dir, "raw_images")
    raw_img_filenames = sorted([x for x in os.listdir(raw_img_dir) if x.endswith(".jpg") or x.endswith(".png")])
    
    color_dir = os.path.join(data_dir, "color")
    color_filenames = sorted([x for x in os.listdir(color_dir) if x.endswith(".png")])
    color_path = os.path.join(color_dir, color_filenames[0])
    img = cv2.imread(color_path)
    img_height, img_width, _ = img.shape 
    
    cam_dir = os.path.join(recon_res_dir, "camera")
    cam_filenames = sorted([x for x in os.listdir(cam_dir) if x.endswith(".npz")])
    frame_total = len(cam_filenames)

    src_color_mesh_path = os.path.join(recon_res_dir, "point_cloud_human3r.ply")
    pcd = o3d.io.read_point_cloud(src_color_mesh_path)
    colors_np = np.array(pcd.colors, dtype=np.float32)
    xyz_pos_np = np.array(pcd.points, dtype=np.float32)

    device = 'cuda'
    colors = torch.from_numpy(colors_np).to(device)
    xyz_pos = torch.from_numpy(xyz_pos_np).to(device)
    
    K_arr = []
    w2c_arr = []
    pbar = tqdm(range(frame_total), desc=f"Pre-processing")
    with torch.no_grad():
        for idx, cam_fn in enumerate(cam_filenames):
            name = cam_fn.split(".")[0]

            pose_path = os.path.join(cam_dir, cam_fn)
            data = np.load(pose_path)
            c2w = data["pose"]
            intri_mat = data["intrinsics"]
            data.close()

            w2c = np.linalg.inv(c2w)
            w2c_arr.append(torch.from_numpy(w2c).to(dtype=torch.float32))
            
            K = intri_mat.copy().astype(np.float32)
            K_arr.append(torch.from_numpy(K).to(dtype=torch.float32))
            pbar.update(1)
        
        for inst_label_name in instances_info_dict:
            cur_insts_arr = instances_info_dict[inst_label_name]
            for _i, cur_inst_info in tqdm(enumerate(cur_insts_arr)):
                inst_arr = cur_inst_info["inst_label"]
                frames_arr = sorted({
                    f for inst_idx in inst_arr
                    for f in detailed_label_dict[str(inst_idx)]["frames"]
                })
                pbar = tqdm(range(len(frames_arr)), desc=f"Selecting views for {inst_label_name}_{_i:02d}")

                point_idx_arr = cur_inst_info["point_idx"]
                cur_points = xyz_pos[point_idx_arr].clone()
                cur_total = len(cur_points)
                    
                valid_info_arr = []
                valid_num = 5
                for idx in frames_arr:
                    pbar.update(1)

                    cam_fn = cam_filenames[idx]
                    name = cam_fn.split(".")[0]
                    
                    #### get object's position in image space 
                    w2c = w2c_arr[idx].to(cur_points.device)
                    K = K_arr[idx].to(cur_points.device)

                    cur_colors = colors[point_idx_arr].clone()
                    obj_img, obj_mask, screen_indices = render_pcd_pt3d(cur_points, cur_colors, K, w2c, [img_height, img_width])
                    obj_screen_total = len(screen_indices)
                    point_num_ratio = obj_screen_total / cur_total
                    if point_num_ratio < 0.2:
                        continue 
                    
                    seg_mask_path = os.path.join(mask_dir, mask_filenames[idx])
                    seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_UNCHANGED)

                    valid_pixels = np.zeros(seg_mask.shape, dtype=bool)
                    for inst_label in inst_arr:
                        valid_pixels[seg_mask == inst_label] = True

                    valid_pixel_num = np.sum(valid_pixels)
                    if valid_pixel_num < 1:
                        continue 

                    obj_mask_bool = obj_mask.bool()
                    obj_pixel_num = obj_mask_bool.sum().item()
                    valid_ratio = valid_pixel_num / obj_pixel_num
                    cur_point_num = obj_screen_total * valid_ratio
            
                    cur_item = {
                        "idx": idx, 
                        "name": name,
                        "point_num": cur_point_num,
                        "valid_pixels": valid_pixels,
                    }

                    if len(valid_info_arr) < valid_num:
                        valid_info_arr.append(cur_item)
                    else:
                        min_item = min(valid_info_arr, key=lambda x: x["point_num"] )
                        if (cur_item["point_num"]) > (min_item["point_num"]):
                            valid_info_arr[valid_info_arr.index(min_item)] = cur_item
                                        
                if len(valid_info_arr) < 1:
                    continue    

                cur_out_dir = os.path.join(out_dir, f"{inst_label_name}_{_i:02d}")
                os.makedirs(cur_out_dir, exist_ok=True)

                valid_info_arr = sorted(valid_info_arr, key=lambda x: x["point_num"], reverse=True)
                render_idx_arr = [torch.tensor(point_idx_arr, dtype=torch.long, device=cur_points.device)]
                for vidx, valid_info in enumerate(valid_info_arr):
                    idx = valid_info["idx"]
                    name = valid_info["name"]
                    valid_pixels = valid_info["valid_pixels"]
                    
                    image_path = os.path.join(raw_img_dir, raw_img_filenames[idx])
                    img = cv2.imread(image_path)
                    ch, cw, _ = img.shape
                    out_path = os.path.join(cur_out_dir, f"{vidx}_{name}_orig.png")
                    cv2.imwrite(out_path, img)
                    
                    mask_image = valid_pixels.copy().astype(np.uint8) * 255
                    mask_image = cv2.resize(mask_image, (cw, ch), interpolation=cv2.INTER_NEAREST)
                    out_path = os.path.join(cur_out_dir, f"{vidx}_{name}_mask.png")
                    cv2.imwrite(out_path, mask_image)
                    
 
if __name__ == "__main__":
    main()

    print("=== done")