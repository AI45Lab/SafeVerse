#!/usr/bin/env python3
import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import shutil
from copy import deepcopy
import imageio.v2 as iio
import roma
import open3d as o3d 

import sys 
cur_filepath = os.path.abspath(__file__)
work_dir = os.path.dirname(os.path.dirname(cur_filepath))
submodule_root = os.path.join(work_dir, "submodules/Human3R")
sys.path.append(submodule_root)

from add_ckpt_path import add_path_to_dust3r



# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    return parser.parse_args()

def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)

def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True, 
        save=False, img_res=None):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save (bool): Whether to save output results.
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer, vis_heatmap, render_meshes
    from src.dust3r.utils.image import unpad_image
    from viser_utils import get_color

    if save:
        os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)


    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")
    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    has_mask = "msk" in outputs["pred"][0]
    # print("=== has_mask", has_mask, flush=True)
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]
    
    masks_bg = []
    points_rgb = []
    for f_id in range(B):
        depth = depths_tosave[f_id].numpy()
        conf = conf_self_tosave[f_id].numpy()
        color = colors_tosave[f_id].numpy() 
        mask = (torch.clamp(msks[f_id], 0, 1).squeeze().numpy() * 255).astype(np.uint8) 
        
        _, mask_thre = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY_INV)
        masks_bg.append(mask_thre)
        mask_thre = mask_thre[..., None ] / 255.0
        color_bg = (color * mask_thre * 255).astype(np.uint8)
        points_rgb.append(color_bg)

        c2w = cam2world_tosave[f_id].numpy()
        intrins = intrinsics_tosave[f_id].numpy()

        if save:
            np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
            np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
            iio.imwrite(
                os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color * 255).astype(np.uint8),
            )
            np.savez(
                os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                pose=c2w,
                intrinsics=intrins,
            )

    masks_bg = np.stack(masks_bg, axis=0)
    points_rgb = np.stack(points_rgb, axis=0)
    points_xyz = pts3ds_other_tosave.cpu().detach().numpy()
    conf = conf_other_tosave.cpu().detach().numpy()
    
    conf_threshold = max(np.percentile(conf, 25), 1.5)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    black_bg_mask = masks_bg > 50
    conf_mask = conf_mask & black_bg_mask
    max_points_for_colmap = 1_000_000 
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
    
    true_indices = np.nonzero(conf_mask)
    np.save(os.path.join(outdir, "true_indices.npy"), true_indices)

    points_xyz = points_xyz[conf_mask].reshape(-1, 3)
    points_rgb = points_rgb[conf_mask].reshape(-1, 3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(points_rgb.astype(np.float32) / 255.0)
    o3d.io.write_point_cloud(os.path.join(outdir, "point_cloud_human3r.ply"), pcd)

def parse_seq_path(p, out_dir):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
    else:
        out_img_dir = os.path.join(out_dir, "raw_images")
        os.makedirs(out_img_dir, exist_ok=True)
        img_paths = sorted(glob.glob(f"{out_img_dir}/*"))

        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if len(img_paths) < total_frames:
            frame_interval = 1
            frame_indices = list(range(0, total_frames, frame_interval))
            print(
                f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
            )
            img_paths = []
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(out_img_dir, f"{i:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                img_paths.append(frame_path)
            cap.release()
    return img_paths

def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import SceneHumanViewer

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    img_res = getattr(model, 'mhmr_img_res', None)
    
    img_paths = parse_seq_path(args.seq_path, args.output_dir)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval
    )

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    prepare_output(outputs, args.output_dir, 1, True, True, img_res)
    print(
        f"Finished: {time.time() - start_time:.2f} seconds."
    )
        

def main():
    args = parse_args()
    run_inference(args)

    
if __name__ == "__main__":
    main()
