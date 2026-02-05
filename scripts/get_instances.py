import os
import argparse 
import json
import cv2
import numpy as np 
import trimesh
import open3d as o3d 
import copy 
from sklearn.cluster import DBSCAN

np.random.seed(100)
def generate_random_colors(label_list):
    colors = {}
    for i, label in enumerate(label_list):
        reroll = True
        iter_cnt = 0
        while reroll and iter_cnt < 100:
            iter_cnt += 1
            reroll = False
            color = np.random.randint(1, 255, 3)
            for selected_color in colors.values():
                if np.linalg.norm(np.array(color) - np.array(selected_color)) < 70:
                    reroll = True
                    break
        colors[int(label)] = color
    return colors

def visualize(points, point_labels, out_path):
    unique_labels = np.unique(point_labels)
    label_num = len(unique_labels)

    point_colors = np.zeros((len(point_labels), 3), dtype=np.float32)
    colors = generate_random_colors(unique_labels)
    for i, label in enumerate(unique_labels):
        point_colors[point_labels == label] = colors[label] / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(out_path, pcd)

def get_valid_label(label_path):
    with open(label_path, 'r') as fin:
        detailed_label_dict = json.load(fin)

    valid_label_dict = {}
    for key, val in detailed_label_dict.items():
        if ("invalid" in val) and val["invalid"]:
            continue 
        label = int(key)
        label_name = val["label"]
        valid_label_dict[label] = label_name
    return valid_label_dict

def load_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    colors = np.array(pcd.colors, dtype=np.float32)
    points = np.array(pcd.points, dtype=np.float32)
    return points, colors

def read_masks(mask_dir):
    mask_arr = []
    mask_filenames = sorted([x for x in os.listdir(mask_dir) if x.endswith(".png")])
    for i, mask_fn in enumerate(mask_filenames):
        mask_path = os.path.join(mask_dir, mask_fn)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_arr.append(mask_img)
    mask_arr = np.stack(mask_arr, axis=0)
    return mask_arr

def get_valid_mask3D(mask_arr, valid_label_dict, true_indices):
    points_mask = mask_arr[true_indices[:, 0], true_indices[:, 1], true_indices[:, 2]].reshape(-1)
    valid_label_arr = list(valid_label_dict.keys())
    points_mask[~np.isin(points_mask, valid_label_arr)] = 0
    return points_mask

def bind_semantic_labels(instance_points, valid_label_dict):
    semantic_points = np.ones_like(instance_points) * -1
    valid_label_names = list(set(valid_label_dict.values()))
    label_info_dict = {}
    for sem_idx, label_name in enumerate(valid_label_names):
        label_info_dict[label_name] = {
            "sem_idx": int(sem_idx),
            "inst_arr": []
        }
    
    inst_unique_labels = np.unique(instance_points)
    for inst_label in inst_unique_labels:
        if inst_label < 1:
            continue 
        
        if inst_label not in valid_label_dict:
            continue 

        sem_label_name = valid_label_dict[inst_label]
        sem_idx = label_info_dict[sem_label_name]["sem_idx"]
        inst_idx_arr = np.where(instance_points == inst_label)[0]
        semantic_points[inst_idx_arr] = sem_idx
        label_info_dict[sem_label_name]["inst_arr"].append(int(inst_label))
    
    return label_info_dict, semantic_points

def filter_semantic_instance(sem_idx_arr, points, instance_labels):
    sem_points = points[sem_idx_arr].copy()
    sem_inst_labels = instance_labels[sem_idx_arr].copy()
    unique_labels, unique_counts = np.unique(sem_inst_labels, return_counts=True)
    
    sort_idx = np.argsort(unique_counts)[::-1] 
    box_info_arr = []
    for idx in sort_idx:
        inst_label = unique_labels[idx]
        if inst_label < 0:
            continue 

        cur_idx_arr = np.where(sem_inst_labels == inst_label)[0]        
        min_samples = min(int(len(cur_idx_arr) / 10), 500)
        eps = 0.25
        db = DBSCAN(eps=eps, min_samples=min_samples) 
        point_cluster_labels = db.fit(sem_points[cur_idx_arr]).labels_
        unique_cluster_labels = np.unique(point_cluster_labels)
        valid_seg = np.sum(unique_cluster_labels > -1)
        cur_idx_arr = np.array(cur_idx_arr)[point_cluster_labels > -1].tolist()
        
        inst_point_idx_arr = sem_idx_arr[cur_idx_arr]
        #### 4 is the threshold for trimesh to build bbox 
        if len(inst_point_idx_arr) < 4:
            continue  

        cur_points = sem_points[cur_idx_arr]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cur_points)
        obb = pcd.get_minimal_oriented_bounding_box()
        box = trimesh.primitives.Box(
            extents=obb.extent,
            transform=np.vstack([
                np.hstack([obb.R, obb.center.reshape(3, 1)]),
                [0, 0, 0, 1]
            ])
        )
        cur_box_info = {
            "inst_idx": [inst_label],
            "points": cur_points,
            "point_idx": list(inst_point_idx_arr),
            "box": box
        } 
        box_info_arr.append(cur_box_info)
        
    return box_info_arr

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--recon_res_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default="./results")
    args = parser.parse_args()
    return args 

def main():
    args = options()
    data_dir = args.data_dir
    recon_res_dir = args.recon_res_dir
    out_root = args.out_dir 
    os.makedirs(args.out_dir , exist_ok=True)

    out_info_dir = os.path.join(out_root, "info")
    os.makedirs(out_info_dir, exist_ok=True)

    detailed_label_path = os.path.join(data_dir, "masks/label_mapping.json")
    valid_label_dict = get_valid_label(detailed_label_path)
    
    mask_dir = os.path.join(data_dir, "masks/raw")
    mask_arr = read_masks(mask_dir)
    
    pcd_path = os.path.join(recon_res_dir, "point_cloud_human3r.ply")
    points, colors = load_pcd(pcd_path)

    true_indices_path = os.path.join(recon_res_dir, "true_indices.npy")
    true_indices = np.load(true_indices_path).T

    points_mask = get_valid_mask3D(mask_arr, valid_label_dict, true_indices)

    instance_labels = points_mask.copy()
    out_inst_seg_path = os.path.join(out_info_dir, "instance_seg.npy")
    np.save(out_inst_seg_path, instance_labels)
    # visualize(points, instance_labels, os.path.join(out_info_dir, "instance_seg.ply"))

    ### semantic part
    label_info_dict, semantic_labels = bind_semantic_labels(instance_labels, valid_label_dict)
    out_sem_seg_path = os.path.join(out_info_dir, "semantic_seg.npy")
    np.save(out_sem_seg_path, semantic_labels)
    # visualize(points, semantic_labels, os.path.join(out_info_dir, "semantic_seg.ply"))

    out_json_path = os.path.join(out_info_dir, "labels.json")
    with open(out_json_path, 'w') as fout:
        json.dump(label_info_dict, fout)

    #### to separate every instance
    out_instance3d_dir = os.path.join(out_root, "instances")
    os.makedirs(out_instance3d_dir, exist_ok=True)

    orig_cls_names = list(label_info_dict.keys())
    out_dict = {}
    unique_sem_labels = np.unique(semantic_labels)
    for sem_label in unique_sem_labels:
        if sem_label < 0:
            continue 

        label_name = orig_cls_names[sem_label]
        if label_name in ['wall', 'floor', 'ceiling', 'window']:
            continue 

        sem_idx_arr = np.where(semantic_labels == sem_label)[0]
        box_info_arr = filter_semantic_instance(sem_idx_arr, points, instance_labels)
        if len(box_info_arr) < 1:
            continue 

        out_dict[label_name] = []
        out_sem_dir = os.path.join(out_instance3d_dir, label_name)
        os.makedirs(out_sem_dir, exist_ok=True)
        for i, box_info in enumerate(box_info_arr):
            point_idx_arr = box_info["point_idx"]
            box_points = points[point_idx_arr]
            box_colors = colors[point_idx_arr]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(box_points)
            pcd.colors = o3d.utility.Vector3dVector(box_colors)
            o3d.io.write_point_cloud(f"{out_sem_dir}/points_{i}.ply", pcd)

            out_dict[label_name].append({
                "point_idx": list(map(int, point_idx_arr)),
                "inst_label": list(map(int, box_info["inst_idx"]))
            })
        print(f"==== sem_label={sem_label}, {label_name}, obj_num={len(box_info_arr)}", flush=True)

    out_path = os.path.join(out_instance3d_dir, "instances_info.json")
    with open(out_path, "w") as fout:
        json.dump(out_dict, fout)

    return  

if __name__ == "__main__":
    main()
    print("=== done", flush=True)