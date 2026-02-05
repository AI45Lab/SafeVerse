import os 
import argparse
import copy 
import json 
import cv2 
import numpy as np 
import open3d as o3d
import trimesh 
from collections import deque
from scipy import ndimage 
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

# Generate random colors
def generate_random_colors(label_list):
    colors = {}
    for i, label in enumerate(label_list):
        if label is None:
            continue 

        reroll = True
        iter_cnt = 0
        while reroll and iter_cnt < 100:
            iter_cnt += 1
            reroll = False
            color = tuple(np.random.randint(1, 255, 3) / 255.0)
            for selected_color in colors.values():
                if np.linalg.norm(np.array(color) - np.array(selected_color)) < 70:
                    reroll = True
                    break
        colors[int(label)] = color
    return colors

def visualize(grid, basic_idx_arr):
    np.random.seed(123)
    predefined_colors = generate_random_colors(basic_idx_arr)

    gx, gy, gz = grid.shape
    valid_idx_arr = np.argwhere(grid >= 0)
    points = []
    colors = []
    unique_labels = []
    for vidx in valid_idx_arr:
        x, y, z = vidx 
        label = grid[x, y, z]
        if (label < 0) or (label not in basic_idx_arr):
            continue 
        color = predefined_colors[label]

        points.append([gx - 1 - x, z, y])
        colors.append(color)

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)

def save_verts(verts, colors=None, file_name="./check.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(verts))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    o3d.io.write_point_cloud(file_name, pcd)

def align_points_to_axes(points, rotation_matrix, center=None):
    if center is None:
        center = np.mean(points, axis=0)
    
    centered_points = points - center
    aligned_points = centered_points @ rotation_matrix
    return aligned_points

def get_floor_rotate(points, semantic_labels, floor_idx):
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    voxel_size = np.around((bbox_max[2] - bbox_min[2]) / 9, decimals=2)
    voxel_size = min(0.5, voxel_size)
    value_ranges = np.stack([bbox_min, bbox_max], axis=1)
    grid = get_grid(bbox_min, bbox_max, voxel_size=voxel_size)
    
    mask = (semantic_labels == floor_idx)
    floor_idx_arr = np.where(mask)[0]
    floor_points = points[floor_idx_arr]
    floor_grid_arr = get_points_on_grid(floor_points, value_ranges, grid.shape)

    max_axis = 0
    max_plane_idx_len = 0
    for axis in range(3):
        vert_plane = floor_grid_arr.copy()
        vert_plane[:, axis] = 0

        vert_plane_idx = np.unique(vert_plane, axis=0)
        if max_plane_idx_len < len(vert_plane_idx):
            max_plane_idx_len = len(vert_plane_idx)
            max_axis = axis 

    half_axis = int(grid.shape[max_axis] / 2)

    floor_axis_arr = floor_grid_arr[:, max_axis].copy()
    # print("=== floor_axis_arr", floor_axis_arr.shape, flush=True)
    floor_axis_val_unique, floor_axis_val_counts = np.unique(floor_axis_arr, return_counts=True)
    
    floor_axis_val_max = floor_axis_val_unique[floor_axis_val_counts.argmax()]
    floor_normal = np.zeros(3)
    # print("=== floor", floor_axis_val_max, max_axis, flush=True)
    floor_normal[max_axis] = -1 if floor_axis_val_max < half_axis else 1 

    # print("=== floor", max_axis, flush=True)
    if max_axis == 2:
        R_adjust = np.eye(3)
        if floor_axis_val_max > half_axis:
            rotation = Rotation.from_euler('y', 180, degrees=True)
            R_adjust = rotation.as_matrix()
        return R_adjust

    v = np.cross(floor_normal, [0, 0, 1])
    s = np.linalg.norm(v)
    c = np.dot(floor_normal, [0, 0, 1])
    
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        
        # Rodrigue's rotation formula
        R_adjust = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    else:
        R_adjust = np.eye(3)
    
    return R_adjust

def compute_aligned_axis_points(points, semantic_labels, label_mapping_dict):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_minimal_oriented_bounding_box()

    obb_center = obb.center 
    rotation_matrix = obb.R 
    points = np.array(pcd.points, dtype=np.float32)
    aligned_points_v1 = align_points_to_axes(points, rotation_matrix, obb_center)

    #### force floor on the negative-z axis 
    floor_idx = label_mapping_dict["floor"]["sem_idx"]
    R_adjust = get_floor_rotate(aligned_points_v1, semantic_labels, floor_idx)
    aligned_points_v2 = align_points_to_axes(aligned_points_v1, R_adjust)

    return aligned_points_v2

def get_bottom_center(pcd):
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()

    bbox_center = (max_bound + min_bound) / 2.0
    bottom_z = min_bound[2]
    
    bbox_center[2] = bottom_z
    return bbox_center

def get_src_init_xform(src_pcd, tgt_pcd, rotation=None):
    if rotation is None:
        rotation = Rotation.from_euler('x', 90, degrees=True)
    R = np.eye(4)
    R[:3, :3] = rotation.as_matrix()
    
    src_center = src_pcd.get_center()
    tgt_center = tgt_pcd.get_center()
    
    T_to_orig = np.eye(4)
    T_to_orig[:3, 3] = -src_center

    T_to_tgt = np.eye(4)
    T_to_tgt[:3, 3] = tgt_center

    xform = T_to_tgt @ R @ T_to_orig
    return xform 

def get_src_scale_xform(src_pcd, tgt_pcd, select_axis=None):
    src_size = src_pcd.get_max_bound() - src_pcd.get_min_bound()
    tgt_size = tgt_pcd.get_max_bound() - tgt_pcd.get_min_bound()
    
    src_center = src_pcd.get_center()
    tgt_center = tgt_pcd.get_center()
    
    T_to_orig = np.eye(4)
    T_to_orig[:3, 3] = -src_center

    T_to_tgt = np.eye(4)
    T_to_tgt[:3, 3] = tgt_center
    
    if select_axis is None:
        init_scale = tgt_size[2] / src_size[2]

        ratio = max(src_size * init_scale / tgt_size)
        if ratio >= 1.5:
            extent_scale = tgt_size / src_size 
            select_axis = np.argsort(extent_scale)[0]
            scale = extent_scale[select_axis]
        else:
            scale = init_scale
            select_axis = 2
    else:
        scale = tgt_size[select_axis] / src_size[select_axis]

    S = np.eye(4)
    np.fill_diagonal(S[:3, :3], scale)

    xform = T_to_tgt @ S @ T_to_orig
    return xform, select_axis

def get_align_bottom_translate(src_pcd, tgt_pcd):
    src_bottom_center = get_bottom_center(src_pcd)
    tgt_bottom_center = get_bottom_center(tgt_pcd)

    translate = tgt_bottom_center - src_bottom_center
    return translate

def get_final_scale_xform(pcd, voxel_size, grid_shape, update_scale=True, int_format=None):
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()
    size = max_bound - min_bound

    mesh_scale = 1.0 / voxel_size
    mesh_size = size * mesh_scale
    if mesh_size[2] < 1.0:
        ideal_size = mesh_size[2]
    else:
        if int_format is None:
            #### 0.3 is a trick value
            ideal_size = int(mesh_size[2] + 0.3)
        elif int_format == "floor":
            ideal_size = np.floor(mesh_size[2])
        elif int_format == "ceil":
            ideal_size = np.ceil(mesh_size[2])
        elif int_format == "orig":
            ideal_size = mesh_size[2]
        elif isinstance(int_format, int) or isinstance(int_format, float):
            ideal_size = int_format
        else:
            raise ValueError(f"invalid int_format {int_format}")

    # print("=== ideal_size", ideal_size, mesh_size, flush=True)
    scale = ideal_size / mesh_size[2] * mesh_scale
    
    x_thre = grid_shape[0] * 0.7
    y_thre = grid_shape[1] * 0.7
       
    if update_scale and ideal_size > 1:
        while ((size[0] * scale) >= x_thre ) or ((size[1] * scale) >= y_thre):
            ideal_size -= 1 
            scale = ideal_size / mesh_size[2] * mesh_scale
            print("==== update scale", ideal_size, size*scale, x_thre, y_thre, flush=True)

            if ideal_size == 1:
                break  

    center = pcd.get_center()

    S = np.eye(4)
    np.fill_diagonal(S[:3, :3], scale)

    T_to_orig = np.eye(4)
    T_to_orig[:3, 3] = -center

    T_to_back = np.eye(4)
    T_to_back[:3, 3] = center

    xform = T_to_back @ S @ T_to_orig
    return xform
    
def chamfer_distance(src_pcd, tgt_pcd):
    dists1 = src_pcd.compute_point_cloud_distance(tgt_pcd)
    dists2 = tgt_pcd.compute_point_cloud_distance(src_pcd)
    
    cd = np.mean(dists1) + np.mean(dists2)
    return cd

def get_opt_transform(src_pcd, tgt_pcd, rotation_dict, threshold=5e-3):
    src_center = src_pcd.get_center()
    tgt_center = tgt_pcd.get_center()
    
    src_points = np.asarray(src_pcd.points)
    tgt_points = np.asarray(tgt_pcd.points)
    points_len = min(len(src_points), len(tgt_points))
    points_thre = min(5, int(points_len * 0.01))

    tgt_tree = KDTree(tgt_points)
    
    max_intersection = 0
    out_xform = None 
    for axis_str, angle_arr in rotation_dict.items():
        for angle_rad in angle_arr:
            rotation = Rotation.from_euler(axis_str, angle_rad)
            R_rel = rotation.as_matrix()
            t_rel = tgt_center - R_rel @ src_center
            xform = np.eye(4)
            xform[:3, :3] = R_rel.copy()
            xform[:3, 3] = t_rel 
            
            new_pcd = copy.deepcopy(src_pcd)
            new_pcd = new_pcd.transform(xform)
            translate = get_align_bottom_translate(new_pcd, tgt_pcd)
            new_pcd = new_pcd.translate(translate)

            new_points = np.asarray(new_pcd.points)
            new_tree = KDTree(new_points)
            dist1, _ = tgt_tree.query(new_points)
            mask1 = dist1 < threshold
            count1 = np.sum(mask1)

            dist2, _ = new_tree.query(tgt_points)
            mask2 = dist2 < threshold
            count2 = np.sum(mask2)
            count = (count1 + count2) / 2.0
            # print(axis_str, angle_rad, rotation.as_rotvec(), count)

            if out_xform is None or (count > max_intersection + points_thre):
                max_intersection = count
                out_xform = xform 

    return out_xform

def get_grid(bbox_min, bbox_max, voxel_size):
    if np.isscalar(voxel_size):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.asarray(voxel_size)
    
    grid_shape = np.ceil((bbox_max - bbox_min) / voxel_size).astype(int)
    grid = np.full(grid_shape, -1, dtype=np.int32)
    return grid 

def adjust_to_grid(points, value_ranges, grid_shape):
    x_indices = ((points[:, 0] - value_ranges[0][0]) / 
                (value_ranges[0][1] - value_ranges[0][0]) * (grid_shape[0] - 1)).astype(int)
    y_indices = ((points[:, 1] - value_ranges[1][0]) / 
                (value_ranges[1][1] - value_ranges[1][0]) * (grid_shape[1] - 1)).astype(int)
    z_indices = ((points[:, 2] - value_ranges[2][0]) / 
                (value_ranges[2][1] - value_ranges[2][0]) * (grid_shape[2] - 1)).astype(int)

    x_indices = np.clip(x_indices, 0, grid_shape[0] - 1)
    y_indices = np.clip(y_indices, 0, grid_shape[1] - 1)
    z_indices = np.clip(z_indices, 0, grid_shape[2] - 1)
    return x_indices, y_indices, z_indices

def get_labels_candidates(points, labels, grid_shape, value_ranges):
    unique_labels = np.unique(labels)
    unique_labels_len = unique_labels.max() + 1 #len(unique_labels)
    # print("unique_labels_len", unique_labels_len)

    candidates = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], unique_labels_len), dtype=int)
    
    x_indices, y_indices, z_indices = adjust_to_grid(points, value_ranges, grid_shape)

    # 创建网格索引的线性版本
    linear_indices = (x_indices * grid_shape[1] * grid_shape[2] + 
                     y_indices * grid_shape[2] + 
                     z_indices)

    # 统计每个网格中出现的label
    unique_linear_indices, counts = np.unique(linear_indices, return_counts=True)
    for linear_idx in unique_linear_indices:
        # 找到所有属于当前网格的点
        mask = linear_indices == linear_idx
        grid_labels = labels[mask]
        
        if len(grid_labels) > 0:
            x = linear_idx // (grid_shape[1] * grid_shape[2])
            remainder = linear_idx % (grid_shape[1] * grid_shape[2])
            y = remainder // grid_shape[2]
            z = remainder % grid_shape[2]

            cur_unique, cur_counts = np.unique(grid_labels, return_counts=True)

            candidates[x, y, z, list(cur_unique)] = cur_counts
            
    return candidates

def fill_holes_conv(grid, label, max_iterations=5, min_neighbors=4):
    filled_grid = grid.copy()
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0
    
    changed = True
    iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        mask = (filled_grid == label).astype(np.float32)
        z_coords = np.unique(np.argwhere(mask > 0)[:, 2]) if np.any(mask > 0) else []
        for z in z_coords:
            layer = mask[:, :, z]            
            # 使用卷积计算每个像素的8邻域中标签的数量
            neighbor_count = ndimage.convolve(layer, kernel, mode='constant')
            # 找到空洞像素（当前为0）但周围有足够多标签的像素
            holes_to_fill = (layer == 0) & (neighbor_count >= min_neighbors)
            if np.any(holes_to_fill):
                filled_grid[holes_to_fill, z] = label
                changed = True
        iteration += 1  
    return filled_grid

def fit_floor(candidates, grid, label_mapping_dict):
    gx, gy, gz, label_num = candidates.shape
    half_gx = int(gx / 2)
    half_gy = int(gy / 2)

    label_idx = label_mapping_dict.get('floor', {}).get('sem_idx')
    init_mask = (candidates[..., label_idx] > 0)
    masked_values = candidates[..., label_idx][init_mask]
    thre = int(masked_values.mean() * 0.2)

    candidate_idx_arr = np.argwhere(candidates[..., label_idx] > thre )
    if len(candidate_idx_arr) > 0:
        z_unique, z_counts = np.unique(candidate_idx_arr[..., 2], return_counts=True)
        # print("=== floor z", z_unique, z_counts)
        z_unique_filtered = z_unique[z_unique < int(gz / 2)]

        if len(z_unique_filtered) > 0:
            z_counts_filtered = z_counts[np.isin(z_unique, z_unique_filtered)]
            z_max = z_unique_filtered[z_counts_filtered.argmax()]
        else:
            z_max = z_unique[0]
    else:
        z_max = 0 ### set at the bottom 
        
    mask = np.any(candidates != 0, axis=2)  # (gx, gy, label_num)
    aggregated_labels = [[np.where(mask[x, y])[0] for y in range(gy)] for x in range(gx)]
    has_labels = mask.any(axis=2)
    filled_grid = grid.copy()
    filled_grid[has_labels, z_max] = label_idx
    filled_grid = fill_holes_conv(filled_grid, label_idx)
    floor_plane = filled_grid[:, :, z_max]
    x_indices, y_indices = np.where(floor_plane == label_idx)

    boundary_plane = np.zeros((gx, gy), dtype=bool)
    towards_plane = np.ones((gx, gy, 2), dtype=int) * -1
    if len(x_indices) > 0:
        x_to_y = {x: y_indices[x_indices == x] for x in set(x_indices)}
        y_to_x = {y: x_indices[y_indices == y] for y in set(y_indices)}
        
        for x, y_vals in x_to_y.items():
            y_min = y_vals.min()
            y_max = y_vals.max()
            
            towards_plane[x, y_min: half_gy+1 , 0] = 0
            towards_plane[x, half_gy+1: y_max+1, 0] = 1
            boundary_plane[x, [y_min, y_max]] = True

        for y, x_vals in y_to_x.items():
            x_min = x_vals.min()
            x_max = x_vals.max()
            towards_plane[x_min: half_gx+1, y, 1] = 2
            towards_plane[half_gx+1: x_max+1, y, 1] = 3
            boundary_plane[[x_min, x_max], y] = True
            
    floor_info = {
        "boundary": boundary_plane,
        "towards": towards_plane,
        "height": z_max,
        "label": label_idx,
    }
    return filled_grid, floor_info

def fit_ceiling(candidates, grid, floor_info, label_mapping_dict):
    gx, gy, gz, label_num = candidates.shape
    
    label_idx = label_mapping_dict.get('ceiling', {}).get('sem_idx')
    init_mask = (candidates[..., label_idx] > 0)
    masked_values = candidates[..., label_idx][init_mask]
    thre = int(masked_values.mean() * 0.2)

    candidate_idx_arr = np.argwhere(candidates[..., label_idx] > thre )
    if len(candidate_idx_arr) > 0:
        z_unique, z_counts = np.unique(candidate_idx_arr[..., 2], return_counts=True)
        z_unique_filtered = z_unique[z_unique > int(gz / 2)]

        if len(z_unique_filtered) > 0:
            z_counts_filtered = z_counts[np.isin(z_unique, z_unique_filtered)]
            z_max = z_unique_filtered[z_counts_filtered.argmax()]
        else:
            z_max = z_unique[0]

        #### considering mc block setting rules, the ceiling needs to be raised by one block to avoid compressing the interior space. 
        z_max = min(z_max + 1, gz - 1)
    else:
        z_max = gz - 1 ### set at the top 
        
    floor_idx = floor_info["label"]
    floor_height = int(floor_info["height"])
    filled_grid = grid.copy()
    floor_plane = filled_grid[:, :, floor_height].copy()
    mask = (floor_plane == floor_idx )
    filled_grid[mask, z_max] = label_idx

    ceiling_info = {
        "height": z_max,
        "label": label_idx
    }
    return filled_grid, ceiling_info

def calc_continuous_block(arr, axis=0, gap_thre=3):
    sorted_arr = np.sort(arr[:, axis])
    delta = np.diff(sorted_arr, prepend=sorted_arr[0] - 2)  
    breaks = np.where(delta != 1)[0]
    blocks = np.split(sorted_arr, breaks)
    blocks = [x for x in blocks if len(x) > 0]
    
    out = []
    bi = 0
    for block in blocks:
        bi += 1
        if len(block) <= gap_thre:
            continue    
        out.append(block)
        break 
    if len(out) < 1:
        return None

    for cur in blocks[bi:]:
        last = out[-1]
        gap = abs(cur[0] - last[-1])  
        if gap < gap_thre:                     
            fill = np.arange(last[-1] + 1, cur[0]) 
            out[-1] = np.concatenate([last, fill, cur])
        else:
            if len(cur) <= gap_thre:
                # out.append([cur[0]])
                continue 
            else:
                out.append(cur)

    return out

def find_the_reasonable_wall_arr(coords, half_val):
    unique_vals, unique_cnts = np.unique(coords, return_counts=True)
    sort_idx = np.argsort(unique_cnts)[::-1] 

    wall_boundary_max_cnts = [0, 0]
    wall_boundary = [-1, -1]
    for i in sort_idx:
        idx = sort_idx[i]
        wall_idx = unique_vals[idx]
        wall_cnts = unique_cnts[idx]
        if wall_idx >= half_val:
            ### compare wall_boundary[1]
            if wall_boundary[1] < 0:
                wall_boundary[1] = wall_idx
                wall_boundary_max_cnts[1] = wall_cnts  
            elif (wall_idx > wall_boundary[1]) and (wall_cnts >= int(wall_boundary_max_cnts[1] * 0.8)):
                wall_boundary[1] = wall_idx 
        else:
            ### compare wall_boundary[0]
            if wall_boundary[0] < 0:
                wall_boundary[0] = wall_idx 
                wall_boundary_max_cnts[0] = wall_cnts 
            elif (wall_idx < wall_boundary[0]) and (wall_cnts >= int(wall_boundary_max_cnts[0] * 0.8)):
                wall_boundary[0] = wall_idx 

    out_arr = []
    for val in wall_boundary:
        if val > -1:
            out_arr.append(val)

    for idx in sort_idx:
        if unique_vals[idx] not in out_arr:
            out_arr.append(unique_vals[idx])

    return out_arr

def get_wall_xy_view(seeds, grid_shape, merge_dist=1, overlap_ratio=0.5, ref_dict=None):
    gx, gy, gz = grid_shape 
    orig_xy_plane = np.zeros((gx, gy), dtype=bool)
    col_num_thre = 2 * merge_dist + 2
    
    cur_xy_coords = seeds[:, :2].copy()
    cur_xy_coords = np.unique(cur_xy_coords, axis=0)
    orig_xy_plane[tuple(cur_xy_coords.T)] = True 

    #### [[fix x, search, y], [fix y, search x ]]
    fix_search_axis_arr = [[0, 1], [1, 0]]

    fill_plane = np.zeros((gx, gy), dtype=bool)
    axis_plane = np.ones((gx, gy), dtype=int) * -1
    boundary_plane = np.zeros((gx, gy), dtype=bool)
    out_ref_dict = {}
    for [fix_axis, search_axis] in fix_search_axis_arr:
        if fix_axis == 1:
            n_fix = gy
            n_search = gx
        else:
            n_fix = gx
            n_search = gy
        n_half_fix = int(n_fix // 2)

        out_ref_dict[f"{fix_axis}_{search_axis}"] = []
        xy_dict = {}

        #### use the most outerbound + reasonable points' total
        unique_vals_sorted = find_the_reasonable_wall_arr(cur_xy_coords[:, fix_axis], n_half_fix)
        
        #### calculate continuous block first 
        for _val in unique_vals_sorted:
            cur_search_vals = cur_xy_coords[cur_xy_coords[:, fix_axis] == _val]
            valid_block_arr = calc_continuous_block(cur_search_vals, axis=search_axis, gap_thre=col_num_thre)
            if valid_block_arr is not None:
                xy_dict[_val] = valid_block_arr.copy()
                
        sorted_keys = sorted(xy_dict, key=lambda k: len(np.concatenate(xy_dict[k])), reverse=True)
        if ref_dict is not None:
            ref_list = ref_dict[f"{fix_axis}_{search_axis}"]
            swap_idx = 0
            for ref_val in ref_list:
                try:
                    ref_idx = sorted_keys.index(ref_val)
                    if ref_idx == swap_idx:
                        continue 

                    swap_val_arr = sorted_keys[swap_idx:ref_idx]
                    sorted_keys[swap_idx] = ref_val 
                    sorted_keys[swap_idx+1:ref_idx+1] = swap_val_arr
                    swap_idx += 1
                except ValueError:
                    print(f"==== ref_val {ref_val} not exist")
                    continue 

        boundary = np.ones((2, n_search), dtype=int) * -1
        visited = np.zeros(n_fix, dtype=bool)
        for _i, _val in enumerate(sorted_keys):
            if visited[_val]:
                continue

            cur_blocks_arr = xy_dict[_val].copy()
            cur_blocks_len = len(cur_blocks_arr)
            visited[_val] = True 

            if _val > int(n_fix * 0.3) and _val < int(n_fix * 0.7):
                print(f"skip {_val}, the wall block is too close to the scene's center", flush=True)
                continue 
            
            cur_value = np.unique(np.concatenate(cur_blocks_arr))
            final_block_idx = [cur_value]
        
            visited_val = []
            for next_val in range(_val + 1, gx):
                if next_val not in xy_dict:
                    break  

                if visited[next_val]:
                    break 

                next_blocks_arr = xy_dict[next_val].copy()
                next_block_len = len(next_blocks_arr)
                if next_block_len < 1:
                    break  
                
                visited_val.append(next_val)

                tmp_idx_arr = []
                next_overlap = 0
                ci = 0
                ni = 0
                while ci < cur_blocks_len and ni < next_block_len:
                    cur_block_idx = cur_blocks_arr[ci]
                    next_block_idx = next_blocks_arr[ni]
                    cblock_len = len(cur_block_idx)
                    nblock_len = len(next_block_idx)
                    min_len = min(cblock_len, nblock_len)
                    # max_len = max(cblock_len, nblock_len)

                    intersect = np.intersect1d(cur_block_idx, next_block_idx)
                    if intersect.size / min_len >= overlap_ratio:
                        union = np.union1d(cur_block_idx, next_block_idx)
                        tmp_idx_arr.append(union)
                        next_overlap += 1
                        ci += 1
                        ni += 1
                    elif cblock_len >= nblock_len:
                        ci += 1
                    else:
                        ni += 1
                
                if next_overlap == next_block_len:
                    visited[next_val] = True
                    final_block_idx.append(np.concatenate(tmp_idx_arr)) 
                else:
                    break  

            for prev_val in range(_val - 1, -1, -1):
                if prev_val not in xy_dict:
                    break  
                
                if visited[prev_val]:
                    break 

                prev_blocks_arr = xy_dict[prev_val].copy()
                prev_block_len = len(prev_blocks_arr)
                if prev_block_len < 1 or prev_block_len > cur_blocks_len:
                    break  
                
                visited_val.append(prev_val)
                tmp_idx_arr = []
                prev_overlap = 0
                ci = 0
                pi = 0
                while ci < cur_blocks_len and pi < prev_block_len:
                    cur_block_idx = cur_blocks_arr[ci]
                    prev_block_idx = prev_blocks_arr[pi]
                    cblock_len = len(cur_block_idx)
                    pblock_len = len(prev_block_idx)
                    min_len = min(cblock_len, pblock_len)
                    # max_len = max(cblock_len, pblock_len)

                    intersect = np.intersect1d(cur_block_idx, prev_block_idx)
                    if intersect.size / min_len >= overlap_ratio:
                        union = np.union1d(cur_block_idx, prev_block_idx)
                        tmp_idx_arr.append(union)
                        prev_overlap += 1
                        ci += 1
                        pi += 1
                    elif cblock_len >= pblock_len:
                        ci += 1
                    else:
                        pi += 1

                if prev_overlap == prev_block_len:
                    visited[prev_val] = True
                    final_block_idx.append(np.concatenate(tmp_idx_arr)) 
                else:
                    break  

            if len(final_block_idx) > 0:
                out_ref_dict[f"{fix_axis}_{search_axis}"].append(_val)

                final_block_idx = np.unique(np.concatenate(final_block_idx))
                if fix_axis == 0:
                    fill_plane[_val, final_block_idx] = True 
                    axis_plane[_val, final_block_idx] = fix_axis
                else:
                    fill_plane[final_block_idx, _val] = True
                    axis_plane[final_block_idx, _val] = fix_axis

                
                bound_idx = 0 if _val < n_half_fix else 1
                for _idx in final_block_idx:
                    if boundary[bound_idx][_idx] < 0:
                        boundary[bound_idx][_idx] = _val 
                        if fix_axis == 0:
                            boundary_plane[_val, _idx] = True 
                        else:
                            boundary_plane[_idx, _val] = True 
            
            # print("=== visited", _val, len(final_block_idx), visited_val, flush=True)

    out_dict = {
        "axis": axis_plane,
        "plane": fill_plane,
        "boudary": boundary_plane,
        "ref": out_ref_dict,
    }
    return out_dict

def update_boundary(floor_boundary, wall_boundary):
    wall_x_coords, wall_y_coords = np.where(wall_boundary)
    if len(wall_x_coords) < 1:
        return floor_boundary

    boundary_plane = np.zeros_like(floor_boundary)

    x_to_y = dict()
    y_to_x = dict()
    for x, y in zip(wall_x_coords, wall_y_coords):
        if x not in x_to_y:
            x_to_y[x] = []
        if y not in y_to_x:
            y_to_x[y] = []
        x_to_y[x].append(y)
        y_to_x[y].append(x)
    
    for x, y_vals in x_to_y.items():
        y_len = len(y_vals)
        floor_y_at_x = np.where(floor_boundary[x, :])[0]   
        if y_len == 1:
            boundary_plane[x, y_vals[0]] = True
            y_min = min(floor_y_at_x)
            y_max = max(floor_y_at_x)
            #### fill other side
            if y_vals[0] - y_min <= y_max - y_vals[0]:
                boundary_plane[x, y_max] = True
            else:
                boundary_plane[x, y_min] = True
        elif y_len == 2:
            boundary_plane[x, y_vals] = True
        elif y_len > 2:
            boundary_plane[x, floor_y_at_x] = True

    for y, x_vals in y_to_x.items():
        x_len = len(x_vals)
        floor_x_at_y = np.where(floor_boundary[:, y])[0]

        if x_len == 1:
            boundary_plane[x_vals[0], y] = True
            x_min = min(floor_x_at_y)
            x_max = max(floor_x_at_y)
            #### fill other side
            if x_vals[0] - x_min <= x_max - x_vals[0]:
                boundary_plane[x_max, y] = True
            else:
                boundary_plane[x_min, y] = True
        elif x_len == 2:
            boundary_plane[x_vals, y] = True
        elif x_len > 2:
            boundary_plane[floor_x_at_y, y] = True
 
    return boundary_plane

def fit_wall(candidates, grid, floor_info, ceiling_info, label_mapping_dict):
    gx, gy, gz, label_num = candidates.shape
    floor_height = floor_info["height"]
    floor_boundary = floor_info["boundary"]
    floor_towards = floor_info["towards"]
    ceiling_height = ceiling_info["height"]
    
    wall_idx = label_mapping_dict.get('wall', {}).get('sem_idx')
    init_mask = (candidates[..., wall_idx] > 0)
    masked_values = candidates[..., wall_idx][init_mask]
    thre = int(masked_values.mean() * 0.1)
    mask = (candidates[..., wall_idx] > thre)
    #### add window + door + curtain
    window_idx = label_mapping_dict.get('window', {}).get('sem_idx')
    if window_idx is not None and window_idx < label_num:
        mask |= (candidates[..., window_idx] > 0)
    
    door_idx = label_mapping_dict.get('door', {}).get('sem_idx')
    if door_idx is not None and door_idx < label_num:
        mask |= (candidates[..., door_idx] > 0)
    
    curtain_idx = label_mapping_dict.get('curtain', {}).get('sem_idx')
    if curtain_idx is not None and curtain_idx < label_num:
        mask |= (candidates[..., curtain_idx] > 0)
    
    candidate_idx_arr = np.argwhere(mask)
    is_cand = np.zeros((gx, gy, gz), dtype=bool)
    is_cand[tuple(candidate_idx_arr.T)] = True

    mask = (grid < 0) & is_cand
    seeds = np.argwhere(mask) 
    seeds[seeds[:, 2] < floor_height, 2] = floor_height
    seeds[seeds[:, 2] > ceiling_height, 2] = ceiling_height
    seeds = np.unique(seeds, axis=0)

    valid_mask = grid[seeds[:, 0], seeds[:, 1], seeds[:, 2]] < 0
    seeds = seeds[valid_mask]
    if len(seeds) == 0:
        return grid, None, None

    ### find wall height
    unique_z = np.unique(seeds[:, 2])
    z_len = len(unique_z)
    z_hf_idx = z_len // 2

    bottom_mask = seeds[..., 2] <= max(floor_height, unique_z[max(0, z_hf_idx - 1)])
    bottom_seeds = seeds[bottom_mask]
    bottom_planes, bottom_cnts = np.unique(bottom_seeds[:, 2], return_counts=True)
    wall_bottom = floor_height + 1

    bottom_xy_dict = get_wall_xy_view(bottom_seeds, grid.shape, merge_dist=1, overlap_ratio=0.5)
    bottom_xy_view = bottom_xy_dict["plane"].copy()
    # cv2.imwrite("./check_bottom.png", bottom_xy_dict["boudary"] * 255)

    top_mask = seeds[..., 2] >= min(ceiling_height, unique_z[min(z_hf_idx + 1, z_len - 1)])
    top_seeds = seeds[top_mask]
    top_planes, top_cnts = np.unique(top_seeds[:, 2], return_counts=True)
    wall_top = ceiling_height - 1

    #### add ref_dict to avoid double walls
    top_xy_dict= get_wall_xy_view(top_seeds, grid.shape, merge_dist=1, overlap_ratio=0.5, ref_dict=bottom_xy_dict["ref"])
    top_xy_view = top_xy_dict["plane"].copy()
    # cv2.imwrite("./check_top.png", top_xy_dict["boudary"] * 255)
    
    wall_boundary = top_xy_dict["boudary"] | bottom_xy_dict["boudary"]
    # cv2.imwrite("./check_full.png", wall_boundary * 255)

    boundary_plane = update_boundary(floor_boundary, wall_boundary)
    # cv2.imwrite('./check_boundary.png', boundary_plane * 255)

    wall_idx_arr = []
    for vi in range(gx):
        for vj in range(gy):
            if boundary_plane[vi][vj] or (top_xy_view[vi][vj] and bottom_xy_view[vi][vj]):
                axis_val = top_xy_dict["axis"][vi][vj] if top_xy_dict["axis"][vi][vj] >= 0 else bottom_xy_dict["axis"][vi][vj] 
              
                for vz in range(wall_bottom, wall_top + 1):
                    wall_idx_arr.append([vi, vj, vz, axis_val])
                

    wall_idx_arr = np.stack(wall_idx_arr, axis=0)
    seeds = wall_idx_arr[:, :3].copy()
    wall_axis = {}
    for wall_info in wall_idx_arr:
        ### no need to use z value
        x, y, _, axis = wall_info
        key = "%d_%d" % (x, y)
        value = floor_towards[x][y][axis]
        wall_axis[key] = value 

    valid_mask = grid[seeds[:, 0], seeds[:, 1], seeds[:, 2]] < 0
    seeds = seeds[valid_mask]
    if len(seeds) == 0:
        return grid, None

    wall_pos_xy = np.unique(seeds[:, :2], axis=0)
    grid[seeds[:, 0], seeds[:, 1], seeds[:, 2]] = wall_idx 

    wall_info = {
        "wall_pos": wall_pos_xy,
        "wall_axis": wall_axis,
        "wall_top": wall_top,
        "wall_bottom": wall_bottom,
    }
    return grid, wall_info

def update_idx_with_wall(pos, wall_pos):
    px, py, pz = pos 
    
    l2_dist = np.linalg.norm(wall_pos - [px, py], axis=-1)
    sort_x_idx = np.argsort(l2_dist)
    new_px, new_py = wall_pos[sort_x_idx[0]]
    return [new_px, new_py, pz]       

def fit_window(candidates, grid, label, wall_pos, wall_axis, floor_height, ceiling_height):
    gx, gy, gz, _ = candidates.shape

    init_mask = (candidates[..., label] > 0)
    masked_values = candidates[..., label][init_mask]
    thre = int(masked_values.mean() * 0.1)

    mask = (candidates[..., label] > thre)
    candidate_idx_arr = np.argwhere(mask)
    out_arr = []

    for cand_idx in candidate_idx_arr:
        pos_idx = update_idx_with_wall(cand_idx, wall_pos)
        if pos_idx[2] <= floor_height:
            pos_idx[2] = floor_height + 1
        elif pos_idx[2] >= ceiling_height:
            pos_idx[2] = ceiling_height - 1

        axis = wall_axis[f"{pos_idx[0]}_{pos_idx[1]}"]
        new_idx = pos_idx + [label, axis]
        out_arr.append(new_idx)

        grid[tuple(pos_idx)] = label
    return out_arr

def fit_door(candidates, grid, label, wall_pos, wall_axis, floor_height):
    door_size = (2, 4) ### width, height
    gx, gy, gz, _ = candidates.shape

    init_mask = (candidates[..., label] > 0)
    masked_values = candidates[..., label][init_mask]
    thre = int(masked_values.mean() * 0.1)
    
    mask = (candidates[..., label] > thre)
    labeled_mask, num_feat = ndimage.label(mask, structure=np.ones((3,3,3)))
    # print("=== door, num_feat", num_feat, flush=True)

    indices = np.argwhere(labeled_mask > 0)          
    labels  = labeled_mask[labeled_mask > 0]         

    insts_idx_arr = [indices[labels == k] for k in range(1, num_feat + 1)]
    center_arr = []
    for i, idx_arr in enumerate(insts_idx_arr):
        z_min = idx_arr[:, 2].min()
        idx_arr[:, 2] = floor_height + 1
        idx_arr = np.unique(idx_arr, axis=0)
        #### x最大，y最小
        best_idx = np.lexsort((idx_arr[:, 1], -idx_arr[:, 0]))[0]
        
        center = update_idx_with_wall(idx_arr[best_idx], wall_pos)
        axis = wall_axis[f"{center[0]}_{center[1]}"]
        if axis in [0, 1]:
            width_axis = 1
        else:
            width_axis = 0
        other_axis = (width_axis + 1) % 2

        width_start = center[width_axis]
        z_start = min(floor_height + 1, center[2])

        new_center = np.zeros(5, dtype=int)
        new_center[width_axis] = width_start
        new_center[other_axis] = center[other_axis]
        new_center[2] = z_start
        new_center[3] = label
        new_center[4] = axis
        center_arr.append(new_center)
        
        for width_val in range(width_start - door_size[0] + 1, width_start + 1):
            new_pos = np.zeros(3, dtype=int)
            new_pos[width_axis] = width_val
            new_pos[other_axis] = center[other_axis]
            for new_z in range(z_start, z_start + door_size[1]):
                new_pos[2] = new_z 
                grid[tuple(new_pos)] = label

    return center_arr

def assign_labels_to_grid(candidates, grid, label_mapping_dict):
    gx, gy, gz, label_len = candidates.shape
    counts = np.array([np.count_nonzero(candidates[..., label]) for label in range(label_len)], dtype=int)
    sort_labels = np.argsort(counts)
    
    print("=== fit floor", flush=True)
    floor_idx = label_mapping_dict.get('floor', {}).get('sem_idx')
    grid, floor_info = fit_floor(candidates, grid, label_mapping_dict)
    floor_height = floor_info["height"]
    print("floor_height", floor_height)

    print("=== fit ceiling", flush=True)
    ceiling_idx = label_mapping_dict.get('ceiling', {}).get('sem_idx')
    grid, ceiling_info = fit_ceiling(candidates, grid, floor_info, label_mapping_dict)
    ceiling_height = ceiling_info["height"]
    print("ceiling_height", ceiling_height)

    print("=== fit wall", flush=True)
    wall_idx = label_mapping_dict.get('wall', {}).get('sem_idx')
    grid, wall_info = fit_wall(candidates, grid, floor_info, ceiling_info, label_mapping_dict)
    wall_axis = wall_info["wall_axis"]
    wall_pos_xy = wall_info["wall_pos"]

    window_arr = None 
    window_idx = label_mapping_dict.get('window', {}).get('sem_idx')
    if window_idx is not None and counts[window_idx] > 0:
        print("=== fit window", flush=True)
        window_arr = fit_window(candidates, grid, window_idx, wall_pos_xy, wall_axis, floor_height, ceiling_height)
    
    door_arr = None 
    door_idx = label_mapping_dict.get('door', {}).get('sem_idx')
    if door_idx is not None and counts[door_idx] > 0:
        print("=== fit door", flush=True)
        door_arr = fit_door(candidates, grid, door_idx, wall_pos_xy, wall_axis, floor_height)
    
    basic_idx_arr = [floor_idx, ceiling_idx, wall_idx, window_idx, door_idx]
    out_dict = {
        "grid": grid,
        "floor_info": floor_info,
        "ceiling_info": ceiling_info,
        "wall_info": wall_info, 
        "window_arr": window_arr,
        "door_arr": door_arr,
        "basic_label": basic_idx_arr,
    }
    return out_dict

def get_points_on_grid(points, value_ranges, grid_shape):
    x_indices, y_indices, z_indices = adjust_to_grid(points, value_ranges, grid_shape)
    linear_indices = (x_indices * grid_shape[1] * grid_shape[2] + 
                     y_indices * grid_shape[2] + 
                     z_indices)
    linear_indices = np.ravel_multi_index(
        (x_indices, y_indices, z_indices),    
        grid_shape                           
    )
    unique_linear_indices = np.unique(linear_indices)
    xyz = np.stack(np.unravel_index(unique_linear_indices, grid_shape), axis=1)
    return xyz 

def get_bottom_info(obj_points, dtype=np.float32):
    z_min = obj_points.min(0)[2]
    obj_center = np.mean(obj_points, axis=0)
    obj_center[2] = z_min 

    obj_xy_pos = np.unique(obj_points[:, :2], axis=0)
    return obj_center.astype(dtype), obj_xy_pos

def adjust_obj_pos(grid_dict, points, size):
    grid = grid_dict["grid"]
    value_ranges = grid_dict["value_ranges"]
    floor_height = grid_dict["floor_info"]["height"]
    ceiling_height = grid_dict["ceiling_info"]["height"]
    wall_pos = grid_dict["wall_info"]["wall_pos"]
    wall_axis = grid_dict["wall_info"]["wall_axis"]
    
    grid_points = get_points_on_grid(points, value_ranges, grid.shape)

    #### check whether obj has intersect with wall
    wall_set = set(map(tuple, wall_pos.astype(int)))
    obj_points = grid_points[:, :2].copy()
    obj_set = set(map(tuple, obj_points.astype(int)))
    intersection = wall_set & obj_set
    if intersection:
        intersect_coords = np.array(list(intersection))
        axis_dict = {}
        for coord in intersect_coords:
            axis = wall_axis["%d_%d" % tuple(coord)]
            if axis < 2:
                intersect_axis = 0 ### x is fixed, search y
            else:
                intersect_axis = 1 #### y is fixed, search x
            if intersect_axis not in axis_dict:
                axis_dict[intersect_axis] = []
            axis_dict[intersect_axis].append(coord)

        offset = [0, 0]
        for axis, coord_arr in axis_dict.items():
            max_val = max(coord[axis] for coord in coord_arr)
            half_axis = int(grid.shape[axis] / 2)
            if max_val < half_axis:
                offset[axis] += max(1, int(np.ceil(size[axis]) / 2))
            else:
                offset[axis] -= max(1, int(np.ceil(size[axis]) / 2))
        obj_points += offset 
    
    grid_points[:, :2] = obj_points.copy()
    obj_center, _ = get_bottom_info(grid_points, dtype=int)
    if obj_center[2] < floor_height + 1:
        offset = floor_height + 1 - obj_center[2]
        obj_center[2] += offset 
        grid_points[2] += offset 
    elif obj_center[2] > ceiling_height - 1:
        offset = ceiling_height - 1 - obj_center[2]
        obj_center[2] += offset 
        grid_points[2] += offset 

    return obj_center, grid_points

def get_global_xform(pcd, bottom_center, grid_shape):
    RT = np.zeros((4, 4))
    RT[0][0] = -1
    RT[1][2] = 1
    RT[2][1] = 1
    RT[3][3] = 1
    RT[0][3] = grid_shape[0] - 1
    new_center = (RT @ np.append(bottom_center, 1))[:3]
    return RT, new_center

def get_vertical_relation(grid_dict, all_points, sem_label_names, instances_info_dict):
    grid = grid_dict["grid"]
    value_ranges = grid_dict["value_ranges"]

    pos_dict = {}
    for inst_key in sem_label_names:
        if inst_key not in instances_info_dict:
            continue 

        inst_arr = instances_info_dict[inst_key]
        for i, inst_info in enumerate(inst_arr):
            inst_name = f"{inst_key}_{i:02d}"
            
            ###### load segmented point clouds
            inst_point_idx_arr = inst_info["point_idx"]
            tgt_points = np.array(all_points[inst_point_idx_arr]).astype(np.float32)
            tgt_grid_points = get_points_on_grid(tgt_points, value_ranges, grid.shape)
            tgt_pos, xy_grid = get_bottom_info(tgt_grid_points, dtype=int)
            pos_dict[inst_name] = {
                "tgt_pos": tgt_pos,
                "xy_grid": xy_grid,
                "inst_key": inst_key,
                "idx": i
            }
    
    ### sorted by z value 
    sorted_keys = sorted(pos_dict.keys(), key=lambda k: pos_dict[k]["tgt_pos"][2])
    out_dict = {}
    prev_z = 0
    for i, key in enumerate(sorted_keys):
        val = pos_dict[key]
        tgt_pos = val["tgt_pos"]
        xy_grid = val["xy_grid"]
        cur_set = set(map(tuple, xy_grid.astype(int)))

        search_valid = False 
        for search_key, search_val_arr in out_dict.items():
            for search_val in search_val_arr:
                search_set = set(map(tuple, search_val["xy_grid"].astype(int)))
                intersection = cur_set & search_set
                ### assume the object above has smaller size
                if intersection and len(intersection) == len(cur_set):
                    out_dict[search_key].append(val)
                    search_valid = True
                    break 

        if not search_valid:
            out_dict[i] = [val]

    return out_dict 

def fit_obj_mesh(glb_dir, out_dir, grid_dict, all_points, sem_label_names, instances_info_dict, global_pos):
    
    grid = grid_dict["grid"]
    gx, gy, gz = grid.shape
    value_ranges = grid_dict["value_ranges"]
    voxel_size = grid_dict["voxel_size"]
    wall_pos = grid_dict["wall_info"]["wall_pos"]
    floor_height = grid_dict["floor_info"]["height"]
    floor_towards = grid_dict["floor_info"]["towards"]
    
    #### get vertical relationship
    xy_dict = get_vertical_relation(grid_dict, all_points, sem_label_names, instances_info_dict)

    select_axis = 2 
    out_dict = {}

    for _, vert_instances_arr in xy_dict.items():
        prev_tgt_pos = None
        prev_src_pos = None
        for vert_instance in vert_instances_arr:
            inst_key = vert_instance["inst_key"]
            i = vert_instance["idx"]
            inst_name = f"{inst_key}_{i:02d}"
            tgt_pos = vert_instance["tgt_pos"]
        
            best_id = 0 
            best_point_idx_arr = instances_info_dict[inst_key][best_id]["point_idx"]
            best_pcd = o3d.geometry.PointCloud()
            best_pcd.points = o3d.utility.Vector3dVector(all_points[best_point_idx_arr])

            inst_info = instances_info_dict[inst_key][i]
            ###### load segmented point clouds
            inst_point_idx_arr = inst_info["point_idx"]
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(all_points[inst_point_idx_arr])
            tgt_obb = tgt_pcd.get_minimal_oriented_bounding_box()

            ###### load generated sam3d mesh
            glb_path = os.path.join(glb_dir, f"{inst_key}_{best_id:02d}.glb")
            if not os.path.exists(glb_path):
                print(f"=== no {glb_path}", flush=True)
                continue 

            src_mesh = trimesh.load(glb_path, force="mesh")
            src_verts = np.array(src_mesh.vertices).astype(np.float32)
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(src_verts)
            src_obb = src_pcd.get_minimal_oriented_bounding_box()
            
            #### align the mesh to the point clouds' coordinates 
            init_xform = get_src_init_xform(src_pcd, tgt_pcd)
            src_pcd = src_pcd.transform(init_xform)

            scale_xform, inst_scale_axis = get_src_scale_xform(src_pcd, best_pcd, select_axis)
            src_pcd = src_pcd.transform(scale_xform)
            
            ### chair can be randomly placed
            if inst_key in ["chair"]:
                rotation_arr = [0, -np.pi/2, np.pi/2, np.pi]
            else:
                rotation_arr = []
                for towards_val in floor_towards[tgt_pos[0], tgt_pos[1]]:
                    if towards_val == 0:
                        rotation_arr.append(np.pi)
                    elif towards_val == 1:
                        rotation_arr.append(0)
                    elif towards_val == 2:
                        rotation_arr.append(np.pi / 2)
                    elif towards_val == 3:
                        rotation_arr.append(-np.pi / 2)
                    else: ### if other values, then use 4 directions
                        rotation_arr = [0, -np.pi/2, np.pi/2, np.pi]
                        break 

            rotation_dict = {"z": rotation_arr}
            xform_opt = get_opt_transform(src_pcd, tgt_pcd, rotation_dict)
            src_pcd = src_pcd.transform(xform_opt)

            src_points = np.array(src_pcd.points).astype(np.float32)
            
            #### adjust obj's size to match grid's size
            final_scale_xform = get_final_scale_xform(src_pcd, voxel_size, grid.shape)            
            src_pcd = src_pcd.transform(final_scale_xform)

            src_size = src_pcd.get_max_bound() - src_pcd.get_min_bound() + 1
            ### to avoid wall intersection
            obj_pos, _ = adjust_obj_pos(grid_dict, src_points, src_size)
            #### to ensure the relative z distance is correct
            if prev_src_pos is None:
                prev_src_pos = obj_pos 
                prev_tgt_pos = tgt_pos
            else:
                obj_pos[2] = prev_src_pos[2] + tgt_pos[2] - prev_tgt_pos[2]
                prev_src_pos = obj_pos
                prev_tgt_pos = tgt_pos
            
            src_bottom_center = get_bottom_center(src_pcd)
            src_pcd = src_pcd.translate(obj_pos - src_bottom_center)
            
            #### minecraft world is y-up axis
            xform_global, obj_pos = get_global_xform(src_pcd, obj_pos, grid.shape)
            src_pcd = src_pcd.transform(xform_global)
            new_size = src_pcd.get_max_bound() - src_pcd.get_min_bound() + 1

            new_inst_key = inst_key.replace(' ', '_')
            key = f"{new_inst_key}_{i:02d}"
            out_dict[key] = {
                "pos": [int(a + b) for a, b in zip(obj_pos, global_pos)],
                "size": list(map(int, new_size))
            }
            print("=== fit", key, out_dict[key], flush=True)
            
            src_mesh.vertices = np.array(src_pcd.points).astype(np.float32)
            out_path = os.path.join(out_dir, f"{key}.glb")
            src_mesh.export(out_path, file_type="glb")
         
    return out_dict

def build_basic_block(grid_dict, label_mapping_dict, global_pos):
    grid = grid_dict["grid"]
    gx, gy, gz = grid.shape

    floor_idx = label_mapping_dict.get('floor', {}).get('sem_idx')
    ceiling_idx = label_mapping_dict.get('ceiling', {}).get('sem_idx')
    wall_idx = label_mapping_dict.get('wall', {}).get('sem_idx')
    
    valid_out = []
    for label_name in ["floor", "ceiling", "wall"]:
        label_idx = label_mapping_dict.get(label_name, {}).get('sem_idx')
        if label_idx is None:
            continue 

        mask = (grid == label_idx)
        idx_arr = np.argwhere(mask)
        if len(idx_arr) < 1:
            continue

        mc_idx = label_mapping_dict.get(label_name, {}).get('mc_idx')
        label_arr = np.ones_like(grid[mask][:, None] ) * mc_idx
        axis_arr = np.ones_like(label_arr) * -1
        valid_arr = np.concatenate([idx_arr, label_arr, axis_arr], axis=-1)
        valid_out.append(valid_arr)

    if grid_dict["window_arr"] is not None:
        new_arr = []
        for val in grid_dict["window_arr"]:
            new_label = label_mapping_dict.get("window", {}).get('mc_idx')
            new_val = copy.deepcopy(val)
            new_val[3] = new_label
            new_arr.append(new_val)
        valid_out.append(new_arr)
            
    if grid_dict["door_arr"] is not None:
        new_arr = []
        for val in grid_dict["door_arr"]:
            new_label = label_mapping_dict.get("door", {}).get('mc_idx')
            new_val = copy.deepcopy(val)
            new_val[3] = new_label
            new_arr.append(new_val)
        valid_out.append(new_arr)
            
    valid_out = np.concatenate(valid_out, axis=0).astype(int)
    ### switch to minecraft coordinates
    valid_out[:, 0] = gx - 1 - valid_out[:, 0] + global_pos[0]
    tmp_val = valid_out[:, 1].copy()
    valid_out[:, 1] = valid_out[:, 2].copy() + global_pos[1]
    valid_out[:, 2] = tmp_val + global_pos[2]
    return valid_out 

def get_mc_instruction(mc_file, basic_mc_data, mesh_pos_dict, mc_out_dir):
    out_path = os.path.join(mc_out_dir, "others.mcfunction")
    with open(out_path, "w", encoding="utf-8") as f:
        for key, val in mesh_pos_dict.items():
            if "target" in val:
                mod_name = val["target"]
            else:
                mod_name = key
            cmd = f"setblock {val['pos'][0]} {val['pos'][1]} {val['pos'][2]} examplemod:{mod_name}\n"            
            f.write(cmd)

    cls_mc = []
    with open(mc_file, 'r') as fin:
        block_dict = json.load(fin)
        cls_real = sorted(list(block_dict.keys()))

        for cls_name in cls_real:
            val_arr = block_dict[cls_name]
            block_names = val_arr[1]
            if len(block_names) > 0:
                cls_mc.append(block_names[0])
            else:
                cls_mc.append("minecraft:grass_block")

    cls_len = len(cls_mc)
    cls_inst = dict()
    for block_info in basic_mc_data:
        bx, by, bz, label, axis = block_info 
        if label >= cls_len:
            continue

        block_type = cls_mc[label]
        cmd = f"setblock {bx} {by} {bz} {block_type}"
        if axis == 0:
            cmd += "[facing=east]"
        elif axis == 1:
            cmd += "[facing=west]"
        elif axis == 2:
            cmd += "[facing=north]"
        elif axis == 3:
            cmd += "[facing=south]"
        
        label_name = cls_real[label]
        if label_name not in cls_inst:
            cls_inst[label_name] = []
        cls_inst[label_name].append(cmd)

    category_order = cls_real 
    first_construct = ['floor', 'ceiling', 'wall', 'window']

    # 生成主函数文件，使用链式schedule调用
    main_function_path = os.path.join(mc_out_dir, "room.mcfunction")
    with open(main_function_path, "w", encoding="utf-8") as f:
        f.write("# Start to Build Scene\n")
        f.write("say Start to Build Scene...\n")
        
        # 立即执行第一个类别，然后安排后续类别
        f.write(f"function namespace:{first_construct[0]}\n")
        delay_seconds = 1
        for i in range(1, len(first_construct)):
            category = first_construct[i]
            if category in cls_inst and cls_inst[category]:
                f.write(f"schedule function namespace:{category} {delay_seconds}s\n")
                delay_seconds += 1
        
        # 为后续类别安排延迟执行
        for i in range(1, len(category_order)):
            category = category_order[i]
            if category in first_construct:
                continue 
            if category in cls_inst and cls_inst[category]:
                f.write(f"schedule function namespace:{category} {delay_seconds}s\n")
                delay_seconds += 1

        f.write(f"schedule function namespace:others {delay_seconds}s\n")

    # 为每个类别生成单独的函数文件
    for category, commands in cls_inst.items():
        # print(category, len(commands))
        if commands:
            category_function_path = os.path.join(mc_out_dir, f"{category}.mcfunction")
            with open(category_function_path, "w", encoding="utf-8") as f:
                f.write(f"# Excecute {category} command \n")
                f.write(f"say Start Building{category}...\n")
                f.write("\n".join(commands))
                f.write(f"\nsay {category} Finish\n")

    print("Finished mcfunction files generation", flush=True)

def get_category_sem_idx(names_arr, name):
    try:
        idx = names_arr.index(name)
    except:
        idx = None 
    return idx 

def get_label_info(label_file, mc_file):
    mapping_dict = {}

    with open(label_file, 'r') as fin:
        orig_cls_dict = json.load(fin)
    
    with open(mc_file, 'r') as fin:
        block_dict = json.load(fin)
        mc_cls_names = sorted(list(block_dict.keys()))

    for label_name, v in orig_cls_dict.items():
        mc_idx = get_category_sem_idx(mc_cls_names, label_name)

        mapping_dict[label_name] = {
            "sem_idx": v["sem_idx"],
            "mc_idx": mc_idx,
        }

    sem_id_max = max(mapping_dict[x]['sem_idx'] for x in mapping_dict.keys())
    #### HAVE TO provide basic structures
    basic_arr = ['floor', 'ceiling', 'wall']
    for i, label_name in enumerate(basic_arr):
        label_idx = mapping_dict.get(label_name, {}).get('sem_idx')
        if label_idx is not None:
            continue 

        mc_idx = get_category_sem_idx(mc_cls_names, label_name)
        mapping_dict[label_name] = {
            "sem_idx": sem_id_max + i + 1,
            "mc_idx": mc_idx
        }

    return mapping_dict


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--recon_res_dir', type=str, required=True)
    parser.add_argument('--lift_dir', type=str, required=True)
    parser.add_argument('--glb_dir', type=str, required=True)
    
    parser.add_argument('--mc_block_file', type=str, default="./data/mcblock_names.json")
    parser.add_argument('--out_dir', type=str, default="./results")
    args = parser.parse_args()

    args = parser.parse_args()
    return args 

def main():
    args = options()

    data_dir = args.data_dir
    recon_res_dir = args.recon_res_dir
    lift_dir = args.lift_dir 
    glb_dir = args.glb_dir
    out_dir = args.out_dir 
    if os.path.exists(out_dir):
        os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=== start loading data", flush=True)
    label_file = os.path.join(lift_dir, "info/labels.json")
    label_mapping_dict = get_label_info(label_file, args.mc_block_file)
    sem_label_names = list(label_mapping_dict.keys())
    print("sem_label_names", sem_label_names)

    instance_info_path = os.path.join(lift_dir, "instances/instances_info.json")
    with open(instance_info_path, 'r') as fin:
       instances_info_dict = json.load(fin)
    
    semantic_label_path = os.path.join(lift_dir, "info/semantic_seg.npy")
    semantic_labels = np.load(semantic_label_path).astype(int)

    color_mesh_path = os.path.join(recon_res_dir, "point_cloud_human3r.ply")
    pcd = o3d.io.read_point_cloud(color_mesh_path)
    all_colors = np.array(pcd.colors, dtype=np.float32)
    all_points = np.array(pcd.points, dtype=np.float32)
    print("=== finish loading data")

    print("=== align point cloud")
    all_points = compute_aligned_axis_points(all_points, semantic_labels, label_mapping_dict)
    pcd.points = o3d.utility.Vector3dVector(all_points)

    print("=== init grid")
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    voxel_size = np.around((bbox_max[2] - bbox_min[2]) / 9, decimals=2)
    voxel_size = min(0.5, voxel_size)
    value_ranges = np.stack([bbox_min, bbox_max], axis=1)
    grid = get_grid(bbox_min, bbox_max, voxel_size=voxel_size)
    print("verts range:", value_ranges, flush=True)
    print(f"grid_shape:{grid.shape}, voxel_size={voxel_size}", flush=True)

    grid_label_candidates = get_labels_candidates(all_points, semantic_labels, grid.shape, value_ranges)
    grid_dict = assign_labels_to_grid(grid_label_candidates, grid, label_mapping_dict)
    grid_dict["value_ranges"] = value_ranges 
    grid_dict["voxel_size"] = voxel_size

    #### only visualize basic parts
    vis_grid_points, vis_grid_colors = visualize(grid_dict["grid"], grid_dict["basic_label"])
    save_verts(vis_grid_points, vis_grid_colors, file_name=os.path.join(out_dir, "vis_grid.ply"))

    ### for final mc position
    global_pos = [2302, 100, 982]

    out_npy_path = os.path.join(out_dir, "voxel_labels.npy")
    basic_mc_data = build_basic_block(grid_dict, label_mapping_dict, global_pos)
    np.save(out_npy_path, basic_mc_data)

    
    glb_out_dir = os.path.join(out_dir, "glb")
    os.makedirs(glb_out_dir, exist_ok=True)
    mesh_pos_dict = fit_obj_mesh(glb_dir, glb_out_dir, grid_dict, all_points, sem_label_names, instances_info_dict, global_pos)
    
    mc_out_dir = os.path.join(out_dir, "mcfunction")
    os.makedirs(mc_out_dir, exist_ok=True)
    get_mc_instruction(args.mc_block_file, basic_mc_data, mesh_pos_dict, mc_out_dir)

    out_json_path = os.path.join(out_dir, "mesh_pos.json")
    with open(out_json_path, 'w') as fout:
        json.dump(mesh_pos_dict, fout)
    
if __name__ == "__main__":
    main()
    print("=== done", flush=True)