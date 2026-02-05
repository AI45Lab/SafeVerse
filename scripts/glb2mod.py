import bpy
import os
import uuid
import json
import sys
import math
from mathutils import Vector
import argparse

PACKAGE_NAME = "com.example.examplemod"
MOD_ID = "examplemod"


def clear_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.images: bpy.data.images.remove(block)

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_and_fix_normals(obj):

    if obj.type != 'MESH' or not obj.data.polygons:
        return False
    
    mesh = obj.data
    was_fixed = False
    
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    try:
        center = Vector([0, 0, 0])
        if len(mesh.vertices) > 0:
            for v in mesh.vertices:
                center += v.co
            center /= len(mesh.vertices)
        
        inward_faces = 0
        if len(mesh.polygons) > 0:
            for poly in mesh.polygons:
                face_center = poly.center
                to_face = face_center - center
                if poly.normal.dot(to_face) < 0:
                    inward_faces += 1
            
            inward_ratio = inward_faces / len(mesh.polygons)
            
            if inward_ratio > 0.6:
                bpy.ops.mesh.flip_normals()
                was_fixed = True
        
        bpy.ops.mesh.normals_make_consistent(inside=False)
        was_fixed = True
        
    except Exception as e:
        try:
            bpy.ops.mesh.normals_make_consistent(inside=False)
            was_fixed = True
        except:
            pass
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        mesh.update()
    
    return was_fixed

def process_and_align_model(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001) 
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    check_and_fix_normals(obj)
    
    dim = obj.dimensions
    original_max_size = max(dim.x, dim.y, dim.z)
    
    target_scale = 1.0
    if original_max_size < 0.2: 
        target_scale = 16.0 
    elif original_max_size > 8.0:
        target_scale = 8.0 / original_max_size
    
    if abs(target_scale - 1.0) > 0.001:
        for v in obj.data.vertices:
            v.co *= target_scale
        obj.data.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not verts: return 0, 0, 0
    
    avg_x = sum(v.x for v in verts) / len(verts)
    avg_y = sum(v.y for v in verts) / len(verts)
    
    min_z = min(v.z for v in verts)
    
    offset = Vector((avg_x, avg_y, min_z))
    
    for v in obj.data.vertices:
        v.co -= offset
    obj.data.update()
    
    final_w = obj.dimensions.x
    final_d = obj.dimensions.y
    final_h = obj.dimensions.z
        
    return final_w, final_d, final_h

def generate_json_files(name_slug, paths, mod_id, material_uuid):
    model_block_data = {
        "parent": "minecraft:block/block",
        "loader": "neoforge:obj",
        "model": f"{mod_id}:models/block/{name_slug}.obj",
        "flip_v": True,
        "textures": {
            "particle": f"{mod_id}:block/{name_slug}",
            f"#{material_uuid}": f"{mod_id}:block/{name_slug}"
        },
        "transform": {
            "translation": [0.5, 0, 0.5],
            "scale": [1.0, 1.0, 1.0] 
        },
        "render_type": "cutout"
    }
    with open(os.path.join(paths['block_models'], f"{name_slug}.json"), 'w') as f:
        json.dump(model_block_data, f, indent=2)

    bs_data = {"variants": {"": { "model": f"{mod_id}:block/{name_slug}" }}}
    with open(os.path.join(paths['blockstates'], f"{name_slug}.json"), 'w') as f:
        json.dump(bs_data, f, indent=2)

    item_data = {
        "parent": f"{mod_id}:block/{name_slug}",
        "display": {
            "gui": { "rotation": [30, 225, 0], "translation": [0, 0, 0], "scale": [0.625, 0.625, 0.625] },
            "ground": { "scale": [0.25, 0.25, 0.25] },
            "fixed": { "scale": [0.5, 0.5, 0.5] },
            "thirdperson_righthand": { "rotation": [75, 45, 0], "translation": [0, 2.5, 0], "scale": [0.375, 0.375, 0.375] },
            "firstperson_righthand": { "rotation": [0, 45, 0], "scale": [0.4, 0.4, 0.4] }
        }
    }
    with open(os.path.join(paths['item_models'], f"{name_slug}.json"), 'w') as f:
        json.dump(item_data, f, indent=2)

def process_mtl_file(mtl_path, texture_name, mod_id):
    try:
        with open(mtl_path, 'r') as f: lines = f.readlines()
    except FileNotFoundError: return
    new_lines = ["# Made in Blender Script\n"]
    for line in lines:
        if line.startswith("newmtl"):
            new_lines.append(line)
            new_lines.append(f"map_Kd {mod_id}:block/{texture_name}\n")
        elif line.startswith("map_Kd"): continue
        elif line.startswith("#"): pass
    new_lines.append("newmtl none\n")
    with open(mtl_path, 'w') as f: f.writelines(new_lines)

def generate_java_registry(item_data_list, output_dir, package_name):
    ensure_folder(output_dir)
    class_name = "GeneratedBlocks"
    file_path = os.path.join(output_dir, f"{class_name}.java")

    java_content = []
    
    java_content.append(f"package {package_name};\n")
    java_content.append("import net.minecraft.world.item.BlockItem;")
    java_content.append("import net.minecraft.world.level.block.Block;")
    java_content.append("import net.minecraft.world.level.block.SoundType;")
    java_content.append("import net.minecraft.world.level.block.state.BlockBehaviour;")
    java_content.append("import net.minecraft.world.level.block.state.BlockState;")
    java_content.append("import net.minecraft.world.level.material.MapColor;")
    java_content.append("import net.minecraft.world.phys.shapes.CollisionContext;")
    java_content.append("import net.minecraft.world.phys.shapes.VoxelShape;")
    java_content.append("import net.minecraft.core.BlockPos;")
    java_content.append("import net.minecraft.world.level.BlockGetter;")
    java_content.append("import net.neoforged.neoforge.registries.DeferredBlock;")
    java_content.append("import net.neoforged.neoforge.registries.DeferredItem;")
    java_content.append("\n")

    java_content.append(f"public class {class_name} {{")
    java_content.append("    public static void init() {} \n")

    for item in item_data_list:
        name_slug = item["name"]
        final_w = item["width"]
        final_d = item["depth"]
        final_h = item["height"]
        
        var_name = name_slug.upper()
        
        safe_w = min(final_w, 2.0)
        safe_d = min(final_d, 2.0)
        
        w_px = safe_w * 16.0
        d_px = safe_d * 16.0
        h_px = final_h * 16.0
        
        w_px = math.floor(w_px / 16.0) * 16.0
        d_px = math.floor(d_px / 16.0) * 16.0
        h_px = math.floor(h_px / 16.0) * 16.0
        
        w_px = max(w_px, 16.0)
        d_px = max(d_px, 16.0)
        h_px = max(h_px, 16.0)
        
        x_min = 8.0 - (w_px / 2.0)
        x_max = 8.0 + (w_px / 2.0)
        z_min = 8.0 - (d_px / 2.0)
        z_max = 8.0 + (d_px / 2.0)
        y_min = 0.0
        y_max = h_px
        
        shape_str = f"Block.box({x_min:.2f}, {y_min:.2f}, {z_min:.2f}, {x_max:.2f}, {y_max:.2f}, {z_max:.2f})"

        if "plant" in name_slug:
             props = "BlockBehaviour.Properties.of().mapColor(MapColor.PLANT).strength(0.5f).sound(SoundType.GRASS).noOcclusion()"
        else:
             props = "BlockBehaviour.Properties.of().mapColor(MapColor.STONE).strength(3.0f).sound(SoundType.STONE).noOcclusion()"

        java_content.append(f"    public static final DeferredBlock<Block> {var_name}_BLOCK = ExampleMod.BLOCKS.register(\"{name_slug}\", () -> new Block({props}) {{")
        java_content.append("        @Override")
        java_content.append("        public VoxelShape getShape(BlockState state, BlockGetter world, BlockPos pos, CollisionContext context) {")
        java_content.append(f"            return {shape_str};")
        java_content.append("        }")
        java_content.append("    });")
        java_content.append(f"    public static final DeferredItem<BlockItem> {var_name}_BLOCK_ITEM = ExampleMod.ITEMS.registerSimpleBlockItem(\"{name_slug}\", {var_name}_BLOCK);\n")

    java_content.append("    public static void registerItemsToTab(net.minecraft.world.item.CreativeModeTab.Output output) {")
    for item in item_data_list:
        var_name = item["name"].upper()
        java_content.append(f"        output.accept({var_name}_BLOCK_ITEM.get());")
    java_content.append("    }")

    java_content.append("}")

    with open(file_path, "w", encoding='utf-8') as f:
        f.write("\n".join(java_content))
    
    print(f"=== Java 代码已生成: {file_path} ===")

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glb_dir', required=True)
    parser.add_argument('--java_dir', required=True)
    parser.add_argument('--assert_dir', required=True) 
    
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
        
    return parser.parse_args(argv)

def main():
    args = options()

    target_input = args.glb_dir
    target_output_assert = args.assert_dir
    java_output_dir = args.java_dir
    target_mod = MOD_ID
    

    scene = bpy.context.scene
    if scene:
        scene.view_settings.view_transform = 'Standard' 
        scene.sequencer_colorspace_settings.name = 'sRGB'
        scene.display_settings.display_device = 'sRGB'

    paths = {
        'block_models': os.path.join(target_output_assert, 'assets', target_mod, 'models', 'block'),
        'item_models': os.path.join(target_output_assert, 'assets', target_mod, 'models', 'item'),
        'blockstates': os.path.join(target_output_assert, 'assets', target_mod, 'blockstates'),
        'textures': os.path.join(target_output_assert, 'assets', target_mod, 'textures', 'block'),
        'lang': os.path.join(target_output_assert, 'assets', target_mod, 'lang'),
    }
    for p in paths.values(): ensure_folder(p)

    files = [f for f in os.listdir(target_input) if f.lower().endswith('.glb')]
    
    lang_entries = {}
    processed_items_data = []

    for file_name in files:
        try:
            clear_scene()
            base_name = os.path.splitext(file_name)[0].lower().replace(" ", "_")
            name_slug = f"{base_name}"
            full_path = os.path.join(target_input, file_name)
            lang_entries[f"block.{target_mod}.{name_slug}"] = name_slug

            bpy.ops.import_scene.gltf(filepath=full_path)
            
            for img in bpy.data.images:
                if img.colorspace_settings.name != 'sRGB':
                    img.colorspace_settings.name = 'sRGB'
            
            mesh_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not mesh_objs: continue

            bpy.context.view_layer.objects.active = mesh_objs[0]
            if len(mesh_objs) > 1:
                ctx = bpy.context.copy()
                ctx['active_object'] = mesh_objs[0]
                ctx['selected_editable_objects'] = mesh_objs
                bpy.ops.object.join(ctx)
            
            target_obj = bpy.context.view_layer.objects.active
            
            final_w, final_d, final_h = process_and_align_model(target_obj)
            
            processed_items_data.append({
                "name": name_slug,
                "width": final_w,
                "depth": final_d,
                "height": final_h
            })

            bb_uuid = f"m_{uuid.uuid4()}"
            if target_obj.data.materials:
                mat = target_obj.data.materials[0]
                mat.name = bb_uuid
                if mat.use_nodes:
                    tex_node = None
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE':
                            tex_node = node
                            break
                    if tex_node and tex_node.image:
                        img = tex_node.image
                        save_tex_path = os.path.join(paths['textures'], f"{name_slug}.png")
                        
                        original_colorspace = img.colorspace_settings.name
                        img.colorspace_settings.name = 'sRGB' 
                        
                        if not img.pixels:
                            img.reload()
        
                        try:
                            img.save_render(save_tex_path)
                        except Exception as e:
                            img.filepath_raw = save_tex_path
                            img.file_format = 'PNG'
                            img.save()
            else:
                new_mat = bpy.data.materials.new(name=bb_uuid)
                target_obj.data.materials.append(new_mat)

            target_obj_path = os.path.join(paths['block_models'], f"{name_slug}.obj")
            bpy.ops.wm.obj_export(
                filepath=target_obj_path,
                export_selected_objects=True,
                forward_axis='NEGATIVE_Z', 
                up_axis='Y'
            )

            target_mtl_path = os.path.join(paths['block_models'], f"{name_slug}.mtl")
            if os.path.exists(target_mtl_path):
                process_mtl_file(target_mtl_path, name_slug, target_mod)
            
            generate_json_files(name_slug, paths, target_mod, bb_uuid)
            
        except Exception as e:
            import traceback
            traceback.print_exc()

    if lang_entries:
        with open(os.path.join(paths['lang'], "en_us.json"), 'w') as f:
            json.dump(lang_entries, f, indent=2, sort_keys=True)

    if processed_items_data:
        generate_java_registry(processed_items_data, java_output_dir, PACKAGE_NAME)


if __name__ == "__main__":
    main()
    print("=== done", flush=True)
