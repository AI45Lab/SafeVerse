MODEL_PATH=src/human3r_896L.pth

DATA_DIR=./example_data
OUTPUT_ROOT=./outputs

echo "=== STEP 1: run Human3R reconstruction ==="
python scripts/run_recon.py \
    --model_path ${MODEL_PATH} \
    --seq_path ${DATA_DIR}/color \
    --size 512 --subsample 1 \
    --use_ttt3r --reset_interval 100 \
    --output_dir ${OUTPUT_ROOT}

echo "=== STEP 2: get instances in 3D ==="
python scripts/get_instances.py \
    --data_dir ${DATA_DIR} \
    --recon_res_dir ${OUTPUT_ROOT} \
    --out_dir ${OUTPUT_ROOT}/lift_to_3d

echo "=== STEP 3: get instances' corresponding images ==="
#### environment needs pytorch3d 
python scripts/select_instances_2d.py \
    --data_dir ${DATA_DIR} \
    --recon_res_dir ${OUTPUT_ROOT} \
    --lift_dir ${OUTPUT_ROOT}/lift_to_3d \
    --out_dir ${OUTPUT_ROOT}/lift_to_3d

echo "=== STEP 4: get instances' best images for sam3d ==="
python scripts/get_bestimg_for_sam3d.py \
    --instances_images ${OUTPUT_ROOT}/lift_to_3d/instances_vis \
    --out_dir ${OUTPUT_ROOT}/best_img


echo "=== STEP 5: get instances' glb from sam3d ==="
# cp scripts/get_glb.py ./submodules/sam-3d-objects
cd ./submodules/sam-3d-objects
python get_glb.py \
    --best_img ../../${OUTPUT_ROOT}/best_img \
    --out_dir ../../${OUTPUT_ROOT}/instance_glb
cd ../../

GLB_DIR=${OUTPUT_ROOT}/instance_glb

echo "=== STEP 6: CONVERSION: align glb files to minecraft coords ==="
python scripts/align_glb_to_mc.py \
    --data_dir ${DATA_DIR} \
    --recon_res_dir ${OUTPUT_ROOT} \
    --lift_dir ${OUTPUT_ROOT}/lift_to_3d \
    --glb_dir ${GLB_DIR} \
    --out_dir ${OUTPUT_ROOT}/lift_to_3d/voxels

echo "===STEP 7: CONVERSION: glb to minecraft mod code==="
./blender_app/blender -b -P scripts/glb2mod.py -- \
    --glb_dir ${OUTPUT_ROOT}/lift_to_3d/voxels/glb \
    --java_dir ./examplemod/src/main/java/com/example/examplemod \
    --assert_dir ./examplemod/src/main/resources

echo "=== STEP 8: get minecraft mod==="
cd examplemod
export JAVA_HOME=${your_conda_env}/mcvla/bin/java
chmod +x gradlew
./gradlew clean
./gradlew build

echo "=== The mod will be at examplemod/build/libs and the mc function will be at ${OUTPUT_ROOT}/lift_to_3d/voxels/mcfunction ==="