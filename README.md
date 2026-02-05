# SafeVerse

## Getting Started

```bash
git clone --recursive https://github.com/AI45Lab/SafeVerse.git
cd SafeVerse
git lfs pull
```

### Installation

#### 1. 3D reconstruction

Please follow the official instructions of [Human3R](https://github.com/fanegg/Human3R) to configure the environment and download the required models and checkpoints.

For selecting the top-k (k=5) views per instance, `PyTorch3D` is needed. Please refer to the [official installation guid for PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for setup.

#### 2. 3D object generation

Please follow the official instructions of [SAM3D](https://github.com/facebookresearch/sam-3d-objects) to set up the environment and download the necessary models and checkpoints.


<details>
   <summary><strong>Troubleshooting: xatlas Segmentation Fault (Click to expand)</strong></summary>

   If you encounter a `Segmentation fault` related to **xatlas** (similar to [Issue #329](https://github.com/microsoft/TRELLIS/issues/329)), try running on a CPU-only machine. If that's not possible, follow these steps:

   1.  **Replace the utility file:**
       Use the provided `postprocessing_utils.py` to replace the specific file in the submodule.

       ```bash
       cp ./utils/postprocessing_utils.py ./submodules/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py
       ```

   2.  **Process the intermediate results:**
       Run the auxiliary script to handle the saved data.

       ```bash
       python ./utils/process.py
       ```

   3.  **Continue the pipeline:**
       Re-run your generation command. The modified script will read the processed files and bypass the crash.
   
   </details>


#### 3. Convert mesh to Minecraft mod

Please download Blender from the [official website](https://www.blender.org/download/), extract and rename the folder to `blender_app`, and place it under the project root.

This step converts the generated 3D meshes into a Minecraft mod using Blender and a NeoForge 1.21.1 mod template (`examplemod/`). The project requires OpenJDK 8, which can be installed via conda:

```bash
conda install --channel=conda-forge openjdk=8 -y
```

### Demo
We provide `example_data.zip`, extract it and place it under the project root. Please follow the steps in `run_recon_pipeline.sh` to run the demo, with dedicated environment requirements for different steps as follows:
- Steps 1–3: Use the environment with support for Human3R reconstruction and PyTorch3D
- Steps 4–6: Use the environment with support for SAM3D
- Step 8: Use the environment with support for OpenJDK 8

**IMPORTANT**: Modify the following configurations before execution:
1. Update DATA_DIR, OUTPUT_DIR, JAVA_HOME in `run_recon_pipeline.sh`
2. Replace API_KEY and ENDPOINT with your own in `scripts/get_bestimg_for_sam3d.py`


Finally, you can execute the generated files in the `mcfunction/` folder in Minecraft to build the final scene. 

![demo output](assets/example_recon.gif)

Note: Automatic alignment of object positions and rotations is still in development, so minor manual adjustments may be needed for optimal results.


## ToDos
- [x] Release Minecraft scene reconstruction code.
- [ ] Release agent training code.

## Acknowledgements
This project is based on the following awesome repositories:

- [Human3R](https://github.com/fanegg/Human3R)
- [SAM3D](https://github.com/facebookresearch/sam-3d-objects)
- [GPT-4o](https://github.com/marketplace/models/azure-openai/gpt-4o)

Thanks for the authors for their valuable works.