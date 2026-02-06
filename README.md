# SafeVerse
SafeVerse, a Minecraft-based embodied AI simulation platform, is built to address the core demand for high-fidelity, customizable simulation environments in Embodied AI research (VLN, embodied interaction, adversarial robustness verification, etc.). It features a video-driven scene generation pipeline that parses real-world video to editable Minecraft virtual scenes, and further provides a unified foundational platform for downstream embodied agent training and adversarial attack defense verification, enabling end-to-end research and validation for Embodied AI.

## Getting Started

```bash
git clone --recursive https://github.com/AI45Lab/SafeVerse.git
cd SafeVerse
git lfs pull
```

For your convenience, we have split the project into independent modules. Please follow the documentation to run the corresponding part:
- [Video to Minecraft scene](doc/video_to_mc.md)
- [Agent Training](doc/agent_training.md)

## Acknowledgements
This project is based on the following awesome repositories:

- [Human3R](https://github.com/fanegg/Human3R)
- [SAM3D](https://github.com/facebookresearch/sam-3d-objects)
- [GPT-4o](https://github.com/marketplace/models/azure-openai/gpt-4o)

Thanks for the authors for their valuable works.