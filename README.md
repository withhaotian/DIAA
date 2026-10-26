# DIAA

## Installation

Create an environment, install a PyTorch build compatible with the target CUDA runtime, and install the project dependencies:

```bash
conda create -n diaa python=3.10 -y
conda activate diaa

# Install the CUDA-enabled PyTorch build appropriate for your device.
pip install torch==2.7.0
pip install -r requirements.txt  # includes gptqmodel for GPTQ checkpoints
```
## Usage

Run the matched Vanilla and DIAA evaluations from the project root:

```bash
bash eval.sh
```

The default LUT is located at:

```text
model/diaa/offline/cache/lut_cache_Llama-3.2-3B-Instruct_topk.safetensors
or
model/diaa/offline/cache/lut_cache_vicuna-7b-v1.5_topk.safetensors
```

The LUT must match the model vocabulary and tokenizer. The default evaluation path supports greedy decoding, batch size one, and a single device.

## Platform (Reported in the Paper)

| Component | Specification |
|---|---|
| Device | NVIDIA Jetson Orin Nano Super |
| Memory | 8 GB Unified RAM |
| Storage | 256 GB SSD |
| GPU | 1,024-core GPU, 67 TOPS |
| JetPack | v6.2 |
| CUDA | v12.6 |

## Citation

```bibtex
@inproceedings{tian2026diaa,
  title={DIAA: A Decoding-Efficient Inference Acceleration Approach for On-Device Large Language Models},
  author={Tian, Hao and Lu, Sheng and Tian, Fuwen and Cui, Guangming and Li, Zheng and Zhang, Xuyun and Sheng, Quan Z and Dou, Wanchun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={25896--25904},
  year={2026}
}
```
