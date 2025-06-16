
# 🔐 Implementation of Effective & Resilient Backdoor Attack Framework

> 📄 **An Effective and Resilient Backdoor Attack Framework against Deep Neural Networks and Vision Transformers**  
> https://arxiv.org/abs/2412.06149

This implementation is adapted from the method proposed in [ATTEQ-NN: Attention-based QoE-aware Evasive Backdoor Attacks (NDSS 2022)](https://www.ndss-symposium.org/ndss-paper/auto-draft-238/).

## 🚀 Reproduction Instructions


### ✅ Step 1: Clone and Setup Environment

```bash
git clone https://github.com/boweitian/Effective_backdoor_2025.git
cd Effective_backdoor_2025

conda env create -f vit2025.yaml
conda activate vit2025
```



### ✅ Step 2: Prepare Datasets

CIFAR10 / CIFAR100 / MNIST:
- These datasets will be automatically downloaded by torchvision.

GTSRB:
- Place the dataset in: ./GTSRB/trainingset/ and ./GTSRB/testset/
- Ensure training.csv and test.csv are present.

ImageNette:
- Place the data in ./imagenette/train/ and ./imagenette/val/

VGGFlower:
- Download from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Place under /mnt/data/ywb_bk/Dataset/VGGFlower or modify `vggflower.py`.



### ✅ Step 3: Run Single-Target ABAC Attack
```bash
python ABAC_attack.py -task cifar10 -d 0 -prate 0.05 -pnum 4 -opac 0.2 -mode 4 \
  -iter 3 -b 64 -nw 4 -lr 1e-3 -e 30
```

Supported tasks: cifar10 / cifar100 / gtsrb / imagenette / vggflower / mnist



# ✅ Output Files

After completion, you will find:
- ./logs/[task]/.../log.txt              (training logs)
- ./logs/[task]/.../ck.pth               (model checkpoint)
- ./intermediate/                        (intermediate files)
- ./poisoned_test_loader.pt              (poisoned dataloader)


## 📄 Citation

If you find our work helpful, please cite:

```bibtex
@misc{gong2024effectiveresilientbackdoorattack,
  title={An Effective and Resilient Backdoor Attack Framework against Deep Neural Networks and Vision Transformers},
  author={Xueluan Gong and Bowei Tian and Meng Xue and Yuan Wu and Yanjiao Chen and Qian Wang},
  year={2024},
  eprint={2412.06149},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.06149}
}
```
