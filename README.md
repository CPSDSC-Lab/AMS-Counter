# AMS-Counter 🔢  

*A PyTorch implementation for **text-guided zero-shot object counting** with CLIP & DINOv2*

---

## ✨ Highlights

* **Adaptive Multi-view Similarity-map (AMS)** – fuses spatial **and** frequency-domain clues  
* **U-shaped Cross-attention Decoder (UC-Decoder)** – refines density maps under language guidance  
* Ready-made **pre-trained weights** for FSC-147  
* Minimal dependencies (PyTorch ≥ 2.1 + Lightning)

---

## 📂 Repository Layout

```text
AMS-Counter
├── CLIP/               # OpenAI CLIP helpers
├── data/               # Datasets live here
├── dinov2/             # Meta-AI DINOv2 fork
├── lightning_logs/     # PL logs & checkpoints
├── models/             # core code of AMS-Counter
├── util/
├── weights/            # ← put pre-trained weights here
├── runner.py           # Main train/eval script
```

---

## 🔧 Installation

> Tested on Python 3.10, CUDA 12.1, Linux

```bash
# 1) clone & create env
git clone https://github.com/CPSDSC-Lab/AMS-Counter.git
cd AMS-Counter
conda create -n ams python=3.10 -y
conda activate ams
pip install -r requirements.txt

# 2) install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# 3) install DINOv2 (or use the bundled fork)
git clone https://github.com/facebookresearch/dinov2.git
pip install -e dinov2
```

---

## 📥 Dataset Preparation

### FSC-147

| Resource        | Direct link                                                  |
| --------------- | ------------------------------------------------------------ |
| Images (384 px) | <https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view> |
| GT density maps | <https://archive.org/download/FSC147-GT/FSC147_GT.zip>       |

```text
data/FSC147/images_384/
data/FSC147/gt_density_map/
```

---

### CARPK (optional)

1. Visit <https://lafi.github.io/LPN/>  
2. Fill the EULA form → download **CARPK.tar.gz**  
3. Extract under `data/CARPK/`

---

## 💾 Pre-trained Weights

| Model                 | Download                                                     |
| --------------------- | ------------------------------------------------------------ |
| AMS-Counter (FSC-147) | <https://github.com/CPSDSC-Lab/AMS-Counter/releases/download/v1.0/ams_counter_fsc147.pth> |

```bash
# example: keep it in weights/
models/ams_counter_fsc147.pth
```

---

## 🚀 Quick Start

### Evaluate on FSC-147

```bash
python runner.py \
  --mode test \
  --dataset FSC147 \
  --ckpt weights/ams_counter_fsc147.pth \
  --batch_size 8
```


### Use Web UI

```bash
python runner.py --mode app --ckpt weights/ams_counter_fsc147.pth
```

---

## 📝 Citation

```bibtex
@inproceedings{qian2025amscounter,
  title     = {AMS-Counter: Text-Guided Zero-shot Object Counting via Adaptive Multi-view Similarity-map},
  author    = {Qian, Cheng and Cao, Jiwu and Mao, Ying and Liu, Kai and Zhu, Peng and Sang, Jun},
  booktitle = {Preprint of the IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2025},
  note      = {ICME 2025 preprint, accepted},
  url       = {https://arxiv.org/abs/xxxx.xxxxx}
}
```

---

## 📜 License

MIT – see `LICENSE`

---

## 🙏 Acknowledgements

* **OpenAI CLIP** <https://github.com/openai/CLIP>  
* **Meta AI DINOv2** <https://github.com/facebookresearch/dinov2>  
* **FSC-147** & **CARPK** datasets

Happy counting 🚀
