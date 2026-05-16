# Enhanced Underwater Image Restoration  
Sea-Thru Preprocessing + FUnIE-GAN + Real-ESRGAN  

---

## Overview

This project extends the FUnIE-GAN framework by introducing a physics-guided preprocessing stage and an optional super-resolution stage for improved underwater image enhancement.

### Pipeline

1. Sea-Thru Inspired Preprocessing  
   - Adaptive white balancing  
   - Depth estimation (dark channel prior)  
   - Attenuation correction  

2. FUnIE-GAN Enhancement  
   - GAN-based image enhancement  

3. Real-ESRGAN Super-Resolution (Optional)  
   - 4× upscaling for sharper textures  

---

## Installation

### 1. Clone FUnIE-GAN repository

git clone https://github.com/xahidbuffon/FUnIE-GAN

### 2. Install dependencies

pip install torch torchvision opencv-python numpy

---

## Download Required Resources

### 1. FUnIE-GAN Weights

Download and place in models/:

funie_generator.pth

Source:  
https://github.com/xahidbuffon/FUnIE-GAN

---

### 2. Real-ESRGAN Weights

The script will auto-download, or manually place:

RealESRGAN_x4plus.pth

Source:  
https://github.com/xinntao/Real-ESRGAN

---

### 3. Datasets

#### EUVP Dataset (Seen Data)

https://irvlab.cs.umn.edu/resources/euvp-dataset

Place test samples in:

EUVP/test_samples/Inp/

---

#### UIEB Dataset (Unseen Data)

https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw

Place images in:

EUVP/out_dataset/

---

## Usage

Run the full pipeline:

python pipeline.py

---

## Output

The script generates comparison images in:

data/output/
├── comparisons_known/
├── comparisons_unknown/

Each output contains:

1. Input image  
2. FUnIE-GAN output  
3. Sea-Thru corrected  
4. Novelty 1 (Sea-Thru + FUnIE-GAN)  
5. Novelty 2 (Sea-Thru + FUnIE-GAN + ESRGAN)  

---

## Individual Components

### Sea-Thru Preprocessing

from sea_thru import sea_thru_correct

corrected, depth = sea_thru_correct(image)

---

### FUnIE-GAN Only

output = run_funiegan(image)

---

### Novelty 1

output, corrected = run_seathru_funiegan(image)

---

### Full Pipeline (Novelty 2)

output = run_full_pipeline(image)

---

## Key Features

- Physics-guided correction improves color balance  
- GAN enhances perceptual quality  
- Super-resolution improves sharpness  
- Works on both seen (EUVP) and unseen (UIEB) datasets  

---

## References

FUnIE-GAN Paper  
https://arxiv.org/pdf/1903.09766  

FUnIE-GAN Repository  
https://github.com/xahidbuffon/FUnIE-GAN  

EUVP Dataset  
https://irvlab.cs.umn.edu/resources/euvp-dataset  

UIEB Dataset  
https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw  

Real-ESRGAN  
https://github.com/xinntao/Real-ESRGAN  

---

## Notes

- GPU recommended for faster inference  
- Real-ESRGAN is computationally expensive  
- Ensure correct folder paths before running  

---

## Author

Group Name: GaMa.CV  
Charitha (B23EE1039)  
Mallam Vishnu Priya (B23EE1040)  
