import os, glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import urllib.request
from sea_thru import sea_thru_correct

import sys
sys.path.append(os.path.dirname(__file__))
from nets.funiegan import GeneratorFunieGAN

# ── device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── FUnIE-GAN ───────────────────────────────────────────────
generator = GeneratorFunieGAN()
generator.load_state_dict(
    torch.load("models/funie_generator.pth", map_location=device)
)
generator.eval().to(device)

# ── Real-ESRGAN (pure torch, no basicsr) ────────────────────
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(
            torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(
            torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

def load_realesrgan():
    model_path = "models/RealESRGAN_x4plus.pth"
    if not os.path.exists(model_path):
        print("Downloading Real-ESRGAN weights (~64MB)...")
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model_path
        )
        print("Downloaded.")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    weights = torch.load(model_path, map_location=device)
    # handle both raw and wrapped checkpoints
    if "params_ema" in weights:
        weights = weights["params_ema"]
    elif "params" in weights:
        weights = weights["params"]
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    print("Real-ESRGAN loaded.")
    return model

esrgan = load_realesrgan()

# ── helpers ──────────────────────────────────────────────────
def preprocess_funiegan(img_bgr, size=(256, 256)):
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 127.5 - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

def postprocess_funiegan(tensor):
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def run_esrgan(img_bgr):
    """Run Real-ESRGAN on a BGR uint8 image, tile if needed."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = esrgan(t)
    out = out.squeeze(0).cpu().detach().numpy()
    out = (out.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def run_funiegan(img_bgr):
    inp = preprocess_funiegan(img_bgr)
    with torch.no_grad():
        out = generator(inp)
    return postprocess_funiegan(out)

def run_seathru_funiegan(img_bgr):
    corrected, _ = sea_thru_correct(img_bgr)
    inp = preprocess_funiegan(corrected)
    with torch.no_grad():
        out = generator(inp)
    return postprocess_funiegan(out), corrected

def run_full_pipeline(img_bgr):
    corrected, _ = sea_thru_correct(img_bgr)
    inp = preprocess_funiegan(corrected)
    with torch.no_grad():
        funiegan_out = generator(inp)
    funiegan_img = postprocess_funiegan(funiegan_out)
    return run_esrgan(funiegan_img)

def add_label(img, text, height=32):
    h, w = img.shape[:2]
    bar = np.zeros((height, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    cv2.putText(bar, text, (6, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

def make_comparison(input_path, out_dir, display_size=(256, 256)):
    img = cv2.imread(input_path)
    if img is None:
        print(f"  Could not read: {input_path}")
        return

    name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"  Processing: {name}")

    repo_out              = run_funiegan(img)
    novelty1_out, seathru = run_seathru_funiegan(img)
    novelty2_out          = run_full_pipeline(img)

    p1 = cv2.resize(img,          display_size)
    p2 = cv2.resize(repo_out,     display_size)
    p3 = cv2.resize(seathru,      display_size)
    p4 = cv2.resize(novelty1_out, display_size)
    p5 = cv2.resize(novelty2_out, display_size)

    p1 = add_label(p1, "1. Input (raw)")
    p2 = add_label(p2, "2. Repo: FUnIE-GAN only")
    p3 = add_label(p3, "3. Sea-Thru corrected")
    p4 = add_label(p4, "4. Novelty 1: Sea-Thru + FUnIE-GAN")
    p5 = add_label(p5, "5. Novelty 2: Sea-Thru + FUnIE-GAN + ESRGAN")

    comparison = np.hstack([p1, p2, p3, p4, p5])

    title_bar = np.zeros((40, comparison.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (20, 20, 20)
    cv2.putText(title_bar,
                f"Full pipeline comparison  |  {name}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    final = np.vstack([title_bar, comparison])
    out_path = os.path.join(out_dir, f"compare_{name}.png")
    cv2.imwrite(out_path, final)
    print(f"  saved -> {out_path}\n")

# ── run ──────────────────────────────────────────────────────

DATASETS = [
    {
        "name":      "known",
        "input_dir": "../EUVP/test_samples/Inp/",
        "limit":     25,
        "out_dir":   "data/output/comparisons_known/",
    },
    {
        "name":      "unknown",
        "input_dir": "../EUVP/out_dataset/",
        "limit":     25,
        "out_dir":   "data/output/comparisons_unknown/",
    },
]

for dataset in DATASETS:
    input_dir  = dataset["input_dir"]
    output_dir = dataset["out_dir"]
    limit      = dataset["limit"]
    name       = dataset["name"]

    os.makedirs(output_dir, exist_ok=True)

    images = sorted(
        glob.glob(input_dir + "*.jpg") +
        glob.glob(input_dir + "*.png")
    )[:limit]  # take first 50, keeps original filenames

    if not images:
        print(f"No images found in {input_dir}")
        continue

    print(f"\n── {name.upper()} dataset: {len(images)} images ──")
    print(f"   Output → {output_dir}\n")

    for path in images:
        make_comparison(path, output_dir)

    print(f"Done! {name} comparisons saved to: {output_dir}")

print("\nAll datasets complete!")
print("  Known   → data/output/comparisons_known/")
print("  Unknown → data/output/comparisons_unknown/")