import cv2
import numpy as np

def estimate_depth(img, patch_size=15):
    img_f = img.astype(np.float32) / 255.0
    dark = np.min(img_f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_ch = cv2.erode(dark, kernel)
    depth = 1.0 - dark_ch
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def white_balance(img):
    """
    Underwater-aware white balance.
    Only corrects channels that are genuinely dominant — avoids
    overcorrecting already dark/neutral images.
    """
    img_f = img.astype(np.float32)
    mean_b = np.mean(img_f[:, :, 0])
    mean_g = np.mean(img_f[:, :, 1])
    mean_r = np.mean(img_f[:, :, 2])
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    def scale(mean_c):
        ratio = mean_gray / (mean_c + 1e-8)
        # clamping  correction — gentle nudge only 
        return np.clip(ratio, 0.85, 1.3)

    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * scale(mean_b), 0, 255)
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] * scale(mean_g), 0, 255)
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * scale(mean_r), 0, 255)
    return img_f.astype(np.uint8)

def correct_attenuation(img, depth):
    """
    Gentle per-channel attenuation correction.
    Red is absorbed most in water, blue least.
    """
    img_f = img.astype(np.float32) / 255.0
    betas = [0.15, 0.08, 0.03]  # R, G, B — very gentle
    J = np.zeros_like(img_f)
    for c, beta in enumerate(betas):
        transmission = np.exp(-beta * depth)
        transmission = np.clip(transmission, 0.75, 1.0)  # high floor = safe
        J[:, :, c] = img_f[:, :, c] / transmission
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)

def sea_thru_correct(img_bgr):
    # step 1: fix color cast
    balanced = white_balance(img_bgr)
    # step 2: estimate depth
    depth = estimate_depth(balanced)
    # step 3: gentle attenuation correction
    corrected = correct_attenuation(balanced, depth)
    return corrected, depth