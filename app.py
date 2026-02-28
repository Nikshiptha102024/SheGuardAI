import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import dct

def analyze_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    scores = []

    # 1️⃣ Frequency Anomaly (less aggressive)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    freq_score = np.clip(np.var(magnitude) / 120000, 0, 1)
    scores.append(freq_score)

    # 2️⃣ Sensor Noise (PRNU-like)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)
    prnu_score = 1 - np.clip(np.var(noise) / 1200, 0, 1)
    scores.append(prnu_score)

    # 3️⃣ Color Channel Inconsistency (VERY IMPORTANT)
    r, g, b = cv2.split(img_np)
    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
    corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
    corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
    color_score = np.clip(1 - np.mean([corr_rg, corr_rb, corr_gb]), 0, 1)
    scores.append(color_score)

    # 4️⃣ Texture Entropy (retuned)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    entropy_score = np.clip((entropy - 6.5) / 2.5, 0, 1)
    scores.append(entropy_score)

    # 5️⃣ DCT Block Artifact Detection (better than diff)
    dct_img = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    block_energy = np.mean(np.abs(dct_img[::8, ::8]))
    compression_score = np.clip(block_energy / 40, 0, 1)
    scores.append(compression_score)

    # 6️⃣ Metadata (low weight)
    metadata_score = 0.2
    try:
        if img._getexif() is None:
            metadata_score = 0.4
        else:
            metadata_score = 0.1
    except:
        metadata_score = 0.4
    scores.append(metadata_score)

    # 7️⃣ Face-based Skin Texture Check
    face_score = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x,y,wf,hf = faces[0]
        face_roi = gray[y:y+hf, x:x+wf]
        lap_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        face_score = np.clip(1 - lap_var / 600, 0, 1)
    scores.append(face_score)

    # 🔹 Weighted fusion (REBALANCED)
    raw_score = (
        0.15 * scores[0] +  # FFT
        0.20 * scores[1] +  # PRNU
        0.20 * scores[2] +  # Color
        0.10 * scores[3] +  # Entropy
        0.10 * scores[4] +  # DCT
        0.10 * scores[5] +  # Metadata
        0.15 * scores[6]    # Face
    )

    # 🔥 FINAL CALIBRATION (KEY FIX)
    probability = np.clip((raw_score - 0.35) * 2.0, 0, 1)

    return round(probability * 100, 2)
    
