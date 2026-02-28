
def analyze_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape

    # 1️⃣ Frequency Domain Analysis (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    freq_score = np.clip(np.var(magnitude) / 50000, 0, 1)

    # 2️⃣ Noise Inconsistency
    noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = np.clip(np.var(noise) / 2000, 0, 1)

    # 3️⃣ Edge Coherence
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w)
    edge_score = np.clip(edge_density * 2.0, 0, 1)

    # 4️⃣ Texture Entropy
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_norm = hist / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    entropy_score = np.clip((entropy - 5) / 3, 0, 1)

    # 5️⃣ JPEG Block Artifact Detection
    block_diff = np.mean(np.abs(np.diff(gray, axis=0))) + \
                 np.mean(np.abs(np.diff(gray, axis=1)))
    compression_score = np.clip(block_diff / 50, 0, 1)

    # 6️⃣ Metadata Integrity
    metadata_score = 0.2
    try:
        exif = img._getexif()
        if exif is None:
            metadata_score = 0.6
        else:
            metadata_score = 0.1
    except:
        metadata_score = 0.6

    # 7️⃣ Face Region Boost (Optional if OpenCV Haar loaded)
    face_score = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        face_score = 0.2

    # Weighted Risk Score
    raw_score = (
        0.20 * freq_score +
        0.15 * noise_score +
        0.15 * edge_score +
        0.15 * entropy_score +
        0.15 * compression_score +
        0.10 * metadata_score +
        0.10 * face_score
    )

    # Calibration bias toward natural images
    probability = np.clip((raw_score - 0.25) * 1.3, 0, 1)

    return round(probability * 100, 2)
