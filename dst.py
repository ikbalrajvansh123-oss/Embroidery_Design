import numpy as np
import cv2
from pyembroidery import EmbPattern, EmbThread, write_dst, STITCH, JUMP

import numpy as np
import cv2
from pyembroidery import EmbPattern, EmbThread, write_dst, STITCH, JUMP

def generate_dst(prediction, dst_file="StitchVision_Output.dst", target_width_mm=100):
    """
    Input: prediction = binary mask (0,255) or grayscale image
    Output: DST file path
    """

    # 1️⃣ Grayscale & uint8
    if prediction.ndim == 3:
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
    if prediction.dtype != np.uint8:
        prediction = (prediction * 255).astype(np.uint8)

    # 2️⃣ Thresholding (binary mask)
    _, binary = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

    # 3️⃣ Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    h, w = binary.shape
    scale = (target_width_mm * 10) / w  # DST unit = 0.1mm
    stitch_length = 25  # 2.5mm per stitch for outlines

    pattern = EmbPattern()
    pattern.add_thread(EmbThread())

    for contour in contours:
        if len(contour) < 2:
            continue

        # JUMP to start of contour
        start_pt = contour[0][0]
        pattern.add_stitch_absolute(JUMP, int(start_pt[0] * scale), int(start_pt[1] * scale))

        # Follow contour
        for i in range(1, len(contour)):
            p1 = contour[i-1][0]
            p2 = contour[i][0]

            dx = (p2[0] - p1[0]) * scale
            dy = (p2[1] - p1[1]) * scale
            dist = np.sqrt(dx**2 + dy**2)

            if dist > stitch_length:
                steps = max(int(dist // stitch_length), 1)
                for s in range(1, steps + 1):
                    frac = s / steps
                    ix = int((p1[0] + (p2[0] - p1[0]) * frac) * scale)
                    iy = int((p1[1] + (p2[1] - p1[1]) * frac) * scale)
                    pattern.add_stitch_absolute(STITCH, ix, iy)
            else:
                pattern.add_stitch_absolute(STITCH, int(p2[0] * scale), int(p2[1] * scale))

    pattern.end()
    write_dst(pattern, dst_file)
    print(f"✅ DST file generated: {dst_file}")
    return dst_file

