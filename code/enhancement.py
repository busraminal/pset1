# image_enhancement.py 
# Görüntü iyileştirme (enhancement) işlemleri için OpenCV tabanlı pipeline


import cv2
import numpy as np

def enhance_image(img):
    """
    💫 Görüntü iyileştirme pipeline’ı:
    - LAB renk uzayında CLAHE
    - Gamma düzeltmesi
    - Laplasyen keskinleştirme
    - HSV uzayında doygunluk artırma
    - Normalizasyon
    """
    img = img.copy()  # Orijinal görüntüyü bozmadan çalış 

    # ============================================================
    #  LAB renk uzayında CLAHE (Kontrast Limitli Uyarlamalı Histogram Eşitleme)
    # ============================================================
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #  Amaç: düşük kontrastlı alanlarda detayları belirginleştirmek

    # ============================================================
    #  Gamma düzeltmesi (aydınlık/doygunluk dengesi)
    # ============================================================
    gamma = 1.1  # >1: daha parlak 🎇
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    # 💡 LUT = Look-Up Table → her pikseli gamma eğrisine göre yeniden eşleştirir

    # ============================================================
    #  Laplasyen keskinleştirme (edge boost)
    # ============================================================
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    #  Kenar kontrastını artırır, detaylar daha net görünür

    # ============================================================
    #  HSV uzayında doygunluk (saturation) iyileştirme
    # ============================================================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.equalizeHist(s)  # sadece doygunluk kanalını güçlendiriyoruz 💥
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #  Renkler daha canlı ama doğal kalır

    # ============================================================
    #  Normalizasyon (0-255 arası yeniden ölçekleme)
    # ============================================================
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # 🎚 Piksel değerlerini dengeler, nihai kontrast stabil hale gelir

    return img  
