# ============================================================
# utils_e.py 
# Görüntü hizalama (alignment) için yardımcı fonksiyonlar
# Thread-safe ve GUI’siz versiyon (çoklu işlem uyumlu)
# ============================================================

import numpy as np
import skimage.io as skio
import os


# ============================================================
#  GLOBAL KAYMA DEĞİŞKENLERİ (İzleme ve loglama için)
# ============================================================
Green_shift_X = []
Green_shift_Y = []
Red_shift_X = []
Red_shift_Y = []


# ============================================================
#  GÖRÜNTÜ HAZIRLAMA VE KANAL AYIRMA
# ============================================================
def goruntu_hizalama(image_Z, crop_amount):
    """
    Renkli görüntüyü kırpar ve 3 kanala (R, G, B) ayırır.
    
    Args:
        image_Z (ndarray): Giriş görüntüsü (ör. Prokudin-Gorskii tarzı üç kanallı negatif)
        crop_amount (float): Görüntü kenarlarından kırpılacak yüzde oranı (%)
    
    Returns:
        (red_channel, blue_channel, green_channel): Kırpılmış renk kanalları
    """

    # --- Görüntü boyutlarını al ---
    original_height, original_width = image_Z.shape[:2]

    # Görüntü dikeyde 3 eşit parçaya bölünür
    one_third = original_height // 3

    # Her kanal ayrı bölgeden alınır
    blue_channel = image_Z[:one_third]
    green_channel = image_Z[one_third:2 * one_third]
    red_channel = image_Z[2 * one_third:3 * one_third]

    # --- Kenarlardan kırpma miktarını hesapla ---
    crop_h = int(one_third * crop_amount / 100)
    crop_w = int(original_width * crop_amount / 100)

    # --- Kırpma işlemi ---
    red_channel = red_channel[crop_h:-crop_h, crop_w:-crop_w]
    green_channel = green_channel[crop_h:-crop_h, crop_w:-crop_w]
    blue_channel = blue_channel[crop_h:-crop_h, crop_w:-crop_w]

    # --- Sonuç ---
    return red_channel, blue_channel, green_channel


# ============================================================
#  SONUÇLARI KAYDET (GUI YOK — Thread-safe, Yöntem Adı Dahil)
# ============================================================
def save_and_display_results(im_out, name, search, total_time, total_r_shift, total_g_shift, method_name="NCC"):
    """
    Hizalanmış görüntüyü kaydeder (ekranda göstermez).
    Thread veya multiprocessing ortamında güvenli çalışır.

    Args:
        im_out (ndarray): Hizalanmış renkli görüntü
        name (str): Görüntü adı (dosya etiketi)
        search (int): Kullanılan arama aralığı
        total_time (float): Toplam hizalama süresi (saniye)
        total_r_shift (list): Kırmızı kanal toplam kayması [x, y]
        total_g_shift (list): Yeşil kanal toplam kayması [x, y]
        method_name (str): Kullanılan algoritma adı ("SSD" veya "NCC")
    """

    # --- Görüntüyü normalize et ve 8-bit’e dönüştür ---
    im_out_norm = im_out / np.max(im_out)
    im_out_8bit = (im_out_norm * 255).astype(np.uint8)

    # --- Sonuç klasörünü oluştur ---
    os.makedirs("results", exist_ok=True)

    # --- Dosya adı oluştur ---
    fname = f"results/{name}_{method_name}_Pyramid_{search}.jpg"

    # --- Görseli diske kaydet ---
    skio.imsave(fname, im_out_8bit)

    # --- Konsol çıktısı ---
    print("\n--- SONUÇLAR ---")
    print(f" Algoritma: {method_name}")
    print(f" Toplam süre = {total_time:.4f} saniye")
    print(f" Red channel total shift = {total_r_shift}")
    print(f" Green channel total shift = {total_g_shift}")
    print(f" Görsel kaydedildi: {fname}")


# ============================================================
#  BONUS: OTOMATİK KENAR KIRPMA (10 PUAN)
# ============================================================
def auto_border_crop(img, threshold_ratio=0.10, min_crop_pixels=5):
    """
    Otomatik Kenar Kırpma Fonksiyonu
    Görüntü kenarlarındaki siyah çerçeveleri ve hizalama artifaktlarını 
    tespit eder ve otomatik olarak kırpar.

    Analiz prensibi:
    - Kenar piksellerinin yoğunluk ortalamasını hesaplar.
    - Belirli bir eşiğin altındaki bölgeleri dört kenardan ayrı ayrı kırpar.

    Args:
        img (ndarray): RGB veya gri görüntü
        threshold_ratio (float): 0–1 arası kırpma eşiği oranı 
                                 (yüksek değer = daha agresif kırpma)
        min_crop_pixels (int): Her kenardan en az kırpılacak piksel miktarı

    Returns:
        cropped_img (ndarray): Kırpılmış yeni görüntü
    """

    # --- Griye dönüştür ---
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img.copy()

    # --- Normalize et (0–1 arası) ---
    gray = gray.astype(np.float32)
    gray /= np.max(gray) if np.max(gray) != 0 else 1

    h, w = gray.shape
    threshold = threshold_ratio * np.mean(gray)

    # --- Başlangıç limitleri ---
    top, bottom, left, right = 0, h, 0, w

    # --- Üst kenar analizi ---
    for i in range(h // 4):
        if np.mean(gray[i, :]) < threshold:
            top = i + min_crop_pixels
        else:
            break

    # --- Alt kenar analizi ---
    for i in range(h - 1, (3 * h) // 4, -1):
        if np.mean(gray[i, :]) < threshold:
            bottom = i - min_crop_pixels
        else:
            break

    # --- Sol kenar analizi ---
    for j in range(w // 4):
        if np.mean(gray[:, j]) < threshold:
            left = j + min_crop_pixels
        else:
            break

    # --- Sağ kenar analizi ---
    for j in range(w - 1, (3 * w) // 4, -1):
        if np.mean(gray[:, j]) < threshold:
            right = j - min_crop_pixels
        else:
            break

    # --- Limitleri güvenli hale getir ---
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    # --- Görüntüyü kırp ---
    cropped = img[top:bottom, left:right]
    print(f"   Otomatik kırpma uygulandı → Yeni boyut: {cropped.shape}")
    return cropped
