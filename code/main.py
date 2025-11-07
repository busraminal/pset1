# ============================================================
# main.py veya run_alignment_pipeline.py
# Paralel çoklu worker (multi-thread / multi-process) pipeline
# SSD / NCC yöntem seçimi destekli
# ============================================================

import time
import os
import numpy as np
import skimage.io as skio
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_e.utils_e import goruntu_hizalama, save_and_display_results, auto_border_crop
from code_e.alignment import Image_pyramid, SSD  # SSD fonksiyonunu da ekledik


# ============================================================
#  GLOBAL KAYMA LİSTELERİ
# ============================================================
Green_shift_X = []
Green_shift_Y = []
Red_shift_X = []
Red_shift_Y = []


# ============================================================
#  DİNAMİK PARAMETRE OPTİMİZASYONU
# ============================================================
def optimize_params(image):
    """ Görüntü boyutuna göre otomatik parametre ayarı. """
    h, w = image.shape[:2]

    # Arama aralığı
    if max(h, w) < 800:
        search = 15
    elif max(h, w) < 2000:
        search = 25
    else:
        search = 35

    # Kırpma oranı
    crop = 6 if h < 800 else 4 if h < 1500 else 3

    # Piramit derinliği
    depth = 6 if h < 800 else 8 if h < 1500 else 10

    return search, crop, depth


# ============================================================
#  TEK GÖRÜNTÜ İŞLEYİCİ (SSD/NCC SEÇİMLİ)
# ============================================================
def process_single_image(image_path, method="NCC"):
    """ Tek bir görüntü üzerinde hizalama + otomatik kırpma işlemi yapar. """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n--- {image_name} için hizalama başlatılıyor ({method}) ---")

    start_time = time.time()
    image_Z = skio.imread(image_path)
    search, crop, depth = optimize_params(image_Z)

    red_channel, blue_channel, green_channel = goruntu_hizalama(image_Z, crop)

    # =======================
    # HİZALAMA YÖNTEMİ SEÇİMİ
    # =======================
    if method.upper() == "SSD":
        green_shift_final, red_shift_final, red_displacement, green_displacement = \
            SSD(red_channel, green_channel, blue_channel, search)
    else:
        green_shift_final, red_shift_final, red_displacement, green_displacement = \
            Image_pyramid(red_channel, green_channel, blue_channel, depth, search)

    # RGB birleştir
    im_out = np.dstack((red_shift_final, green_shift_final, blue_channel))

    # BONUS: otomatik kenar kırpma
    im_out = auto_border_crop(im_out)

    total_time = time.time() - start_time
    total_g_shift = [sum(Green_shift_X), sum(Green_shift_Y)]
    total_r_shift = [sum(Red_shift_X), sum(Red_shift_Y)]

    # Kaydet — yöntem adıyla birlikte
    save_and_display_results(im_out, image_name, search, total_time, total_r_shift, total_g_shift, method_name=method)

    print(f"  {image_name} tamamlandı! ({method}, {total_time:.2f} sn)")
    return image_name


# ============================================================
#  PARALEL ÇALIŞTIRMA BLOĞU
# ============================================================
if __name__ == "__main__":
    folder_path = input("  Lütfen hizalanacak görüntülerin bulunduğu klasörü giriniz: ").strip()

    # Kullanıcıdan hizalama yöntemi seçtir
    method_choice = input("  Hangi hizalama yöntemi kullanılacak? (NCC / SSD): ").strip().upper()
    if method_choice not in ["NCC", "SSD"]:
        print("  Geçersiz seçim! Varsayılan olarak NCC kullanılacak.")
        method_choice = "NCC"

    supported_exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(supported_exts)]

    if not image_files:
        print("  Klasörde hizalanacak uygun görüntü bulunamadı.")
        exit(0)

    print(f"\n  {len(image_files)} görüntü bulundu. Paralel işlem başlıyor...\n")

    # CPU çekirdeği kadar worker aç (maksimum 4)
    max_workers = min(4, os.cpu_count() or 2)
    print(f"  Kullanılan worker sayısı: {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, path, method_choice): path for path in image_files}

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"  {result} başarıyla işlendi.")
            except Exception as e:
                print(f"  Hata oluştu: {e}")

    print(f"\n Tüm görüntüler paralel olarak işlendi! ({method_choice})")
