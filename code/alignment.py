# alignment.py 
# NCC ve SSD tabanlı çok seviyeli (pyramid) görüntü hizalama

import numpy as np
import skimage as sk
import skimage.transform
from code_e.utils_e import save_and_display_results, auto_border_crop


Green_shift_X, Green_shift_Y, Red_shift_X, Red_shift_Y = [], [], [], []


# ============================================================
#  SSD (Sum of Squared Differences)
# ============================================================
def SSD(red_channel, green_channel, blue_channel, search):
    """
    Kareler Farkının Toplamı (SSD) ile optimal kaymayı bulur.
    SSD minimum olduğunda en iyi hizalama elde edilir.
    """
    axis_shift_list = []
    for x_axis in range(-search, search + 1):
        for y_axis in range(-search, search + 1):
            axis_shift_list.append([x_axis, y_axis])

    # --- YEŞİL ---
    green_shift_list = [np.roll(green_channel, shift, axis=(0, 1)) for shift in axis_shift_list]
    SSD_scores = [np.sum((img - blue_channel) ** 2) for img in green_shift_list]
    min_index_g = np.argmin(SSD_scores)
    green_displacement = axis_shift_list[min_index_g]
    green_shift_final = np.roll(green_channel, green_displacement, axis=(0, 1))
    print(f" Green Channel Shift: {green_displacement}")

    # --- KIRMIZI ---
    red_shift_list = [np.roll(red_channel, shift, axis=(0, 1)) for shift in axis_shift_list]
    SSD_scores = [np.sum((img - blue_channel) ** 2) for img in red_shift_list]
    min_index_r = np.argmin(SSD_scores)
    red_displacement = axis_shift_list[min_index_r]
    red_shift_final = np.roll(red_channel, red_displacement, axis=(0, 1))
    print(f" Red Channel Shift: {red_displacement}")

    return green_shift_final, red_shift_final, red_displacement, green_displacement


# ============================================================
#  NCC (Normalized Cross-Correlation)
# ============================================================
def NCC(red_channel, green_channel, blue_channel, search):
    """
    NCC (Normalized Cross-Correlation) ile optimal kaymayı bulur.
    """
    axis_shift_list = []
    for x_axis in range(-search, search + 1):
        for y_axis in range(-search, search + 1):
            axis_shift_list.append([x_axis, y_axis])

    blue_norm = np.linalg.norm(blue_channel)
    blue_1D = np.ravel(blue_channel)

    # --- YEŞİL ---
    green_scores = []
    for shift in axis_shift_list:
        shifted = np.roll(green_channel, shift, axis=(0, 1))
        score = np.dot(np.ravel(shifted) / np.linalg.norm(shifted), blue_1D / blue_norm)
        green_scores.append(score)

    best_g = np.argmax(green_scores)
    green_displacement = axis_shift_list[best_g]
    green_shift_final = np.roll(green_channel, green_displacement, axis=(0, 1))

    # --- KIRMIZI ---
    red_scores = []
    for shift in axis_shift_list:
        shifted = np.roll(red_channel, shift, axis=(0, 1))
        score = np.dot(np.ravel(shifted) / np.linalg.norm(shifted), blue_1D / blue_norm)
        red_scores.append(score)

    best_r = np.argmax(red_scores)
    red_displacement = axis_shift_list[best_r]
    red_shift_final = np.roll(red_channel, red_displacement, axis=(0, 1))

    print(f" Green Channel Shift: {green_displacement}")
    print(f" Red Channel Shift: {red_displacement}")

    return green_shift_final, red_shift_final, red_displacement, green_displacement


# ============================================================
#  Görüntü Piramidi (SSD veya NCC)
# ============================================================
def Image_pyramid(red_chan, green_chan, blue_chan, depth, search_range, image_name="", method="SSD"):
    """
    Görüntü Piramidi ile çok seviyeli hizalama.
    method: "SSD" veya "NCC"
    """
    if depth == 0:
        disp = int(search_range / (2 ** 4))
        disp = max(disp, 1)
        func = SSD if method == "SSD" else NCC
        return func(red_chan, green_chan, blue_chan, disp)

    # Downscale
    red_small = sk.transform.rescale(red_chan, 0.5 ** depth, channel_axis=-1)
    green_small = sk.transform.rescale(green_chan, 0.5 ** depth, channel_axis=-1)
    blue_small = sk.transform.rescale(blue_chan, 0.5 ** depth, channel_axis=-1)

    disp = max(int(search_range / (2 ** (4 - depth))), 1)
    func = SSD if method == "SSD" else NCC
    g_sm, r_sm, r_shift, g_shift = func(red_small, green_small, blue_small, disp)

    # Upscale shift
    r_shift = [int(i * (2 ** depth)) for i in r_shift]
    g_shift = [int(i * (2 ** depth)) for i in g_shift]

    Green_shift_X.append(g_shift[0]); Green_shift_Y.append(g_shift[1])
    Red_shift_X.append(r_shift[0]); Red_shift_Y.append(r_shift[1])

    # Uygula
    red_roll = np.roll(red_chan, r_shift, axis=(0, 1))
    green_roll = np.roll(green_chan, g_shift, axis=(0, 1))

    # Recursive çağrı
    g_final, r_final, r_final_shift, g_final_shift = Image_pyramid(
        red_roll, green_roll, blue_chan, depth - 1, search_range, image_name, method
    )

    # Görüntüyü birleştir
    im_out = np.dstack((r_final, g_final, blue_chan))
    im_out = auto_border_crop(im_out, threshold_ratio=0.08)

    # Kaydet
    from time import time
    total_time = 0.0
    save_and_display_results(im_out, f"{image_name}_{method}", search_range, total_time, r_final_shift, g_final_shift)

    print(f" {image_name} ({method}) hizalandı ve kaydedildi.")
    return g_final, r_final, r_final_shift, g_final_shift
