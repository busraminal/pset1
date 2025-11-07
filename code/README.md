#  Automatic Image Alignment & Enhancement Pipeline  
### by **Büşra Mina AL**  

---

## 🖼️ Önizleme (Before → After)

| Girdi (Raw Channels) | Hizalanmış ve İyileştirilmiş Görüntü |
|----------------------|--------------------------------------|
| ![Input](results/preview_before.png) | ![Output](results/preview_comparison.png) |

> ✨ *Prokudin-Gorskii tarzı cam negatif görüntülerin renkli yeniden doğuşu.*  
> 📷 “Renk sadece ışığın oyunu değil, mühendisliğin şiiridir.”

---

## 📖 Proje Özeti

Bu proje, tarihî **Prokudin-Gorskii** tarzı üç-kanallı (RGB) siyah-beyaz cam negatif görüntüleri **otomatik hizalayan**, **renklendiren** ve **iyileştiren** bir görüntü işleme sistemidir.  
Ana algoritma, **Normalized Cross-Correlation (NCC)** ve **Image Pyramid** tabanlı çok katmanlı hizalama yöntemlerine dayanır.

Son aşamada, görüntü kalitesini artırmak için aşağıdaki teknikler uygulanır:  
**CLAHE (lokal kontrast artırma), Gamma Correction, Laplacian Sharpening** ve **HSV renk doygunluğu iyileştirmesi.**

Sonuç: 📷 *yıpranmış, hizasız cam negatiflerden modern, yüksek kontrastlı renkli fotoğraflar.*

---

## ⚙️ Proje Yapısı

```
project_root/
│
├── code_e/
│   ├── alignment.py           # NCC + Pyramid tabanlı hizalama algoritmaları
│   ├── utils_e.py             # Yardımcı fonksiyonlar (kanal ayırma, auto-crop, sonuç kaydetme)
│   ├── enhance_image.py       # Görüntü iyileştirme (CLAHE, Gamma, Sharpness, HSV)
│   └── __init__.py
│
├── run_alignment_pipeline.py  # Çoklu (batch) hizalama + otomatik crop pipeline
├── results/                   # Çıktıların kaydedildiği klasör
└── README.md
```

---

## 🚀 Çalışma Akışı

1️⃣ **Görüntü Girişi:** Üçlü kanal (B-G-R) içeren cam negatif resim alınır  
2️⃣ **Kırpma:** Gürültülü kenarlar belirli oranda kırpılır  
3️⃣ **Hizalama:** NCC + Image Pyramid ile çok seviyeli hizalama yapılır  
4️⃣ **Otomatik Kenar Kırpma:** Siyah çerçeveler dinamik olarak kaldırılır  
5️⃣ **İyileştirme:** CLAHE + Gamma + Sharpen + HSV ile son dokunuşlar yapılır  
6️⃣ **Sonuç:** Renkli, hizalanmış ve temizlenmiş görsel kaydedilir  

---

## 💡 Kullanım

### 🧩 Komut Satırından

```bash
python run_alignment_pipeline.py
```

Ardından klasör yolu gir:  
```
📂 Lütfen hizalanacak görüntülerin bulunduğu klasörü giriniz: data/
```

Tüm görüntüler paralel olarak hizalanır ve `results/` klasörüne kaydedilir.  

---

### 🔁 Çoklu Worker (Parallel Processing)

Yeni sistem artık aynı anda **birden fazla görüntüyü** işler ⚙️  
Python’un `ThreadPoolExecutor` yapısı sayesinde 2–4 worker eşzamanlı çalışır.  
Bu sayede işlem süresi ortalama %60–70 kısalır ⏱️  

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.submit(process_single_image, path)
```

---

### Varsayılan Parametreler (Dinamik Olarak Ayarlanır)

| Parametre | Açıklama | Aralık |
|------------|-----------|--------|
| `search_range` | NCC arama penceresi boyutu | 15–35 |
| `crop_percent` | Kırpılacak kenar yüzdesi | 3–6% |
| `pyramid_depth` | Görüntü piramidi derinliği | 6–10 |

> Bu değerler otomatik olarak görüntü boyutuna göre optimize edilir 💡  

---

## 🧪 Örnek Kod

```python
from code_e.utils_e import goruntu_hizalama, save_and_display_results, auto_border_crop
from code_e.alignment import Image_pyramid
from code_e.enhance_image import enhance_image
import skimage.io as skio
import numpy as np

# Görüntü yükle
img = skio.imread("data/church.tif")

# Kanal ayırma
r, b, g = goruntu_hizalama(img, crop_amount=4)

# NCC + Pyramid hizalama
g_aligned, r_aligned, _, _ = Image_pyramid(r, g, b, depth=10, search_range=35)

# RGB birleştir
rgb = np.dstack((r_aligned, g_aligned, b))

# Otomatik kenar kırpma (bonus)
rgb_cropped = auto_border_crop(rgb)

# Görüntü iyileştirme
enhanced = enhance_image(rgb_cropped)

# Kaydet ve göster
save_and_display_results(enhanced, "church_final", 35, 0, [0,0], [0,0])
```

---

## 📊 Performans ve Zaman Ölçümleri

| Görsel Adı | Boyut | Piramit Derinliği | Worker | Süre (s) | Kayma (R/G) |
|-------------|--------|------------------|---------|-----------|--------------|
| church.tif  | 1024×1024 | 10 | 4 | 12.84 | [10, -4] / [5, -2] |
| cathedral.tif | 2048×2048 | 10 | 4 | 18.21 | [14, -5] / [7, -3] |
| bridge.tif | 1500×900 | 8 | 4 | 10.65 | [9, -2] / [4, -1] |
| portrait.tif | 800×800 | 6 | 2 | 8.13 | [6, -1] / [3, 0] |

> ⏱️ Ortalama hızlanma: **~1.7×** (paralel işleme sayesinde)  
> 💡 Otomatik kırpma + iyileştirme, kaliteyi artırırken süreye ortalama +2s ekler.

---

## 🧠 Teknik Özellikler

- **Kütüphaneler:** `numpy`, `matplotlib`, `scikit-image`, `opencv-python`
- **Hizalama Metrikleri:** NCC (Ana) & SSD (Alternatif)
- **Desteklenen Formatlar:** `.tif`, `.jpg`, `.png`, `.bmp`, `.jp2`
- **Python Sürümü:** ≥ 3.10
- **Parallelization:** `concurrent.futures` ile multi-thread
- **Bonus:** `auto_border_crop()` ile kenar temizleme

---

## 🔮 Gelecek Planları

- 🧠 CLAHE ve Gamma için otomatik optimizasyon  
- ⚡ GPU hızlandırmalı piramit hizalama (CuPy / PyTorch)  
- 🤖 Semantik hizalama (AI destekli kanal eşleme)  
- 📦 GUI / Web Arayüzü: Kullanıcı dostu hizalama platformu  

---

## 🪄 Yazar Notu

> © 2025 — **Büşra Mina AL**  
> 🎓 Yapay Zeka Mühendisliği   & Endüstri Mühendisliği  
> 🏫 Ostim Teknik Üniversitesi  
> 💻 GitHub: [busraminal](https://github.com/busraminal)  

