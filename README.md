
<div align="center">

# 🎨 Prokudin-Gorskii: Tarihi Fotoğrafları Renge Döndürme  
## *SSD, NCC ve Multi-Scale Pyramid ile Otomatik Görüntü Hizalama*

**Büşra Mina AL**  
OSTİM Teknik Üniversitesi – Yapay Zeka Mühendisliği

</div>

---

## 🧠 Projenin Fikri

1900’lerin başında **Sergey Prokudin-Gorskii**, sahneleri **mavi, yeşil ve kırmızı filtreler** ile ayrı ayrı fotoğrafladı.  
Amaç, bu **üç gri tonlu cam negatiften** tekrar **orijinal renkli görüntüyü** elde etmek.

Problem şu:  
Her kanal **mikro hareketler** yüzünden **kaymış** → renk saçılması, gölgelenme, bulanıklık.

Biz ne yaptık?  
🔹 **SSD** ve **NCC** karşılaştırıldı  
🔹 **Multi-Scale Pyramid** ile hizalama hızlandırıldı  
🔹 **CLAHE + Gamma + Laplacian** ile görüntü iyileştirildi  
🔹 Otomatik kenar kırpma uygulanarak çirkin siyah çerçeveler temizlendi

---

## 🔧 Pipeline Akışı

```
Girdi (Gri Görüntü)
       ↓
Kanal Bölme (R / G / B)
       ↓
Kenar Temizleme ve Normalize
       ↓
Hizalama
  ├─ SSD
  └─ NCC (+ Pyramid)  ← ⭐ En iyi sonuç
       ↓
Görüntü İyileştirme (CLAHE + Gamma + Sharpen)
       ↓
Auto-Crop
       ↓
Çıktı (Renkli Görüntü)
```

---

## 📁 Proje Yapısı

```
pset1/
│ code/
│   ├ alignment.py   → SSD / NCC / Pyramid
│   ├ enhancement.py → CLAHE / Gamma / Sharpen
│   ├ main.py        → pipeline
│   └ utils_e.py     → yardımcı fonksiyonlar
│ data/              → giriş fotoğrafları
│ results/           → çıktı görselleri ✅
│ requirements.txt
│ README.md
```

---

## 🏁 Çalıştırma

```bash
pip install -r requirements.txt
python code/main.py
```

---

## 🔍 Yöntemler Karşılaştırması

| Yöntem | Ortalama Hata | Stabilite | Açıklama |
|------|------|------|------|
| **SSD** | Daha yüksek | Orta | Hızlı ama ışık değişiminde bozuluyor |
| **NCC** | **En düşük** ✅ | **Yüksek** ✅ | Özellikle yüksek kontrastta tutarlı |
| **NCC + Pyramid** | **Şampiyon** 🏆 | **En stabil** | Hem doğru hem hızlı |

---

## 🎯 Sonuçlardan Galeri (Dosyaları repo'ya yükledikten sonra otomatik görünür)

<div align="center">

| Orijinal | Renkli |
|--------|--------|
| `data/00106v.jpg` | `results/00106v_Pyramid_25.jpg` |
| `data/00056v.jpg` | `results/00056v_Pyramid_25.jpg` |
| `data/00398v.jpg` | `results/00398v_Pyramid_25.jpg` |

</div>

---

## 🚀 Gelecek Geliştirmeler
- CUDA ile NCC’yi GPU’da hızlandırma
- SIFT + RANSAC ile feature-based hizalama
- Parametre seçimi için Bayesian optimizasyonu

---

<div align="center">



</div>
