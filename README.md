# 🎬 Video Chef

**Video'daki karakteri ve arka planı değiştiren, %100 yerel çalışan AI pipeline.**

> LumaLabs'in yaptığını lokal, ücretsiz, gizli yapıyoruz.

---

## 🎯 Ne Yapıyor?

```
[Video] + [Karakter Görseli] + [Arka Plan Görseli]
                    ↓
         AI Segmentasyon + Compositing
                    ↓
     [Instagram-ready çıktı videosu] 🎉
```

**Örnek:**
- Girdi: Sokakta yürüyen insan videosu
- Karakter img: Anime karakteri / başka bir kişi
- Arka plan img: Uzay, orman, stüdyo
- Çıktı: Karakter değiştirilmiş, arka plan değiştirilmiş video ✅

---

## 🛠️ Teknoloji Stack (Hafif GPU için optimize)

| Görev | Araç | VRAM |
|-------|------|------|
| Video küçültme | **FFmpeg** | 0 (CPU) |
| Frame extraction | **FFmpeg** | 0 (CPU) |
| Segmentasyon (bg kaldırma) | **rembg (u2net_human_seg)** | ~1GB |
| Arka plan compositing | **OpenCV + Pillow** | 0 (CPU) |
| Karakter swap | **rembg + blending** | ~1GB |
| Video yeniden oluşturma | **FFmpeg** | 0 (CPU) |

**Toplam VRAM:** ~1-2GB (küçük GPU'da çalışır ✅)

---

## 📁 Proje Yapısı

```
video-chef/
├── README.md
├── requirements.txt
├── config.yaml                  # Pipeline ayarları
│
├── src/
│   ├── __init__.py
│   ├── video_processor.py       # Video küçültme + frame extraction
│   ├── segmenter.py             # Karakter/bg segmentasyonu (rembg)
│   ├── background_replacer.py   # Arka plan değiştirme
│   ├── character_replacer.py    # Karakter değiştirme
│   ├── compositor.py            # Katmanları birleştirme
│   └── video_assembler.py       # Frame'lerden video oluşturma
│
├── pipelines/
│   └── instagram_pipeline.py    # Ana pipeline (hepsini sırayla çalıştırır)
│
├── input/                       # Kullanıcı dosyaları buraya
│   ├── video.mp4                # Orijinal video
│   ├── character.png            # Yeni karakter görseli
│   └── background.png           # Yeni arka plan görseli
│
├── output/                      # Çıktılar buraya
│   └── result.mp4
│
├── temp/                        # Geçici dosyalar (otomatik temizlenir)
│   ├── frames/                  # Extracted frames
│   ├── masks/                   # Segmentasyon maskeleri
│   └── composited/              # Birleştirilmiş frame'ler
│
└── tests/
    └── test_pipeline.py
```

---

## 🚀 Kurulum

```bash
# 1. Repo'yu klonla
git clone https://github.com/sevkibaba/video-chef.git
cd video-chef

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Bağımlılıklar
pip install -r requirements.txt

# 4. FFmpeg (yoksa kur)
# Windows: https://ffmpeg.org/download.html
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 5. Test et
python pipelines/instagram_pipeline.py --test
```

---

## 🎬 Kullanım

```bash
python pipelines/instagram_pipeline.py \
  --video input/video.mp4 \
  --character input/character.png \
  --background input/background.png \
  --output output/result.mp4 \
  --format instagram-reels   # veya: instagram-square, instagram-story
```

---

## ⚙️ Konfigürasyon (config.yaml)

```yaml
video:
  target_resolution: 720        # 720p (Instagram için yeterli)
  target_fps: 30
  max_duration_seconds: 60      # 60 saniye max

segmentation:
  model: u2net_human_seg        # Hafif, insanlar için optimize
  device: cuda                  # cuda / cpu
  batch_size: 4                 # VRAM'e göre ayarla (az VRAM = 1-2)

compositing:
  blend_mode: seamless          # seamless / hard / soft
  edge_feather: 5               # Kenar yumuşatma (piksel)
  shadow: true                  # Hafif gölge ekle (daha gerçekçi)

output:
  format: mp4
  codec: h264
  quality: 23                   # CRF: düşük = kaliteli (18-28 arası)
  instagram_optimize: true      # Instagram için optimize et
```

---

## 📱 Instagram Formatları

| Format | Çözünürlük | Oran | Kullanım |
|--------|-----------|------|---------|
| **Reels** | 1080x1920 | 9:16 | Dikey video |
| **Square** | 1080x1080 | 1:1 | Kare video |
| **Story** | 1080x1920 | 9:16 | Hikaye |
| **Feed** | 1080x1350 | 4:5 | Feed post |

---

## 🗺️ Roadmap

### Phase 1: MVP ✅ (Şimdi)
- [ ] Video küçültme + frame extraction
- [ ] rembg ile segmentasyon
- [ ] Basit background replacement
- [ ] Video assembly
- [ ] Instagram export

### Phase 2: Karakter Swap 🔄
- [ ] Karakter segmentasyonu (kişiyi bul)
- [ ] Karakter görselini üzerine bindirme
- [ ] Edge blending (doğal görünüm)
- [ ] Hareket tracking (karakter videoyla hareket etsin)

### Phase 3: Kalite 🎨
- [ ] SAM2-tiny ile daha iyi segmentasyon
- [ ] Işık uyumu (karaktere arka planın ışığını uygula)
- [ ] Gölge ekleme
- [ ] Batch processing (çok video)

---

## ⚡ Performans (Tahmin)

| Video Süresi | GPU (RTX 3060) | CPU only |
|-------------|---------------|---------|
| 15 saniye | ~2 dk | ~8 dk |
| 30 saniye | ~4 dk | ~15 dk |
| 60 saniye | ~8 dk | ~30 dk |

*720p, 30fps için tahmin*

---

## 🔒 Gizlilik

- ✅ %100 lokal çalışır
- ✅ Hiçbir veri dışarı çıkmaz
- ✅ İnternet bağlantısı gerektirmez (kurulumdan sonra)
- ✅ Videolarınız sunucuya yüklenmez

---

*Video Chef - Cook your videos locally 🍳*
