# Video Chef - Teknik Özet

## 🚀 Pipeline'lar

### V1: Basic (character overlay)
- Basit overlay
- rembg ile segmentasyon
- Karakteri bounding box'a göre yerleştirme
- Sorun: yanlış pozisyonlama, yamulma

### V2: Improved (background replacement)
- Arka plan değiştirme eklendi
- Edge feathering iyileştirildi
- Sorun: karakter hala doğru oturmuyor

### V3: Contour-based (EN İYİ ÇALIŞAN)
- Contour detection ile beden algılama
- Daha doğru bounding box
- İyi sonuçlar
- Dosya: `pipelines/character_swap_v3.py`

### V4: Pose-aware (TAMAMLANMADI)
- Advanced geometric analysis
- Multiple threshold levels
- Hata: boyutlandırma sorunları

### V5: SAM Segmentation
- Meta's Segment Anything Model
- Çok daha iyi maskeleme
- Çalışıyor ama hala overlay sorunu
- Dosya: `pipelines/character_swap_v5.py`

### V6: AI Presenter (YENİ)
- Karakteri video içine "yapıştırmak" yerine
- Tamamen yeni frame oluştur
- Benzer framing, orijinal audio korunur
- Breathing animasyonu (opsiyonel)
- Dosya: `pipelines/ai_presenter_v6.py`

---

## 🎯 Kullanım

### V3 (En iyi - karakter değiştirme):
```bash
python pipelines/character_swap_v3.py \
  --video input/video.mp4 \
  --character input/character.png \
  --output output/result.mp4 \
  --max-frames 30
```

### V6 (Yeni - AI Presenter tarzı):
```bash
python pipelines/ai_presenter_v6.py \
  --video input/video.mp4 \
  --character input/character.png \
  --background input/bg.png \
  --output output/presenter.mp4 \
  --position center \
  --scale 0.7
```

---

## 📦 Dosyalar

```
video-chef/
├── README.md                 # Ana dokümantasyon
├── config.yaml               # Pipeline ayarları
├── requirements.txt          # Python bağımlılıkları
├── Dockerfile               # Docker CPU/GPU
├── docker-compose.yml       # Docker Compose
│
├── pipelines/
│   ├── instagram_pipeline.py      # Orijinal pipeline
│   ├── character_swap_v3.py        # En iyi sonuç (contour-based)
│   ├── character_swap_v5.py       # SAM segmentation
│   └── ai_presenter_v6.py         # Yeni yaklaşım
│
├── src/
│   ├── video_processor.py    # Video işleme
│   ├── segmenter.py          # rembg entegrasyonu
│   ├── compositor.py         # Frame birleştirme
│   └── video_assembler.py    # Video oluşturma
│
└── models/
    └── sam/                  # SAM weights (375MB)
```

---

## 🧪 Test Sonuçları

| Pipeline | Çalışıyor | Kalite | Not |
|----------|-----------|--------|-----|
| V1 | ✅ | 🟡 | Basit |
| V2 | ✅ | 🟡 | Orta |
| V3 | ✅ | 🟢 | **En iyi** |
| V4 | ❌ | - | Tamamlanmadı |
| V5 | ✅ | 🟢 | SAM daha iyi mask |
| V6 | 🔄 | 🆕 | Test edilmeli |

---

## 📝 Sonraki Adımlar

1. V6'yı test et
2. Frame'lerin animasyonunu geliştir
3. Daha fazla pozisyon seçeneği ekle
4. Ses+M synchronize et (wav2lip benzeri)

---

*Son güncelleme: 2026-03-23*