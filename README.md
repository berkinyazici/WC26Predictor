# 🏆 WC26 Predictor - FIFA World Cup 2026 Prediction System

Makine öğrenmesi kullanarak 2026 FIFA Dünya Kupası'nın tüm maç sonuçlarını tahmin eden sistem.

## 📋 Proje Özeti

- **Hedef**: Group stage'den finale kadar tüm maçları tahmin etme
- **Veri Kaynağı**: Kaggle FIFA World Cup 2026 Prediction System
- **Teknoloji**: Python, scikit-learn, pandas, TensorFlow

## 📁 Dosya Yapısı

```
WC26 Predictor/
├── data/                    # Kaggle dataseti
├── notebooks/               # Analiz ve experiment notebooks
├── src/
│   ├── data_loader.py      # Veri yükleme ve preprocessing
│   ├── models.py           # ML modeller
│   ├── tournament.py       # Turnuva simulatörü
│   └── utils.py            # Yardımcı fonksiyonlar
├── predictions/            # Tahmin sonuçları
└── requirements.txt        # Bağımlılıklar
```

## 📊 Dataset İçeriği

Kaggle dataset'i aşağıdakileri içerir:
- Tarihsel maç verileri (geçmiş WC turnuvaları)
- Oyuncu istatistikleri
- Takım performans metrikleri
- 2026 WC grup atışmaları ve turnuva bracket'i

## 🎯 Proje Aşamaları

1. **Veri Analizi** - Dataset yapısı ve içeriği
2. **Model Geliştirme** - Match outcome prediction
3. **Tournament Simulation** - Grup → Finalde tahminler
4. **Visualization** - Sonuçların görsele dönüştürülmesi
5. **Validation** - Model doğruluğunun test edilmesi
