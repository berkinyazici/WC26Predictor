# WC26 Predictor

FIFA World Cup 2026 için gerçek fikstür üzerinden grup aşamasından finale kadar turnuva simülasyonu yapan makine öğrenmesi projesi.

Proje artık yalnızca tek maç kazananı tahmin etmekle kalmıyor; resmi grup fikstürünü, maç tarihlerini, venue bilgilerini, Round of 32 slotlarını ve eleme ağacını kullanarak 1000 Monte Carlo simülasyonu üretiyor. Ayrıca bu simülasyonlar içinden en yüksek likelihood'a sahip tek turnuva akışını seçip doldurulmuş bir tournament tree görseli oluşturuyor.

## Mevcut Özellikler

- Gerçek WC26 fikstürü: `data/wc26_real_fixtures.csv`
- 12 grup ve 72 grup maçı gerçek tarih/venue bilgisiyle simüle edilir.
- Match 73-104 arasındaki resmi knockout slotları çözülür.
- Grup aşamasında beraberlik, puan, gol averajı, atılan gol ve en iyi üçüncüler hesaplanır.
- Eleme maçlarında beraberlikler extra time/penalties kararıyla çözülür.
- 1000 Monte Carlo simülasyonundan takım bazlı ilerleme olasılıkları çıkarılır.
- En yüksek likelihood'a sahip tek simülasyon ayrıca seçilir.
- Final report ve görseller otomatik üretilir.

## Model Özeti

Model tarafında mevcut `WC26EnsembleModel` kullanılır:

- XGBoost classifier
- Random Forest classifier
- 60/40 ensemble blend
- Isotonic probability calibration
- Engineered feature set: strength, attack potency, defensive solidity, squad quality, form consistency, possession dominance, star power

Turnuva simülasyonunda modelin takım skorları pairwise match probability'ye dönüştürülür. Skorlar Poisson tabanlı bir maç motoruyla üretilir.

Not: Bazı gerçek WC26 takımları Kaggle veri setinde bulunmadığı için bu takımlar için konfederasyon ve FIFA rank benzeri metadata ile fallback feature satırları oluşturulur. Bu durum final report'ta model caveat olarak belirtilir.

## Önemli Çıktılar

Final rapor:

- `reports/final_report.md`

Simülasyon tabloları:

- `outputs/tournament/stage_probabilities.csv`
- `outputs/tournament/best_group_matches.csv`
- `outputs/tournament/best_knockout_matches.csv`
- `outputs/tournament/feature_importance.csv`

Görseller:

- `outputs/tournament/figures/champion_probabilities.png`
- `outputs/tournament/figures/stage_probabilities.png`
- `outputs/tournament/figures/group_tables.png`
- `outputs/tournament/figures/tournament_bracket.png`
- `outputs/tournament/figures/tournament_tree_best.png`
- `outputs/tournament/figures/feature_importance.png`

## Son Simülasyon Sonuçları

1000 Monte Carlo çalıştırmasında:

- En yüksek şampiyonluk olasılığı: Argentina, yaklaşık 12.2%
- En yüksek likelihood'a sahip tek simülasyonun şampiyonu: France
- En iyi tek simülasyon finali: Algeria 0-3 France

Bu iki sonuç farklı olabilir: Monte Carlo olasılığı tüm koşuların dağılımını gösterir, best single simulation ise 1000 koşu içindeki en olası komple turnuva yoludur.

## Dosya Yapısı

```text
WC26 Predictor/
├── data/
│   ├── wc26_real_fixtures.csv       # Gerçek WC26 fikstürü
│   └── fifa_wc2026_pipeline.py
├── models/
│   ├── ensemble_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
├── outputs/
│   └── tournament/
│       ├── best_group_matches.csv
│       ├── best_knockout_matches.csv
│       ├── stage_probabilities.csv
│       └── figures/
├── processed_data/
├── predictions/
├── reports/
│   └── final_report.md
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── match_features.py
│   ├── models.py
│   ├── tournament.py
│   ├── validation.py
│   └── visualization.py
├── generate_final_report.py
└── run_cv_validation.py
```

## Çalıştırma

Bağımlılıklar:

```bash
pip install -r requirements.txt
```

Final report ve tüm turnuva görsellerini üretmek:

```bash
python3 generate_final_report.py --simulations 1000
```

Syntax kontrolü:

```bash
PYTHONPYCACHEPREFIX=.pycache_tmp python3 -m compileall src generate_final_report.py
```

Cross-validation pipeline:

```bash
python3 run_cv_validation.py
```

## Geliştirme Notları

Bir sonraki model geliştirme adımı, takım bazlı winner label yerine gerçek match-level veriyle `home_win / draw / away_win` veya skor dağılımı modeli eğitmek olmalı. Böylece grup beraberlikleri, expected goals ve knockout sonuçları doğrudan modelden üretilebilir.
