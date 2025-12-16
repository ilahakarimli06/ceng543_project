# Grid Search Deney Matrisi

## 🔥 YENİ DURUM: SADECE OTOMATİK GRID

**Elle yazılmış eski yml dosyaları silindi.**

Artık sadece otomatik olarak oluşturulan 27 config dosyası kullanılıyor:

- Hepsi: `configs/sliding/grid/` klasöründe
- Her biri: `w{window}_ov{overlap}_g{global}.yml` formatında
- Elle config eklemeye gerek yok, hepsi script ile üretiliyor

---

## 🚀 Grid Search Akışı

1. **Config dosyalarını oluştur:**
   ```bash
   python scripts/generate_grid_configs.py
   ```
2. **Tüm grid'i çalıştır:**
   ```bash
   .\scripts\run_full_grid.ps1
   ```
3. **Sonuçları analiz et:**
   ```bash
   python scripts/analyze_grid_results.py
   ```

---

## Grid Parametreleri ve Kombinasyonlar

- window_size: 512, 1024, 2048
- overlap: 0, 0.25W, 0.5W (her window için mutlak değere çevrilir)
- global_tokens: 0, 16, 64

Toplam: 3 × 3 × 3 = 27 kombinasyon

---

## Neden sadece otomatik config?

- Tekrarlanabilirlik ve güncelleme kolaylığı
- Parametre değişince script ile tüm config'ler güncellenir
- Elle dosya yönetimi ve hata riski ortadan kalkar

---

## Beklenen Çıktı Metrikleri

Her deney için `results/` klasöründe CSV dosyaları oluşacak:

- **rougeL**: ROUGE-L F1 skoru (özet kalitesi)
- **bertscore_f1**: BERTScore F1 (semantik benzerlik)
- **latency**: İşlem süresi (saniye)
- **mem_mb**: GPU bellek kullanımı (MB)

Grid değişkenleri:

- model, attention, window, overlap, global_tokens

---

## Analiz Adımları

1. **Baseline belirleme**: Otomatik grid oluşturup baseline konfigürasyonu çalıştır (örnek: `configs/sliding/grid/w1024_ov256_g16.yml`)
2. **Window size etkisi**: w512, w1024, w2048 karşılaştır
3. **Overlap etkisi**: 0%, 25%, 50% karşılaştır
4. **Global tokens etkisi**: 0, 16, 64 karşılaştır
5. **Speed-up**: flash2 vs default karşılaştır

**Trade-off analizi:**

- Latency vs ROUGE-L (hız/kalite)
- Memory vs ROUGE-L (verimlilik/kalite)
- Window size vs overlap (en iyi kombinasyon)
