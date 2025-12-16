"""
Grid Search Sonuçlarını Analiz Et
Tüm CSV dosyalarını birleştir ve en iyi konfigürasyonları bul
"""

import pandas as pd
import glob
from pathlib import Path

def analyze_grid():
    """Grid search sonuçlarını analiz et"""
    
    # Tüm results dosyalarını oku
    csv_files = glob.glob("results/grid/*.csv")
    
    if not csv_files:
        print("❌ Hiç sonuç dosyası bulunamadı!")
        return
    
    # Tüm sonuçları birleştir
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ {csv_file} okunamadı: {e}")
    
    if not dfs:
        print("❌ Hiç geçerli sonuç bulunamadı!")
        return
    
    # Birleştir
    all_results = pd.concat(dfs, ignore_index=True)
    
    # Özet istatistikler
    print("\n" + "="*60)
    print("GRID SEARCH SONUÇ ÖZETİ")
    print("="*60)
    print(f"\nToplam deney sayısı: {len(all_results)}")
    
    # Metrik özeti
    print("\n" + "-"*60)
    print("METRIK İSTATİSTİKLERİ")
    print("-"*60)
    metrics = ["rougeL", "bertscore_f1", "latency", "mem_mb"]
    print(all_results[metrics].describe())
    
    # En iyi sonuçlar (ROUGE-L'e göre)
    print("\n" + "-"*60)
    print("EN İYİ 5 KONFİGÜRASYON (ROUGE-L)")
    print("-"*60)
    top_rouge = all_results.nlargest(5, "rougeL")[
        ["window", "overlap", "global_tokens", "rougeL", "bertscore_f1", "latency", "mem_mb"]
    ]
    print(top_rouge.to_string(index=False))
    
    # En iyi sonuçlar (BERTScore'a göre)
    print("\n" + "-"*60)
    print("EN İYİ 5 KONFİGÜRASYON (BERTScore)")
    print("-"*60)
    top_bert = all_results.nlargest(5, "bertscore_f1")[
        ["window", "overlap", "global_tokens", "rougeL", "bertscore_f1", "latency", "mem_mb"]
    ]
    print(top_bert.to_string(index=False))
    
    # En hızlı konfigürasyonlar
    print("\n" + "-"*60)
    print("EN HIZLI 5 KONFİGÜRASYON (Latency)")
    print("-"*60)
    top_speed = all_results.nsmallest(5, "latency")[
        ["window", "overlap", "global_tokens", "rougeL", "bertscore_f1", "latency", "mem_mb"]
    ]
    print(top_speed.to_string(index=False))
    
    # En verimli (memory)
    print("\n" + "-"*60)
    print("EN VERİMLİ 5 KONFİGÜRASYON (Memory)")
    print("-"*60)
    top_memory = all_results.nsmallest(5, "mem_mb")[
        ["window", "overlap", "global_tokens", "rougeL", "bertscore_f1", "latency", "mem_mb"]
    ]
    print(top_memory.to_string(index=False))
    
    # Birleştirilmiş sonuçları kaydet
    output_file = "results/grid/COMBINED_RESULTS.csv"
    all_results.to_csv(output_file, index=False)
    print(f"\n✅ Birleştirilmiş sonuçlar kaydedildi: {output_file}")
    
    # Parametrelere göre grupla ve ortalama al
    print("\n" + "-"*60)
    print("PARAMETRE ETKİLERİ (Ortalamalar)")
    print("-"*60)
    
    print("\n--- Window Size Etkisi ---")
    window_effect = all_results.groupby("window")[metrics].mean()
    print(window_effect)
    
    print("\n--- Overlap Etkisi ---")
    overlap_effect = all_results.groupby("overlap")[metrics].mean()
    print(overlap_effect)
    
    print("\n--- Global Tokens Etkisi ---")
    global_effect = all_results.groupby("global_tokens")[metrics].mean()
    print(global_effect)
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas gerekli! Yüklemek için: pip install pandas")
        exit(1)
    
    analyze_grid()
