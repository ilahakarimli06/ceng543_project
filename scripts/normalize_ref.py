import re
import unicodedata
import json
import sys
from pathlib import Path
from typing import Optional

class TextNormalizer:
    """
    Metin/özet temizleme yardımcıları.
    - Formül (LaTeX/sembol) içeren satırlarda Unicode normalizasyonunu atlar.
    - Diğer durumlarda NFKC ile normalize eder.
    """

    # LaTeX/Math/sembol ipuçları: $, \frac, \sum, \int, Yunan harfleri, ∞ ≈ ≠ ≤ ≥ → ← vb.
    _MATH_PATTERN = re.compile(r"(\$.*?\$|\\frac|\\sum|\\int|[α-ωΑ-Ω∞≈≠≤≥→←])")

    @staticmethod
    def normalize_safe(s: Optional[str]) -> str:
        """
        Formülleri koruyarak temizler.
        - \n ve \r -> boşluk
        - Çoklu boşluk -> tek boşluk
        - Formül varsa: Unicode normalization uygulanmaz
        - Formül yoksa: NFKC normalization uygulanır
        """
        if not isinstance(s, str):
            return ""
        if TextNormalizer._MATH_PATTERN.search(s):
            # Sadece boşluk/satır sonu temizliği
            s = s.replace("\n", " ").replace("\r", " ")
            s = re.sub(r"\s+", " ", s.strip())
            return s
        # Formül yoksa tam temizlik (NFKC + boşluklar)
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s.strip())
        return s


def process_file(input_path: Path, output_path: Path):
    """
    JSONL dosyasını okur, SADECE 'ref' alanını normalize eder,
    diğer alanları olduğu gibi bırakır ve yeni dosyaya yazar.
    """
    if not input_path.exists():
        print(f"[ERR] Dosya bulunamadı: {input_path}", file=sys.stderr)
        return 1
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = processed = 0
    print(f"[INFO] İşleniyor: {input_path.name}")
    
    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            total += 1
            try:
                doc = json.loads(line)
            except Exception as e:
                print(f"[WARN] Satır {total} parse hatası: {e}", file=sys.stderr)
                continue
            
            # SADECE 'ref' alanını normalize et
            if "ref" in doc:
                doc["ref"] = TextNormalizer.normalize_safe(doc["ref"])
            
            # Diğer alanlar (text, id, vb.) olduğu gibi kalır
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            processed += 1
    
    print(f"[DONE] {processed}/{total} döküman işlendi → {output_path.name}")
    return 0


def main():
    # Input: src/data/uncleaned, Output: src/data/cleaned
    input_dir = Path("src/data/uncleaned")
    output_dir = Path("src/data/cleaned")
    
    files_to_process = [
        "medium_examples_arxiv.jsonl",
        "medium_examples_longform.jsonl",
        "long_examples_arxiv.jsonl",
        "long_examples_longform.jsonl",
        "extra_long_examples_arxiv.jsonl",
        "extra_long_examples_longform.jsonl"
    ]
    
    print("="*60)
    print("REF NORMALIZATION - Sadece 'ref' alanı temizlenir")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    for filename in files_to_process:
        input_path = input_dir / filename
        output_path = output_dir / f"cleaned_{filename}"
        
        if input_path.exists():
            process_file(input_path, output_path)
        else:
            print(f"[SKIP] Dosya yok: {filename}")
    
    print("="*60)
    print("[INFO] Tüm dosyalar işlendi!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
