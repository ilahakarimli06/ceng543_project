# Tam Grid Search Çalıştırma Scripti
# 3×3×3 = 27 deney kombinasyonunu sırayla çalıştırır

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "TAM GRID SEARCH - 27 Deney Kombinasyonu" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Grid config'leri oluştur
Write-Host "`n[Adım 1/2] Grid config dosyaları oluşturuluyor..." -ForegroundColor Yellow
python scripts/generate_grid_configs.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Config oluşturma başarısız!" -ForegroundColor Red
    exit 1
}

# Results klasörünü oluştur
New-Item -ItemType Directory -Force -Path "results/grid" | Out-Null

# Tüm grid config'leri çalıştır
Write-Host "`n[Adım 2/2] Deneyler çalıştırılıyor..." -ForegroundColor Yellow

$configs = Get-ChildItem -Path "configs/sliding/grid" -Filter "*.yml" | 
    Where-Object { $_.Name -ne "MANIFEST.yml" } |
    Sort-Object Name

$total = $configs.Count
$current = 0

foreach ($config in $configs) {
    $current++
    $percent = [math]::Round(($current / $total) * 100, 1)
    
    Write-Host "`n[$current/$total - $percent%] $($config.Name)" -ForegroundColor Green
    Write-Host "─────────────────────────────────────────────" -ForegroundColor Gray
    
    python main.py --config $config.FullName
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️ Deney başarısız - devam ediliyor..." -ForegroundColor Yellow
    } else {
        Write-Host "✓ Tamamlandı" -ForegroundColor Green
    }
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "✅ TÜM DENEYLER TAMAMLANDI!" -ForegroundColor Green
Write-Host "Sonuçlar: results/grid/" -ForegroundColor Cyan
Write-Host "Config'ler: configs/sliding/grid/" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
