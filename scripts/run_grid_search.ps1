# Grid Search Toplu Çalıştırma Scripti
# PowerShell için - tüm deney konfigürasyonlarını sırayla çalıştırır

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Grid Search Deneyleri Başlatılıyor" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

Write-Host "Preparing and running full grid (27 configs)..." -ForegroundColor Yellow

# Create grid configs (will populate configs/sliding/grid/)
python scripts/generate_grid_configs.py

# Run the full grid runner
if (Test-Path .\scripts\run_full_grid.ps1) {
    Write-Host "Launching run_full_grid.ps1" -ForegroundColor Green
    .\scripts\run_full_grid.ps1
} else {
    Write-Host "run_full_grid.ps1 not found; please run configs manually or create the runner." -ForegroundColor Red
}

Write-Host "`n====================================" -ForegroundColor Green
Write-Host "Tüm deneyler tamamlandı!" -ForegroundColor Green
Write-Host "Sonuçlar: results/ klasörüne kaydedildi" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
