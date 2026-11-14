# Kimodo RX 6650 XT / Windows test odaklı yol haritası

| Adım | Çalışma | Başarı kriteri | Güncel durum |
|---|---|---|---|
| K0 | GitHub kaynak/bağımlılık incelemesi | GPU/CPU yolu, CUDA bağımlılıkları ve lisans gereksinimleri belgeli | PASS - statik inceleme |
| K1 | Yerel Windows gfx1032 çalışma ortamı | AMD ROCm PyTorch kuruldu, GPU doğru tespit edildi | PASS - 2026-11-14, torch 2.14.0+rocm10.1.0 (Akıllı Uygulama Denetimi kapatıldı) |
| K2 | Çekirdek hesaplama | GPU float32 matmul CPU ile eşleşiyor | PASS - 2026-11-14, max hata 1e-05 |
| K3 | Kimodo minimum CLI | İçeri aktarma, checkpoint ve kimodo_gen yardım ekranı | PASS - 2026-11-14 (SOMA-X yerelden, MHR lod1 hariç) |
| K4 | LLM2Vec metin kodlayıcı | HF izni, RAM içinde ya da uyumlu uzak encoder API'siyle çalışıyor | PASS - 2026-11-14, yerel CPU + 28 GB sayfa dosyası; MISSING LoRA uyarısı zararsız (lora_B=0) |
| K5 | 1-3 saniye animasyon | NPZ ve BVH gerçek üretilmiş, boş değil | PASS - 2026-11-14, yürüme 7.9 m / el sallama 0.01 m kök hareketi; ~2.6 dk/üretim |
| K6 | Blender/Unreal retarget | Animasyon hedef iskelette oynuyor | PASS - 2026-11-14: Blender 5.3 FBX/GLB (köşe hatası ≤52 µm, round-trip 0.03 mm); UE 5.8 IK Retargeter SOMA→Manny, 29/29 zincir, hareket korelasyonu ≥0.995 |
| K7 | MotionCorrection | C++ uzantısı derleniyor ve hata düzeltme doğrulanıyor | PASS - 2026-11-14: MSVC 2022 + pip pybind11 (Py 3.13), run_test geçti; A/B 4 prompt: temas kayması %34-51 azaldı, tepe hız yarıya indi. Yan etki: salınımda ayak tabanı yere daha yakın geçiyor (yükseklik metriği 3/4'te arttı); Studio'da varsayılan açık |
| K8 | Etkileşimli demo (Viser yerine Kimodo Studio) | Timeline ve constraint UI ile üretim | PASS - 2026-11-14: Studio'da segment zaman çizelgesi, çizilen kök yolu, poz/el/ayak anahtar kareleri. Ölçüm: yol ort. 1.5-3.6 cm, poz/el 0.0 cm, anahtar etrafında sıçrama yok. Kimodo hatası (çok segment + boş kalan kısıt) worker'da yamandı. Kök yolu artık zaman aralığına sınırlanabiliyor (segment seçimi); yol yalnızca yürüme segmentindeyken zıplama segmenti gerçekten zıplıyor (0 → 10 havada kare) |

## Kapılar

K1 ve K2 geçilmeden AMD GPU ile çalışıyor denmez. Donanımda koşmamış bir GPU testi PASS sayılmaz, BLOCKED olarak raporlanır (`backend_probe.json`, `doctor.ps1`). CPU-only hesaplama AMD GPU başarısı değildir. K4'te Windows'taki 16 GB RAM yetersiz kalabilir. K5'te postprocess kapalıdır. K7 açıkken (Studio varsayılanı) temas kayması azalır; salınımdaki ayak yüksekliği hafif düşer. Kaynak kodu ve model ağırlıkları üçüncü taraflardan ayrıca indirilecektir.

## Her geliştirmede test

1. `python -m unittest discover -s tests -v` başlangıç paketinin statik/iş mantığı testleri.
2. `doctor.ps1` gerçek Windows+ROCm GPU testi.
3. `generate.ps1` uçtan uca Kimodo testi.
4. Regresyon ve CPU-GPU karşılaştırma. GPU testleri yalnızca donanımda koştuysa PASS.
