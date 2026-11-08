# Bağımsız kök neden incelemesi: PNG/BMP RGB SRNet

Tarih: 2026-11-08. Bu dosya konuşmaların tamamı değildir; doğrulanmış sonuçları,
açık soruları ve kaynak kodun ilgili bölümlerini içeren bağımsız inceleme paketidir.
Sohbet geçmişine veya önceki asistanın hafızasına ihtiyaç duymadan okunabilir.

## İnceleyene görev

RGB steganaliz modelinin neden öğrenme eşiğine ulaşmadığını incele. Önceki
asistanın açıklamasını doğru varsayma. Veri/etiket, embedding, mimari, giriş
ölçeği, loss/optimizer, BatchNorm ve değerlendirme mantığını birbirinden ayır.
“Veri az” veya “daha uzun eğit” sonucuna kanıt olmadan atlama. Bir hata
bulursan dosya/fonksiyon ve mekanizmasıyla açıkla. Kesin bulguları hipotezlerden
ayır; mevcut kanıtın neyi dışlayamadığını belirt. En az ek hesaplama gerektiren
ayırt edici deneyi öner. Kanıt yetmiyorsa kesin kök neden bulunduğunu söyleme.

Repo erişimin varsa önce `AGENTS.md` dosyasını oku ve incelemeyi salt okunur
tut. Yeni GPU işi, optimizer güncellemesi, indirme, eski deneyi tekrar başlatma,
checkpoint değiştirme veya ayrılmış piksellere erişme bu incelemenin parçası
değildir. Kullanıcı yerel GPU yerine Colab tercih ediyor. Bu belge oluşturulurken
hiçbir harici asistana dosya gönderilmedi.

## Gerçekte ne yapıldı?

- Tamamlanan native RGB eğitimi: iki taze model tohumu, her birinde 20 epoch /
  320 güncelleme. Toplam 640 güncelleme /5.120 sunum /sekiz sayısal checkpoint.
- Eğitim: 64 özgün görüntü /38 bilinen soy-köken bileşeni. Geliştirme: 20 /12.
  Ayrılmış 21 RGB /13 L görüntü kapalı. Gerçek kaynak rolleri hâlâ unassigned;
  bölünme ayrı bir erişim katmanı. Yeni veri edinilmedi.
- RGB 256×256, ham 0–255 FP32, AMP/TF32 kapalı. Batch sekiz: dört farklı
  orijinalden cover/stego çiftleri; etiketler `[0,1,0,1,0,1,0,1]`.
- Scalar AdamW: lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0,
  foreach/fused=False. Payload tohumu 2026102805; model tohumları aşağıda.
- Payload plane-0'a, kanal örneklerinin yaklaşık %40'ına rastgele bit
  yerleştiriyor. RGB'de bu 78.640 bit /9.830 bayttır; **0,4 bit/pixel ile aynı
  ölçü değildir**. Gerçekte değişen örnekler ortalama %20,0039.
- Ortam: Colab Runtime 2026.07, Tesla T4, driver 580.82.07, Python 3.12.13,
  NumPy 2.0.2, Pillow 11.3.0, Torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900.
- İki modelin de sabit öğrenme kuralı başarısız: train AUC≥.85, development
  AUC≥.70 ve son epoch loss≤ilk loss×.8. GPU oturumu silindi; modeller özeldir.
- Aşağıdaki sayılar **ROC-AUC**'dir; belirli eşikte classification accuracy
  ölçümleri değildir. Başarısız sonuç değiştirilmedi veya gizlenmedi.

## Son sonuçlar

| Tohum | Train AUC | Development AUC | İlk /son epoch loss | Train doğru eşleşmiş çift | Development doğru eşleşmiş çift |
| --- | ---: | ---: | --- | ---: | ---: |
| 2026102301 | 0.505615234 | 0.517500 | 0.702335812 /0.694389019 | 51/64 | 15/20 |
| 2026102302 | 0.514648438 | 0.527500 | 0.703802656 /0.692473006 | 61/64 | 20/20 |

Her development stego skoru bütün cover skorlarıyla karşılaştırıldığında AUC
hesaplanır. Bir orijinalin kendi çifti içinde doğru sıralanması, farklı bir
orijinale karşı güvenilir karar verdiği anlamına gelmez. Yirmi çiftte 400
karşılaştırmanın yalnız yirmisi aynı orijinale aittir; diğer 380'de AUC
.505263 /.502632. Development çiftlerinin orta skorları arasındaki standart
sapma, stego-cover farklarının RMS'inden 123,42 /43,65 kat büyüktür.

Önceki yorum: görüntüye özgü temel skor kaymaları, zayıf embedding yanıtını
bastırıyor. Bu bir **gözlenen başarısızlık mekanizmasıdır**; veri büyüklüğü,
güncelleme sayısı veya mimarinin neden bu sonuca yol açtığını tek başına kanıtlamaz.
Bu yorumun kendisini de eleştirel incele.

## Kontrol edilenler ve sınırları

- 64 train crop'u Colab'a taşınan baytlarla birebir aynı; hash'ler farklı.
  Her orijinal için bağımsız scalar payload recovery başarılı. Üst yedi bit
  korunuyor; değişen değerler yalnız -1/0/+1. HWC→CHW ve cover/stego sırası doğru.
- 4.780.768 parametre /105 tensor. Bütün optimizer step'leri checkpoint'lerde
  80/160/240/320. İki modelde de 105 tensorun tamamı epoch 5→20 arasında
  değişiyor; Adam ikinci momenti hiçbir parametre tensorunda tamamen sıfır değil.
- Son modellerde birer sekiz satırlık train batch'inde 105 gradient tensorunun
  tamamı mevcut, sonlu ve sıfırdan farklı. Bu salt backward incelemesinde
  optimizer oluşturulmadı; parametreler değişmedi ve BN buffer'ları geri alındı.
  Geçmiş fit'te her adımın gradient normu kaydedilmemişti.
- Sayısal checkpoint'lerin CPU inference'ı son GPU train skorlarıyla en fazla
  1,79e-7 /3,58e-7 ayrılıyor; AUC'ler birebir aynı. Yerel CPU sürümü farklıdır:
  Python 3.11.14 /NumPy 2.4.6 /Torch 2.14.0+cpu. Bu original-environment resume
  değildir; yükleyici kaynak/veri/hash bağlarını doğrulayarak array okur.
- Parametreler sabitken train batch istatistikleriyle BN kullanmak train AUC'yi
  .512939 /.538330 yapıyor. Bu kontrol tek başına sorunu çözmüyor; BatchNorm'un
  hiç etkisi olmadığını kanıtlamaz. Bilinen eşleşmiş batch'lerle puanlama
  deploy edilebilir bir karar kuralı olarak sunulmamalı.
- Önceki başarılı sentetik testin cover örnekleri `randint(0,128)*2`:
  bütün düşük bitler sıfır. Stego'da yaklaşık %20'si bir oluyor. Native train
  cover/stego düşük-bit bir oranları ortalama %50,2920 /%50,1801. Sentetik test
  native görevde bulunmayan büyük bir yapay ipucuna izin veriyor; eski modelin
  tam olarak hangi özelliği kullandığı ölçülmedi. Eski pass yalnız generated-only.
- Grup denetimleri bilinen group/author/page bağlarını transitif birleştiriyor;
  bilinmeyen önceki ortak kaynakları dışlamıyor. Development zaten birçok kez
  görülmüş araştırma kümesi; bağımsız yeterlilik testi sayılmıyor.

## Başlangıç skorları: ek inceleme sorusu

- 2026102301: başlangıçta kaydedilmiş stego-sınıfı olasılıkları: train: min=max=0.0; development: min=max=0.0.
- 2026102302: başlangıçta kaydedilmiş stego-sınıfı olasılıkları: train: min=max=0.0; development: min=max=0.0.

Burada iki sütun cover ve stego örneklerinin **aynı pozitif sınıf** skorudur;
iki ayrı sınıf olasılığı değildir. Başlangıçtaki BN running-stat/cold-start ve
softmax doygunluğu/logit ölçeği incelenmeye değer. Logit'ler kaydedilmediği için
mekanizma doğrudan doğrulanmadı. Eğitimin train-mode BN ile ilerlediği ve
sonraki checkpoint'lerde skorların ayrıştığı da hesaba katılmalı; bu başlangıç
bulgusu sürekli öğrenme başarısızlığının kanıtlanmış nedeni değildir.

## Bütün kayıtlı train checkpoint'leri

| Tohum | Epoch | Güncelleme | Global train AUC | Doğru eşleşmiş çift | Görüntü farkı /embedding yanıtı |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026102301 | 5 | 80 | 0.506591797 | 55/64 | 180.00 |
| 2026102301 | 10 | 160 | 0.509765625 | 56/64 | 43.93 |
| 2026102301 | 15 | 240 | 0.506835938 | 47/64 | 48.63 |
| 2026102301 | 20 | 320 | 0.505615234 | 51/64 | 105.48 |
| 2026102302 | 5 | 80 | 0.503417969 | 43/64 | 261.90 |
| 2026102302 | 10 | 160 | 0.507568359 | 57/64 | 127.41 |
| 2026102302 | 15 | 240 | 0.508789062 | 60/64 | 74.98 |
| 2026102302 | 20 | 320 | 0.514648438 | 61/64 | 23.78 |

Bunlar mevcut sekiz state'in yeni, salt okunur CPU inference'larıdır. İkinci
tohumda zayıf eşleşmiş yanıt güçleniyor, ilkinde ilerleme monoton değil. Hiçbir
kayıtlı noktada yüksek train/düşük development aşırı öğrenme örüntüsü yok.
Bu gözlem “daha uzun eğitim kesin çözer” sonucunu vermez.

## Kanıtlanmamış nedenler ve sonraki taslak

64 heterojen train görüntüsü, yalnız 320 güncelleme/tohum, öğrenme oranı ve
optimizer seçimi, içeriği bastıramayan temsil/mimari hâlâ adaylar. Hiçbiri
bağımsız kontrollü karşılaştırmayla kök neden olarak izole edilmedi.

Metadata planı hazır: 8/64 iç içe train kaynağı, bütün soy grupları korunarak
6/38 bileşen; iki taze tohum, her büyüklükte 320/1.280 güncelleme uçları.
Dört yeni trajectory, sekiz uç, toplam 5.120 güncelleme /40.960 sunum öneriliyor.
Küçük kümede kalan 56 eski train kaynağı ek araştırma holdout'u; aynı eski
development her iki büyüklükte ortak. Bir küçük altküme kaynak çeşitliliğini ve
eşit adımda tekrar sıklığını da değiştirir; saf evrensel veri-boyutu etkisi sayılmaz.

**Bu deney yürütülmedi.** Step-bounded çalıştırıcı/supervisor, yeni executable
QA/source/environment/cost freeze ve somut ayrı çalışma izni eksik. Eski native
Plan yirmi epoch ile sınırlı; limiti veya doğrulanmış schedule'ı monkey-patch
ederek planı koşturma. Eski checkpoint'leri devam ettirme. 113,85 dakikalık
eski-komut ölçeklemesi yalnız yaklaşık planlama proxy'si, kabul edilmiş maliyet
ya da harcama izni değildir. Mevcut taslağın daha küçük ve daha ayırt edici bir
deneyle değiştirilebileceğini bağımsız olarak değerlendir.

## Kaynak dosyalar ve teyit

Asıl raporlar:

- `docs/PNG_BMP_COLAB_NATIVE_RGB_RESULTS_20261107.md`
- `docs/PNG_BMP_NATIVE_RGB_DIAGNOSIS_20261107.md`
- `docs/PNG_BMP_NATIVE_CHECKPOINT_TRAJECTORY_20261108.md`
- `docs/PNG_BMP_NATIVE_DATA_UPDATE_STUDY_20261108.md`

Ölçüm JSON'ları aynı ad/tarihlerle `benchmarks/` altında. 8.992 ve 32.768
bağımsız pairwise AUC karşılaştırması, sekiz state/hash ve 522 yazılım girdisi
doğrulandı. Mevcut full QA 4.396 pass /iki açık CUDA skip /%97,225442 coverage;
son incelemeler belge/kanıt değişikliği, yeni full-suite koşusu değil.

State'ler ve tam skorlar özel `.benchmark/` dizininde; bu pakete medya,
checkpoint ağırlığı, kimlik bilgisi, ekran görüntüsü veya sohbet dökümü eklenmedi.
Kaynak kod alıntıları otomatik alınmış birebir bölümlerdir; bağımsız çalışan
bir script değildir. Dosya SHA-256'ları tam kaynak dosyasına aittir. İnceleme
alıntısı için decorator satırları da korunur. İlgili diğer yardımcılar repoda
bulunur; eksik bir kaynağa ihtiyacın varsa yolunu iste.

## Kod alıntıları

### `core/srnet.py` — `network`

Tam dosya SHA-256: `11a69761d691b2291eccf83d8ba449ff122b1e6db49580c46817f649053c3345`. Satırlar 12–88.

```python
def network():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SRNet preparation requires the optional research extra") from exc

    class ConvBN(torch.nn.Sequential):
        def __init__(self, before, after, kernel=3, stride=1, relu=False):
            modules = [
                torch.nn.Conv2d(
                    before,
                    after,
                    kernel,
                    stride=stride,
                    padding=kernel // 2,
                    dtype=torch.float32,
                    device="cpu",
                ),
                torch.nn.BatchNorm2d(
                    after, eps=1e-5, momentum=0.1, dtype=torch.float32, device="cpu"
                ),
            ]
            if relu:
                modules.append(torch.nn.ReLU())
            super().__init__(*modules)

    class Residual(torch.nn.Module):
        def __init__(self, before, after, down=False):
            super().__init__()
            self.branch = torch.nn.Sequential(
                ConvBN(before, after, relu=True),
                ConvBN(after, after),
            )
            self.pool = (
                torch.nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
                if down
                else torch.nn.Identity()
            )
            self.skip = ConvBN(before, after, kernel=1, stride=2) if down else torch.nn.Identity()

        def forward(self, values):
            # No activation after the residual addition.
            return self.pool(self.branch(values)) + self.skip(values)

    class SRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = ARCHITECTURE
            self.front = torch.nn.Sequential(
                ConvBN(1, 64, relu=True),
                ConvBN(64, 16, relu=True),
                *(Residual(16, 16) for _ in range(5)),
            )
            self.middle = torch.nn.Sequential(
                *(
                    Residual(before, after, down=True)
                    for before, after in ((16, 16), (16, 64), (64, 128), (128, 256))
                ),
            )
            self.tail = torch.nn.Sequential(
                ConvBN(256, 512, relu=True),
                ConvBN(512, 512),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
            self.classifier = torch.nn.Linear(512, 2, bias=False, dtype=torch.float32, device="cpu")
            for module in self.modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0.2)
            torch.nn.init.normal_(self.classifier.weight, std=0.01)

        def forward(self, values):
            return self.classifier(self.tail(self.middle(self.front(values))))

    return SRNet()
```

### `core/nonjpeg_models.py` — `network`

Tam dosya SHA-256: `ea2e4cca67498a0d1018485e56867cc0c402b49cb3f4d99bc0a3be8f120a90c8`. Satırlar 30–94.

```python
def network(domain: str):
    if domain not in DOMAINS:
        raise ValueError("unknown native model domain")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("native model skeleton unavailable: optional Torch missing") from exc

    if domain in (GRAY_DOMAIN, RGB_DOMAIN):
        from core.srnet import network as srnet_network

        model = srnet_network()
        if domain == RGB_DOMAIN:
            conv = torch.nn.Conv2d(3, 64, 3, padding=1, dtype=torch.float32, device="cpu")
            torch.nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            if conv.bias is not None:
                torch.nn.init.constant_(conv.bias, 0.2)
            model.front[0][0] = conv
        model.architecture = domain
        return model

    class Conv(torch.nn.Sequential):
        def __init__(self, before, after, stride=1):
            super().__init__(
                torch.nn.Conv1d(before, after, 5, stride=stride, padding=2, dtype=torch.float32),
                torch.nn.BatchNorm1d(after, dtype=torch.float32),
                torch.nn.ReLU(),
            )

    class Residual(torch.nn.Module):
        def __init__(self, before, after, stride=1):
            super().__init__()
            self.branch = torch.nn.Sequential(
                Conv(before, after, stride),
                torch.nn.Conv1d(after, after, 5, padding=2, dtype=torch.float32),
                torch.nn.BatchNorm1d(after, dtype=torch.float32),
            )
            self.skip = (
                torch.nn.Identity()
                if before == after and stride == 1
                else torch.nn.Conv1d(before, after, 1, stride=stride, dtype=torch.float32)
            )

        def forward(self, values):
            return torch.relu(self.branch(values) + self.skip(values))

    class PCMNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = AUDIO_DOMAIN
            self.body = torch.nn.Sequential(
                Conv(3, 32),
                Residual(32, 32),
                Residual(32, 32),
                Residual(32, 64, 2),
                Residual(64, 128, 2),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
            )
            self.classifier = torch.nn.Linear(128, 2, dtype=torch.float32)

        def forward(self, values):
            return self.classifier(self.body(values))

    return PCMNet()
```

### `core/spatial_training_data.py` — `positions`

Tam dosya SHA-256: `0341d06d816d8194fcb92a6535e1da8fc178510fd70d475badce950295eaf15e`. Satırlar 58–63.

```python
def positions(samples: int, scatter_seed: int) -> np.ndarray:
    seed(scatter_seed)
    if type(samples) is not int or samples not in (SIDE * SIDE, SIDE * SIDE * 3):
        raise ValueError("bounded spatial sample count required")
    bits = (samples * 2 // 5) // 8 * 8
    return np.random.Generator(np.random.PCG64(scatter_seed)).permutation(samples)[:bits]
```

### `core/spatial_training_data.py` — `embed`

Tam dosya SHA-256: `0341d06d816d8194fcb92a6535e1da8fc178510fd70d475badce950295eaf15e`. Satırlar 66–75.

```python
def embed(value: np.ndarray, payload: bytes, scatter_seed: int) -> tuple[np.ndarray, np.ndarray]:
    value = pixels(value)
    selected = positions(value.size, scatter_seed)
    if not isinstance(payload, bytes) or len(payload) != len(selected) // 8:
        raise ValueError("exact forty-percent byte-rounded payload required")
    result = value.copy()
    flat = result.reshape(-1)
    bits = np.unpackbits(np.frombuffer(payload, dtype="u1"), bitorder="big")
    flat[selected] = (flat[selected] & 254) | bits
    return result, selected
```

### `core/spatial_training_data.py` — `pair`

Tam dosya SHA-256: `0341d06d816d8194fcb92a6535e1da8fc178510fd70d475badce950295eaf15e`. Satırlar 98–132.

```python
def pair(spec: GeneratedSpec, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if type(spec) is not GeneratedSpec or type(index) is not int or not 0 <= index < spec.originals:
        raise ValueError("generated source/index required")
    # Count is not part of source identity: expanding a spec cannot invent new groups
    # for the same seed/split/index. Split and mode use disjoint seed namespaces.
    descriptor = {
        "recipe": RECIPE,
        "mode": spec.mode,
        "seed": spec.seed,
        "split": spec.split,
        "index": index,
    }
    identity = a.digest(a.encode(descriptor))
    rng = np.random.Generator(np.random.PCG64(int(identity[:16], 16)))
    channels = 1 if spec.mode == "L" else 3
    cover = rng.integers(0, 128, (SIDE, SIDE, channels), dtype="u1") * np.uint8(2)
    payload = rng.integers(0, 256, (cover.size * 2 // 5) // 8, dtype="u1").tobytes()
    scatter_seed = int(identity[16:24], 16)
    control, selected = embed(cover, payload, scatter_seed)
    if recover(control, selected) != payload:
        raise ValueError("generated spatial payload recovery failed")
    return (
        cover,
        control,
        {
            "id": "generated-" + identity,
            "group": "generated-" + identity,
            "split": spec.split,
            "cover_sha256": a.digest(cover.tobytes()),
            "control_sha256": a.digest(control.tobytes()),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "positions_sha256": a.digest(selected.astype("<i8").tobytes()),
            "payload_bytes": len(payload),
        },
    )
```

### `core/spatial_native_data.py` — `pair`

Tam dosya SHA-256: `7d24498550e5d91462d774f9f7150b138c2769f9dbead14ba07bf22855969419`. Satırlar 324–362.

```python
def pair(cache: Cache, index: int, payload_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    scattered.seed(payload_seed)
    if type(cache) is not Cache or type(index) is not int or not 0 <= index < len(cache.rows):
        raise ValueError("native cache source/index required")
    row = cache.rows[index]
    channels = 1 if row.mode == "L" else 3
    raw = a.read(cache.root / (row.id + ".u8"), 256 * 256 * channels)
    if len(raw) != 256 * 256 * channels or a.digest(raw) != row.crop_sha256:
        raise ValueError("native cached pixels hash/extent")
    cover = np.frombuffer(raw, dtype="u1").reshape(256, 256, channels).copy()
    identity = a.digest(
        a.encode(
            {
                "recipe": RECIPE,
                "original": row.original_sha256,
                "mode": row.mode,
                "seed": payload_seed,
            }
        )
    )
    rng = np.random.Generator(np.random.PCG64(int(identity[:16], 16)))
    payload = rng.integers(0, 256, (cover.size * 2 // 5) // 8, dtype="u1").tobytes()
    control, selected = scattered.embed(cover, payload, int(identity[16:24], 16))
    if scattered.recover(control, selected) != payload:
        raise ValueError("native scattered payload recovery failed")
    return (
        cover,
        control,
        {
            "id": row.id,
            "group": row.group,
            "role": "unassigned",
            "cover_sha256": row.crop_sha256,
            "control_sha256": a.digest(control.tobytes()),
            "payload_sha256": a.digest(payload),
            "payload_bytes": len(payload),
            "positions_sha256": a.digest(selected.astype("<i8").tobytes()),
        },
    )
```

### `core/spatial_native_data.py` — `batch`

Tam dosya SHA-256: `7d24498550e5d91462d774f9f7150b138c2769f9dbead14ba07bf22855969419`. Satırlar 365–384.

```python
def batch(
    cache: Cache, mode: str, indices: tuple[int, ...], payload_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(cache) is not Cache
        or mode not in ("L", "RGB")
        or not isinstance(indices, tuple)
        or not 1 <= len(indices) <= 4
        or any(
            type(i) is not int or not 0 <= i < len(cache.rows) or cache.rows[i].mode != mode
            for i in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("one to four distinct native pairs of one mode required")
    values: list[np.ndarray] = []
    for index in indices:
        cover, control, _ = pair(cache, index, payload_seed)
        values.extend((cover.transpose(2, 0, 1), control.transpose(2, 0, 1)))
    return np.stack(values).astype("<f4"), np.tile(np.array([0, 1], dtype="<i8"), len(indices))
```

### `core/spatial_native_training.py` — `order`

Tam dosya SHA-256: `c01fd509a811a04384715fb8efcca90263ee8403d89313258da054f153de035b`. Satırlar 165–172.

```python
def order(plan: Plan) -> np.ndarray:
    if type(plan) is not Plan:
        raise ValueError("native plan required")
    rng = np.random.Generator(np.random.PCG64(plan.model_seed))
    n = len(plan.split.train)
    return np.concatenate([rng.permutation(n) + epoch * n for epoch in range(plan.epochs)]).astype(
        "<i8"
    )
```

### `core/spatial_native_training.py` — `_Session.__init__`

Tam dosya SHA-256: `c01fd509a811a04384715fb8efcca90263ee8403d89313258da054f153de035b`. Satırlar 176–219.

```python
    def __init__(self, cache: data.Cache, plan: Plan, device: str, max_seconds: float):
        import torch

        self.cache = data.load(cache.root, plan.split.manifest_sha256)
        if self.cache.rows != cache.rows:
            raise ValueError("native cache source mismatch")
        plan.split.validate(self.cache)
        self.lookup = {row.id: i for i, row in enumerate(self.cache.rows)}
        self.plan, self.device, self.max_seconds = plan, device, max_seconds
        self.started = time.monotonic()
        self.prior_seconds = 0.0
        self.failed = False
        self.cursor = self.updates = self.presentations = self.output_bytes = 0
        self.schedule = order(plan)
        self.data_sha256 = a.digest(
            a.encode(
                {
                    "manifest": cache.manifest_sha256,
                    "split": asdict(plan.split),
                    "recipe": data.RECIPE,
                    "payload_seed": plan.payload_seed,
                }
            )
        )
        self.binding = a.digest(
            a.encode(
                {
                    "plan": asdict(plan),
                    "device": device,
                    "recipe": data.RECIPE,
                    "optimizer": "scalar-adamw-1e-4-v1",
                }
            )
        )
        self.model = models.network(plan.domain).to(device).train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )
```

### `core/spatial_native_training.py` — `_Session.step`

Tam dosya SHA-256: `c01fd509a811a04384715fb8efcca90263ee8403d89313258da054f153de035b`. Satırlar 226–271.

```python
    def step(self) -> dict:
        import torch

        self.check()
        if self.cursor == len(self.schedule):
            raise StopIteration
        count = len(self.plan.split.train)
        stop = min(self.cursor + self.plan.batch_images // 2, (self.cursor // count + 1) * count)
        indices = tuple(int(v % count) for v in self.schedule[self.cursor : stop])
        try:
            x, labels = data.batch(
                self.cache,
                self.plan.split.mode,
                tuple(self.lookup[self.plan.split.train[i]] for i in indices),
                self.plan.payload_seed,
            )
            values = torch.from_numpy(x).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                self.model(values), torch.from_numpy(labels).to(self.device)
            )
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(
                p.grad is None or not bool(torch.isfinite(p.grad).all())
                for p in self.model.parameters()
            ):
                raise ValueError("native nonfinite/absent gradients")
            self.check()
            self.optimizer.step()
            # Charge immediately, including an update that subsequently fails a check.
            self.updates += 1
            self.presentations += 2 * len(indices)
            self.cursor = stop
            if any(not bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()):
                raise ValueError("native nonfinite model state")
            self.check()
        except Exception:
            self.failed = True
            raise
        return {
            "optimizer_updates": self.updates,
            "presentations": self.presentations,
            "cursor": self.cursor,
            "loss": float(loss.detach().cpu()),
            "scope": "native_adapter",
        }
```

### `core/spatial_colab_fit.py` — `auc`

Tam dosya SHA-256: `6c48c369669f040f878f7c2f4155888d3c28df70a4aeb8aca277df482b2bf124`. Satırlar 239–261.

```python
def auc(values) -> float:
    """Bounded average-rank ROC AUC; audit uses an independent pairwise calculation."""
    pairs = np.asarray(values, dtype=np.float64)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not 1 <= len(pairs) <= 128
        or not np.isfinite(pairs).all()
        or np.any((pairs < 0) | (pairs > 1))
    ):
        raise ValueError("bounded finite paired scores required")
    scores = pairs.ravel()
    ordered = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[ordered[stop]] == scores[ordered[start]]:
            stop += 1
        ranks[ordered[start:stop]] = (start + 1 + stop) / 2
        start = stop
    n = len(pairs)
    return float((ranks[1::2].sum() - n * (n + 1) / 2) / (n * n))
```

### `core/spatial_colab_fit.py` — `_Session.scores`

Tam dosya SHA-256: `6c48c369669f040f878f7c2f4155888d3c28df70a4aeb8aca277df482b2bf124`. Satırlar 265–287.

```python
    def scores(self, role, *, record_scores=True):
        import torch

        if role not in ("train", "development") or record_scores is not True:
            raise ValueError("fit scoring is restricted to train/development")
        previous, values = self.model.training, []
        try:
            self.model.eval()
            with torch.inference_mode():
                for source_id in getattr(self.plan.split, role):
                    self.check()
                    x, _ = data.batch(
                        self.cache,
                        self.plan.split.mode,
                        (self.lookup[source_id],),
                        self.plan.payload_seed,
                    )
                    logits = self.model(torch.from_numpy(x).to(self.device))
                    values.append(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
        finally:
            self.model.train(previous)
        self.check()
        return {"scores": values, "auc": auc(values), "originals": len(values)}
```

### `core/spatial_colab_fit.py` — `session`

Tam dosya SHA-256: `6c48c369669f040f878f7c2f4155888d3c28df70a4aeb8aca277df482b2bf124`. Satırlar 290–310.

```python
@contextmanager
def session(cache, plan, *, device="cpu", max_seconds=120):
    import torch

    if device not in ("cpu", "cuda:0") or not 0 < finite(max_seconds) <= 1800:
        raise ValueError("explicit fitting device/deadline required")
    state = random.getstate(), np.random.get_state(), torch.get_num_threads()
    policy = profile.cuda_policy() if device == "cuda:0" else engine.capacity.fp32_policy()
    try:
        with policy, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.set_num_threads(2)
            torch.random.default_generator.manual_seed(plan.model_seed)
            if device == "cuda:0":
                torch.cuda.manual_seed(plan.model_seed)
            random.seed(plan.model_seed)
            np.random.seed(plan.model_seed)
            yield _Session(cache, plan, device, max_seconds)
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_num_threads(state[2])
```

### `core/spatial_colab_fit.py` — `train`

Tam dosya SHA-256: `6c48c369669f040f878f7c2f4155888d3c28df70a4aeb8aca277df482b2bf124`. Satırlar 313–431.

```python
def train(
    cache,
    split,
    out,
    binding,
    *,
    device="cuda:0",
    seconds=1700,
    epochs=20,
    max_checkpoint_bytes=128 * a.MIB,
):
    """Shared loop. Reduced CPU schedules are restricted to generated QA namespaces."""
    if (
        type(epochs) is not int
        or not 1 <= epochs <= 20
        or (
            epochs != 20
            and (
                device != "cpu"
                or any(not row.id.startswith("generated-profile-") for row in cache.rows)
            )
        )
    ):
        raise ValueError("reduced exposure is generated CPU QA only")
    split.validate(cache)
    a.identity(binding)
    a.directory(out)
    deadline, commits, cells = time.monotonic() + finite(seconds, positive=True), 0, []
    steps = math.ceil(len(split.train) / 4)
    for cell, seed in enumerate(profile.pilot.SEEDS):
        plan = engine.Plan(split, seed, RECIPE["payload_seed"], epochs, 8, binding)
        with session(
            cache, plan, device=device, max_seconds=profile.pilot.remaining(deadline)
        ) as s:
            initial = {role: s.scores(role) for role in ("train", "development")}
            rows, checkpoints, losses = [], [], []
            for epoch in range(1, epochs + 1):
                epoch_losses = []
                for step in range(steps):
                    intent = {"cell": cell, "epoch": epoch, "step": step, "native": True}
                    profile.pilot.record_event(out, commits + 1, "intent", intent)
                    result = s.step()
                    commits += 1
                    epoch_losses.append(result["loss"])
                    losses.append(result["loss"])
                    profile.pilot.record_event(out, commits, "commit", {**intent, "result": result})
                development = s.scores("development")
                row = {
                    "seed": seed,
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "development": development,
                    "optimizer_updates": s.updates,
                }
                rows.append(row)
                a.put(out / f"seed-{seed}-epoch-{epoch:02d}.json", a.encode(row))
                profile.pilot.historical.progress(out, {"contract_sha256": binding, **row})
                print(jsonline(row), flush=True)
                if epoch % 5 == 0 or epoch == epochs:
                    path = out / f"seed-{seed}-epoch-{epoch:02d}.numeric"
                    sha = s.checkpoint(path)
                    if path.stat().st_size > max_checkpoint_bytes:
                        raise ValueError("measured checkpoint bound exceeded")
                    checkpoints.append(
                        {
                            "name": path.name,
                            "sha256": sha,
                            "epoch": epoch,
                            "bytes": path.stat().st_size,
                        }
                    )
            final = {"train": s.scores("train"), "development": development}
            expected = epochs * len(split.train)
            if (
                s.cursor != expected
                or s.updates != steps * epochs
                or s.presentations != 2 * expected
            ):
                raise ValueError("complete fitting exposure mismatch")
            first, last = float(np.mean(losses[:steps])), float(np.mean(losses[-steps:]))
            passed = (
                final["train"]["auc"] >= 0.85
                and final["development"]["auc"] >= 0.70
                and last <= 0.8 * first
            )
            cells.append(
                {
                    "seed": seed,
                    "initial": initial,
                    "final": final,
                    "epochs": rows,
                    "losses": losses,
                    "checkpoints": checkpoints,
                    "passed": passed,
                    "updates": s.updates,
                    "presentations": s.presentations,
                }
            )
        del s
        profile.pilot.release(device)
    result = {
        "schema": "colab-rgb-fit-result-v1",
        "contract_sha256": binding,
        "status": "passed_exploratory_native_gate"
        if all(c["passed"] for c in cells)
        else "failed_learning_gate",
        "split": asdict(split),
        "cells": cells,
        "fit_updates": commits,
        "presentations": sum(c["presentations"] for c in cells),
        "physical_cuda": device == "cuda:0",
        "device": device,
        "epochs": epochs,
        "qualified": False,
        "deployment": False,
        "future_reserved_opened": False,
    }
    a.put(out / "fit.json", a.encode(result), 4 * a.MIB)
    return a.document(out / "fit.json", 4 * a.MIB)
```

