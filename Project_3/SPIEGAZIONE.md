# Project 3 — Image Inpainting con U-Net: spiegazione del codice

## Il problema

Il task è **image inpainting**: date 10 000 immagini 28×28 in scala di grigi (cifre scritte a mano, stile MNIST) in cui la regione centrale 8×8 è stata azzerata, bisogna ricostruire quei pixel mancanti.

La rete viene addestrata su 60 000 immagini complete, a cui viene applicata artificialmente la maschera, così da avere sia l'input (immagine bucata) che il target (immagine originale). Solo il centro viene valutato a submission.

---

## Struttura del codice

Il file `template_solution.py` è organizzato in 5 blocchi principali:

1. **Setup del device** — dove gira il calcolo
2. **`load_data()`** — carica e prepara i dati
3. **`ConvBlock` + `Model`** — definizione della rete neurale
4. **`training()`** — addestramento della rete
5. **`testing()`** — inferenza e salvataggio del file di submission

---

## 1. Setup del device

```python
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

PyTorch può eseguire i calcoli su CPU o GPU. La GPU è molto più veloce per le reti neurali perché può fare migliaia di operazioni in parallelo.

- **`cuda`**: GPU NVIDIA (PC con scheda grafica dedicata)
- **`mps`**: GPU Apple Silicon (Mac M1/M2/M3) — Metal Performance Shaders
- **`cpu`**: fallback se non c'è nessuna GPU

Il device viene controllato una volta sola all'avvio. Poi tutti i tensori e il modello devono stare sullo stesso device, altrimenti PyTorch va in errore.

---

## 2. `load_data()` — Caricamento e preparazione dei dati

### Caricamento

```python
train_data = np.load("train_data.npz")["data"]
train_data = torch.tensor(train_data, dtype=torch.float32)

test_data_input = np.load("test_data.npz")["data"]
test_data_input = torch.tensor(test_data_input, dtype=torch.float32)
```

I file `.npz` sono archivi NumPy compressi. `["data"]` accede alla chiave che contiene l'array.

`torch.tensor(..., dtype=torch.float32)` converte l'array NumPy (che è `uint8`, cioè interi 0–255) in un tensore PyTorch di float a 32 bit. Le reti neurali lavorano con numeri floating point, non interi.

La shape risultante è `[N, 1, 28, 28]`:
- `N`: numero di immagini (60 000 train, 10 000 test)
- `1`: numero di canali (1 = scala di grigi; per immagini RGB sarebbe 3)
- `28, 28`: altezza e larghezza in pixel

### Normalizzazione

```python
train_data = train_data / 255.0
test_data_input = test_data_input / 255.0
```

I pixel originali vanno da 0 a 255. Dividiamo per 255 per portarli nel range `[0, 1]`.

**Perché normalizzare?** Le reti neurali convergono molto meglio con valori piccoli. Con valori grandi (0–255) i gradienti diventano instabili e il training è lento. Normalizzare è una prassi standard quasi sempre obbligatoria.

### Creazione delle coppie input/label

```python
train_data_label = train_data.clone()       # immagine completa = target
train_data_input  = train_data.clone()      # copia su cui applico la maschera
train_data_input[:, :, 10:18, 10:18] = 0.0 # azzero il centro 8×8
```

`.clone()` crea una copia indipendente del tensore (senza clone, modificare `train_data_input` modificherebbe anche `train_data_label` perché punterebbero alla stessa memoria).

`[:, :, 10:18, 10:18]` è uno slice 4D:
- `:` → tutte le immagini
- `:` → tutti i canali
- `10:18` → righe dalla 10 alla 17 (8 righe)
- `10:18` → colonne dalla 10 alla 17 (8 colonne)

Questo replica esattamente la maschera già applicata nelle immagini di test.

### Visualizzazione (opzionale)

```python
for i in tqdm(range(20), desc="Plotting train images"):
    plt.subplot(1, 2, 1)
    plt.imshow(train_data_input[i].squeeze(), cmap="gray")
    plt.title("Training Input")
    plt.subplot(1, 2, 2)
    plt.imshow(train_data_label[i].squeeze(), cmap="gray")
    plt.title("Training Label")
    plt.savefig(f"train_image_output/image_{i}.png")
    plt.close()
```

Salva le prime 20 coppie input/label come immagini PNG nella cartella `train_image_output/`. Utile per verificare visivamente che i dati siano stati preparati correttamente. `.squeeze()` rimuove la dimensione dei canali (da `[1, 28, 28]` a `[28, 28]`) perché `imshow` vuole un array 2D per immagini in scala di grigi.

---

## 3. L'architettura: U-Net

### Perché una U-Net?

L'inpainting è un problema **image-to-image**: l'input è un'immagine (bucata) e l'output è un'immagine (ricostruita). La U-Net è nata per questo tipo di problemi. Ha due caratteristiche fondamentali:

1. **Encoder-decoder**: l'encoder comprime l'immagine catturando il contesto ("questo è un 7 scritto in un certo modo"), il decoder ricostruisce l'immagine a piena risoluzione.
2. **Skip connections**: collegano direttamente ogni livello dell'encoder al livello corrispondente del decoder, così il decoder può recuperare i dettagli fini (bordi, spessori dei tratti) che andrebbero persi nella compressione.

### `ConvBlock` — il mattone base

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
```

Ogni `ConvBlock` è una sequenza di due volte: **Conv2d → BatchNorm → ReLU**.

- **`Conv2d(in_ch, out_ch, kernel_size=3, padding=1)`**: convoluzione 2D con kernel 3×3. Scorre il kernel su tutta l'immagine e impara a rilevare pattern locali (bordi, curve, texture). `padding=1` aggiunge un bordo di zeri attorno all'immagine così l'output ha le stesse dimensioni spaziali dell'input.
- **`BatchNorm2d`**: normalizza i valori all'interno del batch. Stabilizza il training, permette learning rate più alti e rende la rete più robusta. Praticamente obbligatorio nelle reti CNN moderne.
- **`ReLU(inplace=True)`**: funzione di attivazione non lineare — mette a zero tutti i valori negativi. Senza non-linearità, una rete con molti layer sarebbe equivalente a un solo layer lineare. `inplace=True` risparmia memoria modificando il tensore direttamente.

`nn.Sequential` incatena i layer: l'output di uno diventa l'input del successivo.

### `Model` — la U-Net completa

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)      # 28×28 → 14×14
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)      # 14×14 → 7×7
        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(64 + 32, 32)
        # Output
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
```

**Encoder** (compressione progressiva):
- `enc1`: 1 canale → 32 feature maps, dimensione spaziale invariata (28×28)
- `pool1`: `MaxPool2d(2)` dimezza la risoluzione (28×28 → 14×14). Prende il massimo in ogni finestra 2×2. Aumenta il campo recettivo — ogni neurone nel livello successivo "vede" una porzione più grande dell'immagine originale.
- `enc2`: 32 → 64 feature maps a 14×14
- `pool2`: 14×14 → 7×7

**Bottleneck**: il punto di massima compressione. 64 → 128 feature maps a 7×7. Qui il modello deve "capire" globalmente l'immagine (es. che cifra è, come è orientata) per poi ricostruire il centro.

**Decoder** (ricostruzione progressiva):
- `up1`: `Upsample(scale_factor=2, mode="bilinear")` raddoppia la risoluzione (7×7 → 14×14) interpolando i pixel. Bilineare è più fluido rispetto a nearest-neighbor e non produce gli artefatti a scacchiera delle ConvTranspose2d.
- `dec1`: prende `torch.cat([up1(b), s2], dim=1)` — concatena lungo la dimensione dei canali l'output dell'upsampling (128 ch) con la skip connection dall'encoder (64 ch) → 192 canali totali → riduce a 64.
- `up2`: 14×14 → 28×28
- `dec2`: concatena dec1 (64 ch) + skip s1 (32 ch) → 96 canali → riduce a 32

**Head** (output finale):
- `Conv2d(32, 1, kernel_size=1)`: convoluzione 1×1, combina linearmente le 32 feature maps in 1 canale (l'immagine ricostruita). Il kernel 1×1 non guarda il vicinato, agisce pixel per pixel.
- `Sigmoid()`: schiaccia l'output in `[0, 1]`, compatibile con i label normalizzati.

### Forward pass

```python
def forward(self, x):
    s1 = self.enc1(x)                                    # [N, 32, 28, 28]
    s2 = self.enc2(self.pool1(s1))                       # [N, 64, 14, 14]
    b  = self.bottleneck(self.pool2(s2))                 # [N, 128, 7, 7]
    d1 = self.dec1(torch.cat([self.up1(b), s2], dim=1)) # [N, 64, 14, 14]
    d2 = self.dec2(torch.cat([self.up2(d1), s1], dim=1))# [N, 32, 28, 28]
    return self.head(d2)                                 # [N, 1, 28, 28]
```

`s1` e `s2` vengono salvati come variabili perché servono dopo come skip connections — sono il "ricordo" dei dettagli spaziali prima della compressione.

`torch.cat([A, B], dim=1)` concatena due tensori lungo la dimensione dei canali (dim=1). Se A ha 128 canali e B ne ha 64, il risultato ne ha 192.

---

## 4. `training()` — Addestramento

### Inizializzazione

```python
model = Model()
model.train()
model.to(device)
```

`Model()` istanzia la rete con pesi casuali. `model.train()` attiva le modalità specifiche del training (BatchNorm usa le statistiche del batch corrente, Dropout — se presente — è attivo). `model.to(device)` sposta tutti i parametri sul device scelto.

### Loss function

```python
def masked_mse(pred, target):
    return F.mse_loss(pred[:, :, 10:18, 10:18], target[:, :, 10:18, 10:18])
criterion = masked_mse
```

La **loss** misura quanto le predizioni del modello si discostano dal target. Usiamo **MSE** (Mean Squared Error) — media dei quadrati degli errori pixel per pixel.

Il calcolo viene fatto **solo sulla patch centrale** `[10:18, 10:18]`. Questo è cruciale: se calcolassimo la loss sull'intera immagine, la rete sprecherebbe capacità a "imparare" i bordi che sono già corretti nell'input. Concentrandosi solo sul centro, ogni passo di aggiornamento va interamente a migliorare la qualità dell'inpainting — esattamente quello che viene valutato.

### Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

L'**optimizer** aggiorna i pesi della rete in base al gradiente della loss. **Adam** è lo standard de facto: adatta automaticamente il learning rate per ogni parametro e converge velocemente. `lr=1e-3` (0.001) è un buon punto di partenza per Adam.

### DataLoader

```python
batch_size = 256
dataset = TensorDataset(train_data_input, train_data_label)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

`TensorDataset` accoppia input e label: ogni elemento `i` restituisce `(train_data_input[i], train_data_label[i])`.

`DataLoader` gestisce l'iterazione sui dati in **batch** (mini-gruppi). Invece di aggiornare i pesi su una sola immagine alla volta (troppo rumoroso) o su tutte le 60k contemporaneamente (troppa memoria), usiamo batch da 256 immagini — un buon compromesso.

`shuffle=True` mescola i dati a ogni epoca. Senza shuffle, la rete vedrebbe sempre prima tutti gli zeri, poi tutti gli uni, ecc. — il gradiente sarebbe molto biased e il training instabile.

### Loop di training

```python
for epoch in range(n_epochs):
    for x, y in tqdm(data_loader, ...):
        x, y = x.to(device), y.to(device)  # sposta il batch sul device
        optimizer.zero_grad()               # azzera i gradienti accumulati
        output = model(x)                   # forward pass
        loss = criterion(output, y)         # calcola la loss
        loss.backward()                     # backward pass: calcola i gradienti
        optimizer.step()                    # aggiorna i pesi
```

Un'**epoca** è un passaggio completo su tutti i 60k esempi di training. Con batch_size=256, ogni epoca fa 235 iterazioni.

`optimizer.zero_grad()` è necessario perché PyTorch accumula i gradienti di default — se non si azzerano, i gradienti delle iterazioni precedenti si sommano a quelli correnti.

`loss.backward()` calcola le derivate parziali della loss rispetto a ogni parametro della rete (backpropagation). `optimizer.step()` usa quelle derivate per spostare i pesi nella direzione che riduce la loss.

Con **15 epoche** la rete vede ogni immagine 15 volte, convergendo a una loss MSE ~0.038 sul centro.

---

## 5. `testing()` — Inferenza e submission

### Modalità eval e no_grad

```python
model.eval()
with torch.no_grad():
    ...
```

`model.eval()` disattiva BatchNorm (usa statistiche globali calcolate durante il training invece di quelle del batch corrente) e Dropout (se presente). Fondamentale per avere predizioni deterministiche e corrette.

`torch.no_grad()` disabilita il calcolo dei gradienti — non serve per l'inferenza e risparmia memoria + tempo.

### Inferenza a batch

```python
for i in range(0, test_data_input.shape[0], batch_size):
    output = model(test_data_input[i : i + batch_size])
    test_data_output.append(output.cpu())
test_data_output = torch.cat(test_data_output)
```

Anche in test si processa per batch per non esaurire la RAM/VRAM. `.cpu()` sposta ogni batch di output dalla GPU alla CPU (necessario prima di convertire in NumPy). `torch.cat` concatena tutti i batch in un unico tensore finale.

### Rescaling e salvataggio

```python
test_data_output = test_data_output * 255.0       # [0,1] → [0,255]
save_data_clipped = np.clip(test_data_output, 0, 255)  # garantisce il range
save_data_uint8 = save_data_clipped.astype(np.uint8)   # converte in interi

save_data = np.zeros_like(save_data_uint8)
save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

np.savez_compressed("submit_this_test_data_output.npz", data=save_data)
```

Poiché i dati erano stati normalizzati a `[0,1]` e il modello produce `Sigmoid → [0,1]`, moltiplichiamo per 255 per tornare al range originale `[0,255]`.

`np.clip` taglia eventuali valori fuori range (può succedere per errori numerici). `.astype(np.uint8)` converte in interi a 8 bit (0–255).

Poi si crea un array di soli zeri e si copiano **solo le predizioni della patch centrale**: il resto dell'immagine non viene valutato, quindi non c'è motivo di salvarlo (risparmio di spazio nel file compresso).

Infine `np.savez_compressed` salva il file `.npz` compresso — questo è il file da consegnare.

---

## 6. `main()` — Entry point

```python
def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    train_data_input, train_data_label, test_data_input = load_data()
    model = training(train_data_input, train_data_label)
    testing(model, test_data_input)
```

Impostare un seed fisso (`seed=0`) rende il training **riproducibile**: inizializzando i generatori di numeri casuali sempre allo stesso modo, due esecuzioni identiche producono lo stesso modello. Utile per il debug.

Poi orchestra la pipeline: carica i dati → addestra il modello → genera il file di submission.

---

## Riepilogo visivo del flusso

```
train_data.npz          test_data.npz
(60k immagini intere)   (10k immagini mascherate)
        │                       │
        ▼                       ▼
   load_data()          normalizza /255
   • normalizza /255
   • label = immagine intera
   • input = immagine con centro azzerato
        │
        ▼
   training()
   • U-Net encoder-decoder
   • Loss MSE solo sul centro 8×8
   • Adam, 15 epoche, batch 256
        │
        ▼ model addestrato
   testing()
   • inferenza sul test set
   • ×255, clip, uint8
   • salva solo patch [10:18,10:18]
        │
        ▼
submit_this_test_data_output.npz  ← file da consegnare
```
