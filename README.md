# Membuat model CNN untuk klasifikasi penyakit pada daun ceri

## Dataset

Total **3.604 gambar** daun ceri dalam 5 kelas:

| Kelas | Jumlah gambar |
|---|---|
| Leaf Scorch | 1.098 |
| Purple Leaf Spot | 983 |
| Brown Spot | 613 |
| Normal | 473 |
| Shot Hole | 437 |

Dataset **tidak seimbang** (Leaf Scorch ±2,5x lebih banyak dari Shot Hole), sehingga pada model modifikasi digunakan *class weight*.

---

## Alur Pengerjaan

### 1. Persiapan data & EDA (LO2)
- Set seed (`1234`) untuk reprodusibilitas.
- **Pengecekan duplikat** menggunakan perceptual hash (`imagehash.phash`) per kelas; duplikat akan dihapus. Setelah proses ini jumlah data tetap 3.604 gambar.
- **EDA**: sebaran resolusi (scatter lebar vs tinggi) dan sebaran *aspect ratio* seluruh gambar.
- **Split 70 : 15 : 15** setelah data diacak:

| Split | Jumlah |
|---|---|
| Train | 2.522 |
| Validation | 540 |
| Test | 542 |

### 2. Pre-processing & augmentasi
- Resize ke **224 × 224**, normalisasi piksel ke `[0, 1]`, label *one-hot*, batch size 32.
- Augmentasi (hanya pada data train): `RandomFlip` (horizontal & vertikal), `RandomRotation(0.2)`, `RandomZoom(0.2)`, `RandomContrast(0.1)`.

### 3. Baseline: AlexNet manual (LO2–LO4)
Dibangun dari nol (tanpa *pre-trained*), 5 layer konvolusi (96 → 256 → 384 → 384 → 256 filter), 3 layer max-pooling, dua Dense 4096 + Dropout 0,5, dan output softmax 5 kelas. Total parameter ±**46,8 juta**. Optimizer Adam, loss categorical cross-entropy, *early stopping* (`val_loss`, patience 7).

### 4. Modifikasi: Custom CNN (LO1–LO4)
3 blok konvolusi (64 → 128 → 256 filter, kernel 3×3) dengan **BatchNormalization**, **MaxPooling**, dan **Dropout 0,25** di tiap blok, lalu Dense 512 + BN + Dropout 0,5. Optimizer Adam (`lr = 1e-4`), **class weight** *balanced*, *early stopping* (patience 7). Total parameter ±**103,2 juta**.

### 5. Evaluasi (LO2–LO4)
Evaluasi pada data test menggunakan **Accuracy, Precision, Recall, dan F1-score** (`classification_report`). ROC-AUC tidak dipakai karena kasusnya *multiclass*. Recall diberi perhatian khusus karena kesalahan memprediksi daun sakit sebagai daun sehat (*false negative*) dapat menyebabkan penyakit menyebar.

---

## Hasil (data test, 542 gambar)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **AlexNet (baseline)** | **0,93** | **0,91** | **0,93** |
| Custom CNN (modifikasi) | 0,61 | 0,56 | 0,64 |

**Per kelas — AlexNet (baseline)**

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Brown Spot | 0,96 | 0,95 | 0,95 | 110 |
| Normal | 0,70 | 0,95 | 0,81 | 60 |
| Purple Leaf Spot | 0,96 | 0,97 | 0,97 | 136 |
| Leaf Scorch | 0,98 | 0,96 | 0,97 | 164 |
| Shot Hole | 0,96 | 0,75 | 0,84 | 72 |

**Per kelas — Custom CNN**

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Brown Spot | 0,48 | 0,55 | 0,52 | 110 |
| Normal | 0,21 | 0,50 | 0,29 | 60 |
| Purple Leaf Spot | 0,97 | 0,57 | 0,72 | 136 |
| Leaf Scorch | 0,90 | 0,88 | 0,89 | 164 |
| Shot Hole | 0,67 | 0,28 | 0,39 | 72 |

> Nama kelas di atas disusun berdasarkan urutan indeks pada `classes = ['brown', 'normal', 'purple', 'scorch', 'shot hole']`.

### Ringkasan proses training

| Model | Data | Epoch berjalan | Epoch terbaik | Val. accuracy terakhir |
|---|---|---|---|---|
| AlexNet | Asli | 25 (early stop) | 18 | ±0,86 |
| AlexNet | Augmentasi | 50 | 49 | ±0,91 |
| Custom CNN | Asli | 22 (early stop) | 15 | ±0,70 (train acc 1,00 → overfitting) |
| Custom CNN | Augmentasi | 7 (early stop) | 1 | ±0,74 |

### Analisis singkat
- AlexNet baseline memberikan generalisasi yang jauh lebih baik dibanding Custom CNN pada data test.
- Augmentasi membantu AlexNet lebih general (val. accuracy naik dan gap dengan training mengecil), meski membutuhkan lebih banyak epoch.
- Custom CNN mengalami **overfitting berat** (train accuracy ±100%, val. accuracy jauh tertinggal dan val. loss > 2).
- Kelas **Normal** dan **Shot Hole** paling sulit diprediksi, sejalan dengan jumlah datanya yang paling sedikit.

### Saran pengembangan
- Fine-tuning model *pre-trained* seperti EfficientNet/DenseNet (membuka *freeze* pada layer atas).
- Menerapkan cross-validation / K-Fold untuk evaluasi yang lebih kuat.
- Mencoba segmentasi untuk melokalisasi area penyakit pada daun.
- Tuning hyperparameter Custom CNN (learning rate, ukuran Dense layer, kekuatan Dropout).

---

## Catatan
- Kedua model dilatih dua kali secara berurutan pada objek model yang sama (data asli, lalu data augmentasi), sehingga hasil evaluasi test mencerminkan bobot setelah tahap training kedua.
- Sel evaluasi memberi label "EfficientNet" untuk model modifikasi, padahal model tersebut adalah Custom CNN (bukan EfficientNet).
