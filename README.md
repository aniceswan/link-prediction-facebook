# Perbandingan Sepuluh Metrik Kemiripan untuk Prediksi Tautan

Kode dan data eksperimen pendamping makalah Matematika Diskrit (IF1220) tentang
perbandingan sepuluh metrik kemiripan berbasis teori graf dan operasi himpunan
untuk rekomendasi teman ("Orang yang Mungkin Anda Kenal") pada jejaring sosial.

Seluruh metrik diturunkan dari operasi himpunan pada himpunan tetangga `N(u)`,
`N(v)` dan derajat `k(u)`, `k(v)`, lalu dievaluasi dengan validasi silang
Monte Carlo (20 ulangan, rasio uji 20%) memakai AUC dan Precision@k.

## Metrik yang dibandingkan

| Metrik | Rumus | Keluarga |
|--------|-------|----------|
| CN (Common Neighbors) | \|N(u) ∩ N(v)\| | cacah |
| Jaccard | \|N(u) ∩ N(v)\| / \|N(u) ∪ N(v)\| | ternormalisasi |
| Salton | \|N(u) ∩ N(v)\| / √(k(u)·k(v)) | ternormalisasi |
| Sørensen | 2\|N(u) ∩ N(v)\| / (k(u) + k(v)) | ternormalisasi |
| HPI | \|N(u) ∩ N(v)\| / min(k(u), k(v)) | ternormalisasi |
| HDI | \|N(u) ∩ N(v)\| / max(k(u), k(v)) | ternormalisasi |
| LHN | \|N(u) ∩ N(v)\| / (k(u)·k(v)) | ternormalisasi |
| RA (Resource Allocation) | Σ 1/k(w), w ∈ N(u) ∩ N(v) | berbobot |
| AA (Adamic–Adar) | Σ 1/log k(w), w ∈ N(u) ∩ N(v) | berbobot |
| PA (Preferential Attachment) | k(u)·k(v) | cacah (pembanding) |

## Dataset

[SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) —
jaringan pertemanan Facebook yang telah dianonimkan, dikumpulkan oleh
McAuley & Leskovec (2012). 4.039 simpul, 88.234 sisi, derajat rata-rata 43,69.
Berkas `data/facebook_combined.txt` berisi daftar sisi (satu pasang simpul per
baris). Dataset ini **bukan milik penulis**; sumber dan sitasi lengkap ada di
makalah.

## Cara menjalankan

Hanya membutuhkan Python 3 (pustaka standar) dan `matplotlib` untuk figur.

```bash
pip install matplotlib

# Jalankan eksperimen Monte Carlo -> menghasilkan results.json
python link_prediction.py

# Buat figur dari results.json -> folder gambar/
python make_figures.py
```

## Isi repositori

```
link_prediction.py        eksperimen utama (10 metrik, Monte Carlo, AUC, Precision@k)
make_figures.py           pembuatan figur dari results.json
results.json              keluaran numerik eksperimen
hasil_eksperimen.txt      keluaran teks lengkap (mudah dibaca manusia)
data/facebook_combined.txt  dataset SNAP ego-Facebook
gambar/                   figur hasil (.png)
```

## Lisensi

Kode dirilis untuk keperluan akademik/edukasi. Dataset mengikuti ketentuan
penggunaan SNAP (lihat tautan di atas).
