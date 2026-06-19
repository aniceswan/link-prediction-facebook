"""Membuat figur untuk makalah: contoh graf, distribusi derajat, AUC, Precision@k, ragam Monte Carlo.

Sumber data hasil: results.json (dihasilkan oleh link_prediction.py).
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

import link_prediction as lp

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT = "gambar"
KS = [10, 50, 100, 500, 1000]


def muat_hasil(path="results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def urutkan(ringkas):
    """Daftar nama metrik diurut menurun berdasar AUC rata-rata."""
    return sorted(ringkas, key=lambda m: ringkas[m]["auc_mean"], reverse=True)


# Nama tampilan agar ejaan pada figur konsisten dengan teks makalah.
LABEL = {"Sorensen": "Sørensen"}


def label(m):
    return LABEL.get(m, m)


# --------------------------------------------------------------------------
# Gambar 1: contoh graf kecil (untuk bagian metode)
# --------------------------------------------------------------------------
def gambar_contoh():
    pos = {
        "A": (-2.0, 0.0), "B": (2.0, 0.0),
        "C": (0.0, 1.3), "D": (0.0, -1.3),
        "E": (-3.0, 1.4), "F": (3.0, -1.4), "G": (-1.4, 2.6),
    }
    sisi = [("A", "C"), ("A", "D"), ("A", "E"),
            ("B", "C"), ("B", "D"), ("B", "F"),
            ("C", "G"), ("E", "G")]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for u, v in sisi:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="#888888", lw=1.6, zorder=1)
    warna = {"A": "#1f77b4", "B": "#1f77b4", "C": "#d62728", "D": "#d62728"}
    for n, (x, y) in pos.items():
        c = warna.get(n, "#cccccc")
        ax.scatter([x], [y], s=900, color=c, edgecolors="black",
                   zorder=2, linewidths=1.2)
        ax.text(x, y, n, ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", zorder=3)
    ax.set_title("Pasangan (A, B) yang belum berteman;\n"
                 "C dan D adalah teman bersama (irisan tetangga)")
    ax.axis("off")
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-2.1, 3.2)
    fig.savefig(f"{OUT}/contoh_graf.png")
    plt.close(fig)
    print("OK contoh_graf.png")


# --------------------------------------------------------------------------
# Gambar 2: distribusi derajat data Facebook (menegaskan ini graf sosial nyata)
# --------------------------------------------------------------------------
def gambar_distribusi_derajat():
    _, sisi = lp.muat_graf(lp.DATA_PATH)
    N = lp.adjacency(sisi)
    derajat = [len(s) for s in N.values()]
    cnt = Counter(derajat)
    xs = sorted(cnt)
    ys = [cnt[x] for x in xs]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.loglog(xs, ys, marker="o", linestyle="none", markersize=3,
              color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Derajat simpul (jumlah teman)")
    ax.set_ylabel("Jumlah pengguna")
    ax.set_title("Distribusi derajat dataset ego-Facebook (skala log-log)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.savefig(f"{OUT}/distribusi_derajat.png")
    plt.close(fig)
    print("OK distribusi_derajat.png")


# --------------------------------------------------------------------------
# Gambar 3: AUC rata-rata semua metrik dengan error bar (simpangan baku MC)
# --------------------------------------------------------------------------
def gambar_auc(ringkas):
    nama = urutkan(ringkas)
    mean = [ringkas[m]["auc_mean"] for m in nama]
    sd = [ringkas[m]["auc_sd"] for m in nama]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = range(len(nama))
    bars = ax.bar(x, mean, yerr=sd, capsize=3, color="#4c78a8",
                  edgecolor="black", linewidth=0.7,
                  error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
    ax.axhline(0.5, color="#d62728", ls="--", lw=1.0, label="tebakan acak (0,5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(m) for m in nama], rotation=45, ha="right",
                       fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC (rata-rata 20 ulangan)")
    ax.set_title("AUC tiap metrik kemiripan pada data uji ego-Facebook")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower left")
    for b, m in zip(bars, mean):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.savefig(f"{OUT}/auc.png")
    plt.close(fig)
    print("OK auc.png")


# --------------------------------------------------------------------------
# Gambar 4: Precision@k semua metrik
# --------------------------------------------------------------------------
def gambar_precision(ringkas):
    nama = urutkan(ringkas)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for i, m in enumerate(nama):
        ys = [ringkas[m]["precision"][str(k)] for k in KS]
        ax.plot(KS, ys, marker="o", markersize=4, linewidth=1.5,
                color=cmap(i % 10), label=label(m))
    ax.set_xscale("log")
    ax.set_xlabel("k (jumlah rekomendasi teratas)")
    ax.set_ylabel("Precision@k")
    ax.set_title("Presisi rekomendasi terhadap k")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.savefig(f"{OUT}/precision_k.png")
    plt.close(fig)
    print("OK precision_k.png")


if __name__ == "__main__":
    gambar_contoh()
    gambar_distribusi_derajat()
    hasil = muat_hasil()
    ringkas = hasil["ringkas"]
    gambar_auc(ringkas)
    gambar_precision(ringkas)
    print("Semua figur selesai dibuat di folder gambar/")
