"""
Perbandingan sepuluh metrik kemiripan berbasis tetangga untuk prediksi tautan
(rekomendasi teman "Orang yang Mungkin Anda Kenal") pada jejaring sosial,
dievaluasi dengan validasi silang Monte Carlo pada data Facebook nyata.

Seluruh metrik diturunkan dari operasi himpunan pada himpunan tetangga N(u),
N(v) dan derajat k(u), k(v):
    CN        : |N(u) ∩ N(v)|
    Jaccard   : |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    Salton    : |N(u) ∩ N(v)| / sqrt(k(u) k(v))
    Sorensen  : 2|N(u) ∩ N(v)| / (k(u) + k(v))
    HPI       : |N(u) ∩ N(v)| / min(k(u), k(v))
    HDI       : |N(u) ∩ N(v)| / max(k(u), k(v))
    LHN       : |N(u) ∩ N(v)| / (k(u) k(v))
    RA        : Σ_{w ∈ N(u)∩N(v)} 1 / k(w)
    AA        : Σ_{w ∈ N(u)∩N(v)} 1 / log k(w)
    PA        : k(u) k(v)                       (basis pembanding)

Dataset: SNAP ego-Facebook (4.039 simpul, 88.234 sisi).
"""

import math
import random
import statistics
from collections import defaultdict

DATA_PATH = "data/facebook_combined.txt"
SEED = 42
TEST_RATIO = 0.20
N_RUNS = 20                # banyaknya ulangan Monte Carlo
KS = (10, 50, 100, 500, 1000)

NAMA_METRIK = ["CN", "Jaccard", "Salton", "Sorensen", "HPI",
               "HDI", "LHN", "RA", "AA", "PA"]


# --------------------------------------------------------------------------
# 1. Pemuatan graf
# --------------------------------------------------------------------------
def muat_graf(path):
    sisi = set()
    simpul = set()
    with open(path) as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            a, b = baris.split()
            a, b = int(a), int(b)
            if a == b:
                continue
            sisi.add(frozenset((a, b)))
            simpul.add(a)
            simpul.add(b)
    return simpul, sisi


def adjacency(sisi):
    N = defaultdict(set)
    for e in sisi:
        u, v = tuple(e)
        N[u].add(v)
        N[v].add(u)
    return N


# --------------------------------------------------------------------------
# 2. Penyekoran: satu kali penelusuran irisan menghasilkan semua metrik
# --------------------------------------------------------------------------
def skor_semua(N, u, v):
    nu, nv = N[u], N[v]
    ku, kv = len(nu), len(nv)
    kecil, besar = (nu, nv) if ku <= kv else (nv, nu)
    inter = 0
    ra = 0.0
    aa = 0.0
    for w in kecil:
        if w in besar:
            inter += 1
            d = len(N[w])
            if d > 0:
                ra += 1.0 / d
            if d > 1:
                aa += 1.0 / math.log(d)
    union = ku + kv - inter
    return {
        "CN": float(inter),
        "Jaccard": inter / union if union else 0.0,
        "Salton": inter / math.sqrt(ku * kv) if ku and kv else 0.0,
        "Sorensen": 2.0 * inter / (ku + kv) if (ku + kv) else 0.0,
        "HPI": inter / min(ku, kv) if min(ku, kv) else 0.0,
        "HDI": inter / max(ku, kv) if max(ku, kv) else 0.0,
        "LHN": inter / (ku * kv) if ku and kv else 0.0,
        "RA": ra,
        "AA": aa,
        "PA": float(ku * kv),
    }


# --------------------------------------------------------------------------
# 3. Contoh kecil (verifikasi manual untuk makalah)
# --------------------------------------------------------------------------
def contoh_kecil():
    sisi = [("A", "C"), ("A", "D"), ("A", "E"),
            ("B", "C"), ("B", "D"), ("B", "F"),
            ("C", "G"), ("E", "G")]
    sisi = [frozenset(e) for e in sisi]
    N = adjacency(sisi)
    u, v = "A", "B"
    irisan = N[u] & N[v]
    gabungan = N[u] | N[v]
    print("=== Contoh kecil (verifikasi manual) ===")
    print(f"N({u}) = {sorted(N[u])}, k({u}) = {len(N[u])}")
    print(f"N({v}) = {sorted(N[v])}, k({v}) = {len(N[v])}")
    print(f"N(A) ∩ N(B) = {sorted(irisan)} -> |irisan| = {len(irisan)}")
    print(f"N(A) ∪ N(B) = {sorted(gabungan)} -> |gabungan| = {len(gabungan)}")
    print("derajat teman bersama: " +
          ", ".join(f"k({w})={len(N[w])}" for w in sorted(irisan)))
    skor = skor_semua(N, u, v)
    for nama in NAMA_METRIK:
        print(f"   {nama:<9s} = {skor[nama]:.4f}")
    print()
    return sisi


# --------------------------------------------------------------------------
# 4. Pemisahan train/test dan pembentukan kandidat
# --------------------------------------------------------------------------
def split_train_test(sisi, rasio, rng):
    sisi = list(sisi)
    rng.shuffle(sisi)
    n_test = int(len(sisi) * rasio)
    test = set(sisi[:n_test])
    train = set(sisi[n_test:])
    return train, test


def kandidat_pasangan(N_train, train_sisi):
    """Pasangan tak-bertetangga berjarak dua (punya >=1 teman bersama)."""
    kandidat = set()
    for tetangga in N_train.values():
        tetangga = list(tetangga)
        for i in range(len(tetangga)):
            for j in range(i + 1, len(tetangga)):
                pasangan = frozenset((tetangga[i], tetangga[j]))
                if pasangan not in train_sisi:
                    kandidat.add(pasangan)
    return kandidat


# --------------------------------------------------------------------------
# 5. Evaluasi
# --------------------------------------------------------------------------
def evaluasi(skor, label):
    """Hitung AUC (peringkat Mann-Whitney) dan Precision@k dengan sekali urut.

    skor, label: list paralel. AUC = peluang skor positif > skor negatif,
    seri dihitung 0,5. Precision@k = proporsi positif pada k skor tertinggi.
    """
    idx = sorted(range(len(skor)), key=lambda i: skor[i])   # menaik
    n = len(idx)
    n_pos = sum(label)
    n_neg = n - n_pos
    # AUC via jumlah peringkat positif (peringkat rata-rata untuk seri)
    auc = float("nan")
    if n_pos and n_neg:
        jum_pos = 0.0
        i = 0
        while i < n:
            j = i
            while j < n and skor[idx[j]] == skor[idx[i]]:
                j += 1
            r = (i + 1 + j) / 2.0
            for t in range(i, j):
                if label[idx[t]] == 1:
                    jum_pos += r
            i = j
        U = jum_pos - n_pos * (n_pos + 1) / 2.0
        auc = U / (n_pos * n_neg)
    # Precision@k dari k skor tertinggi (ekor kanan urutan menaik)
    precs = {}
    for k in KS:
        ambil = idx[max(0, n - k):]
        precs[k] = sum(label[i] for i in ambil) / len(ambil) if ambil else 0.0
    return auc, precs


# --------------------------------------------------------------------------
# 6. Satu ulangan Monte Carlo
# --------------------------------------------------------------------------
def satu_run(sisi, rasio, rng):
    train, test = split_train_test(sisi, rasio, rng)
    N_train = adjacency(train)
    kandidat = kandidat_pasangan(N_train, train)

    skor = {m: [] for m in NAMA_METRIK}
    label = []
    for p in kandidat:
        u, v = tuple(p)
        label.append(1 if p in test else 0)
        s = skor_semua(N_train, u, v)
        for m in NAMA_METRIK:
            skor[m].append(s[m])

    hasil = {}
    for m in NAMA_METRIK:
        auc, precs = evaluasi(skor[m], label)
        hasil[m] = {"auc": auc, "precision": precs}
    return hasil, len(kandidat), sum(label)


# --------------------------------------------------------------------------
# 7. Eksperimen utama: validasi silang Monte Carlo
# --------------------------------------------------------------------------
def eksperimen(n_runs=N_RUNS, rasio=TEST_RATIO, verbose=True):
    simpul, sisi = muat_graf(DATA_PATH)
    if verbose:
        derajat = [len(s) for s in adjacency(sisi).values()]
        print("=== Statistik graf (SNAP ego-Facebook) ===")
        print(f"Simpul : {len(simpul)} | Sisi : {len(sisi)}")
        print(f"Derajat rata-rata : {sum(derajat)/len(derajat):.2f} | "
              f"maksimum : {max(derajat)}")
        print(f"Ulangan Monte Carlo : {n_runs} | rasio uji : {rasio}\n")

    auc_runs = {m: [] for m in NAMA_METRIK}
    prec_runs = {m: {k: [] for k in KS} for m in NAMA_METRIK}
    info = None
    for r in range(n_runs):
        rng = random.Random(SEED + r)
        hasil, n_kand, n_pos = satu_run(sisi, rasio, rng)
        if info is None:
            info = (n_kand, n_pos)
        for m in NAMA_METRIK:
            auc_runs[m].append(hasil[m]["auc"])
            for k in KS:
                prec_runs[m][k].append(hasil[m]["precision"][k])
        if verbose:
            print(f"  run {r+1:2d}/{n_runs} selesai "
                  f"(kandidat={n_kand}, positif={n_pos})")

    ringkas = {}
    for m in NAMA_METRIK:
        a = auc_runs[m]
        mean = statistics.fmean(a)
        sd = statistics.pstdev(a) if len(a) > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(len(a)) if len(a) > 1 else 0.0
        precs = {k: statistics.fmean(prec_runs[m][k]) for k in KS}
        ringkas[m] = {"auc_mean": mean, "auc_sd": sd, "auc_ci": ci,
                      "auc_runs": a, "precision": precs}

    if verbose:
        print("\n=== Ringkasan AUC (rerata ± simpangan baku, %d run) ===" % n_runs)
        urut = sorted(NAMA_METRIK, key=lambda m: ringkas[m]["auc_mean"],
                      reverse=True)
        for m in urut:
            d = ringkas[m]
            print(f"  {m:<9s} AUC = {d['auc_mean']:.4f} ± {d['auc_sd']:.4f} "
                  f"(95% CI ±{d['auc_ci']:.4f})")
        terbaik = urut[0]
        print(f"\nMetrik terbaik (rerata AUC): {terbaik}")
        for m in urut[1:]:
            menang = sum(1 for x, y in zip(ringkas[terbaik]["auc_runs"],
                                           ringkas[m]["auc_runs"]) if x > y)
            print(f"  {terbaik} > {m} pada {menang}/{n_runs} run")
        print("\n=== Rerata Precision@k ===")
        head = "  metrik     " + "".join(f"P@{k:<6d}" for k in KS)
        print(head)
        for m in urut:
            row = "".join(f"{ringkas[m]['precision'][k]:<8.3f}" for k in KS)
            print(f"  {m:<9s}  {row}")
        print(f"\nInfo kandidat: total={info[0]}, positif={info[1]}")

    return ringkas, info


def sensitivitas(rasios=(0.10, 0.20, 0.30), n_runs=20):
    """Rerata AUC tiap metrik pada beberapa rasio uji."""
    print("\n=== Analisis sensitivitas terhadap rasio uji ===")
    tabel = {}
    for rasio in rasios:
        ringkas, _ = eksperimen(n_runs=n_runs, rasio=rasio, verbose=False)
        tabel[rasio] = {m: ringkas[m]["auc_mean"] for m in NAMA_METRIK}
        baris = ", ".join(f"{m}={tabel[rasio][m]:.3f}" for m in NAMA_METRIK)
        print(f"  rasio {rasio:.2f}: {baris}")
    return tabel


# --------------------------------------------------------------------------
# 8. Contoh rekomendasi konkret
# --------------------------------------------------------------------------
def rekomendasi_untuk(N_train, pengguna, top=5):
    kandidat = set()
    for teman in N_train[pengguna]:
        for ff in N_train[teman]:
            if ff != pengguna and ff not in N_train[pengguna]:
                kandidat.add(ff)
    skor = [(c, skor_semua(N_train, pengguna, c)["RA"]) for c in kandidat]
    skor.sort(key=lambda x: x[1], reverse=True)
    print(f"\n=== Top-{top} rekomendasi (metrik RA) untuk pengguna {pengguna} ===")
    for c, s in skor[:top]:
        print(f"   pengguna {c:<5d} | skor RA = {s:.4f} | "
              f"teman bersama = {len(N_train[pengguna] & N_train[c])}")


if __name__ == "__main__":
    import json
    contoh_kecil()
    ringkas, info = eksperimen()
    sens = sensitivitas()
    _, sisi = muat_graf(DATA_PATH)
    rng = random.Random(SEED)
    train, _ = split_train_test(sisi, TEST_RATIO, rng)
    rekomendasi_untuk(adjacency(train), pengguna=0, top=5)

    out = {"ringkas": ringkas, "info": list(info),
           "sensitivitas": {f"{r:.2f}": sens[r] for r in sens}}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nresults.json tersimpan.")
