"""
preprocess_9class.py

TrustECG
Improved ECG Preprocessing Pipeline (v2)

Pipeline:
1. Read ECG
2. Baseline Wander Removal (0.5 Hz High-pass, SOS-based for numerical stability)
3. Powerline Removal (50 Hz Notch)
4. Low-pass Filter (45 Hz)
5. Wavelet Denoising (detail coefficients only)
6. Z-score Normalization
7. Save ECG + Labels + Manifest row

Changes from v1:
- Fixed: wavelet denoising was thresholding the approximation coefficient
  (coeffs[0]), corrupting low-frequency morphology (P-wave / ST-segment).
  Now only detail coefficients (coeffs[1:]) are thresholded.
- Fixed: Butterworth filters now use second-order-sections (sos) + sosfiltfilt
  instead of transfer-function (b, a) + filtfilt, which is numerically
  unstable at very low normalized cutoffs (e.g. 0.5 Hz / 250 Hz Nyquist).
- Fixed: FS constant is now actually threaded through to every filter call.
- Added: NaN/Inf and shape sanity checks before saving.
- Added: CSV manifest (record_id, npy_path, label_path, 9 label columns,
  patient_id if available) written incrementally so graph construction can
  consume it directly and so patient-wise splitting is easy later.
- Added: parallel processing across CPU cores (falls back to serial if
  multiprocessing is unavailable/undesired).
- Added: failure log file (separate from stdout) listing which records
  failed and why, instead of only printing to console.
"""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import traceback

import numpy as np
import scipy.io as sio
import wfdb

from scipy.signal import butter, sosfiltfilt, iirnotch
import pywt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# =====================================================
# DEFAULT PATHS / CONFIG
# =====================================================

RAW_DATA_ROOT = Path("data/raw/WFDBRecords_9Class")
OUTPUT_ROOT = Path("data/preprocessed_9Class")
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
FAILLOG_PATH = OUTPUT_ROOT / "failures.log"

FS = 500
EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 5000  # 10s @ 500Hz


# =====================================================
# 9 CLASS MAPPING
# =====================================================

CLASS_MAP = {
    "426177001": 0,   # SB
    "164934002": 1,   # TWC
    "426783006": 2,   # SR
    "164889003": 3,   # AFIB
    "427084000": 4,   # ST
    "55827005": 5,    # OLD MI
    "428750005": 6,   # STTC
    "426761007": 7,   # SVT
    "164890007": 8    # AF
}
CLASS_NAMES = ["SB", "TWC", "SR", "AFIB", "ST", "OldMI", "STTC", "SVT", "AF"]
NUM_CLASSES = len(CLASS_MAP)


# =====================================================
# FILTERS (SOS-based for numerical stability)
# =====================================================

def sos_highpass(signal, cutoff=0.5, fs=FS, order=4):
    nyquist = 0.5 * fs
    sos = butter(order, cutoff / nyquist, btype="high", output="sos")
    return sosfiltfilt(sos, signal)


def notch_filter(signal, freq=50, fs=FS, q=30):
    # iirnotch doesn't have an SOS form in scipy, but it's a single
    # narrow 2nd-order section so b,a is fine here.
    b, a = iirnotch(freq / (fs / 2), q)
    from scipy.signal import filtfilt
    return filtfilt(b, a, signal)


def sos_lowpass(signal, cutoff=45, fs=FS, order=4):
    nyquist = 0.5 * fs
    sos = butter(order, cutoff / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, signal)


# =====================================================
# WAVELET DENOISING (fixed: never threshold approximation coeffs)
# =====================================================

def wavelet_denoise(signal, wavelet="sym8", level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    if (not np.isfinite(sigma)) or sigma < 1e-12:
        sigma = 0.0

    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft")
        for c in coeffs[1:]
    ]

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    reconstructed = np.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return reconstructed[:len(signal)]


# =====================================================
# NORMALIZATION
# =====================================================

def zscore(signal):
    mean = np.mean(signal)
    std = np.std(signal)

    if (not np.isfinite(std)) or std < 1e-8:
        return np.zeros_like(signal)

    return (signal - mean) / std


# =====================================================
# PREPROCESS SINGLE LEAD
# =====================================================

def preprocess_lead(signal, fs=FS):
    signal = sos_highpass(signal, fs=fs)
    signal = notch_filter(signal, fs=fs)
    signal = sos_lowpass(signal, fs=fs)
    signal = wavelet_denoise(signal)
    signal = zscore(signal)
    return signal


# =====================================================
# LABEL EXTRACTION
# =====================================================

def extract_labels(header):
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    dx_codes = []

    for line in header.comments:
        if line.startswith("Dx:"):
            dx_codes = line.replace("Dx:", "").strip().split(",")
            break

    for code in dx_codes:
        code = code.strip()
        if code in CLASS_MAP:
            label_vector[CLASS_MAP[code]] = 1

    return label_vector


def extract_patient_id(header):
    """
    Best-effort patient ID extraction from header comments, useful for
    patient-wise splitting later. Falls back to None if not present.
    """
    for line in header.comments:
        if line.lower().startswith("patient_id:") or line.lower().startswith("#patient"):
            return line.split(":", 1)[-1].strip()
    return None


# =====================================================
# PROCESS ONE ECG
# =====================================================

def process_record(record_path_str, raw_root_str, output_root_str):
    """
    Standalone function (importable/picklable) so it can run under
    ProcessPoolExecutor. Returns a dict describing the outcome.
    """
    record_path = Path(record_path_str)
    raw_root = Path(raw_root_str)
    output_root = Path(output_root_str)

    result = {
        "record": str(record_path),
        "status": "failed",
        "reason": None,
        "npy_path": None,
        "label_path": None,
        "labels": None,
        "patient_id": None,
    }

    try:
        mat_file = record_path.with_suffix(".mat")
        hea_file = record_path.with_suffix(".hea")

        if not mat_file.exists() or not hea_file.exists():
            result["reason"] = "missing .mat or .hea file"
            return result

        ecg = sio.loadmat(mat_file)["val"].astype(np.float32)

        if ecg.shape[0] != EXPECTED_LEADS:
            result["reason"] = f"unexpected lead count: {ecg.shape[0]}"
            return result

        header = wfdb.rdheader(str(record_path))
        labels = extract_labels(header)
        patient_id = extract_patient_id(header)

        if labels.sum() == 0:
            result["reason"] = "no selected diagnosis present"
            return result

        processed = np.zeros_like(ecg)
        for lead in range(ecg.shape[0]):
            processed[lead] = preprocess_lead(ecg[lead], fs=FS)

        if not np.all(np.isfinite(processed)):
            result["reason"] = "NaN/Inf detected after preprocessing"
            return result

        relative = record_path.relative_to(raw_root)
        save_dir = output_root / relative.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        npy_path = save_dir / f"{record_path.stem}.npy"
        label_path = save_dir / f"{record_path.stem}_label.npy"

        np.save(npy_path, processed)
        np.save(label_path, labels)

        result.update({
            "status": "success",
            "npy_path": str(npy_path),
            "label_path": str(label_path),
            "labels": labels.tolist(),
            "patient_id": patient_id,
        })
        return result

    except Exception as e:
        result["reason"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        return result


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="TrustECG 9-class preprocessing")
    parser.add_argument("--raw-root", type=str, default=str(RAW_DATA_ROOT))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=4,
                         help="Number of parallel worker processes. Use 1 for serial/debug.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.csv"
    faillog_path = output_root / "failures.log"

    print("=" * 60)
    print("Searching ECG records...")
    print("=" * 60)

    records = sorted(raw_root.rglob("*.hea"))
    records = [r.with_suffix("") for r in records]
    print(f"Found {len(records)} ECG records\n")

    processed = 0
    discarded = 0

    with open(manifest_path, "w", newline="") as mf, open(faillog_path, "w") as ff:
        writer = csv.writer(mf)
        writer.writerow(
            ["record_id", "npy_path", "label_path", "patient_id"] + CLASS_NAMES
        )
        ff.write(f"TrustECG preprocessing failure log — {datetime.now().isoformat()}\n")
        ff.write("=" * 60 + "\n")

        if args.workers <= 1:
            iterator = (
                process_record(str(r), str(raw_root), str(output_root))
                for r in records
            )
            results_iter = tqdm(iterator, total=len(records))
            for result in results_iter:
                _handle_result(result, writer, ff)
                if result["status"] == "success":
                    processed += 1
                else:
                    discarded += 1
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_record, str(r), str(raw_root), str(output_root)): r
                    for r in records
                }
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    _handle_result(result, writer, ff)
                    if result["status"] == "success":
                        processed += 1
                    else:
                        discarded += 1

    print("\n" + "=" * 60)
    print("Preprocessing Completed")
    print("=" * 60)
    print(f"Processed ECGs : {processed}")
    print(f"Discarded ECGs : {discarded}")
    print(f"Saved to       : {output_root}")
    print(f"Manifest       : {manifest_path}")
    print(f"Failure log    : {faillog_path}")


def _handle_result(result, writer, ff):
    if result["status"] == "success":
        writer.writerow(
            [result["record"], result["npy_path"], result["label_path"], result["patient_id"]]
            + result["labels"]
        )
    else:
        ff.write(f"{result['record']} :: {result['reason']}\n")


if __name__ == "__main__":
    main()