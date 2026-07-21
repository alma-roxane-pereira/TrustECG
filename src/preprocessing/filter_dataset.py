"""
Creates a new dataset containing only ECGs
having at least one of the selected 9 diagnoses.
"""

from pathlib import Path
import shutil
import wfdb

# -------------------------
# Original Dataset
# -------------------------

SOURCE_ROOT = Path("data/raw/WFDBRecords")

# -------------------------
# New Dataset
# -------------------------

DEST_ROOT = Path("data/raw/WFDBRecords_9Class")

# -------------------------
# Target SNOMED Codes
# -------------------------

TARGET_CODES = {
    "426177001",   # SB
    "164934002",   # TWC
    "426783006",   # SR
    "164889003",   # AFIB
    "427084000",   # ST
    "55827005",    # 55827005
    "428750005",   # STTC
    "426761007",   # SVT
    "164890007"    # AF
}

DEST_ROOT.mkdir(parents=True, exist_ok=True)

kept = 0
discarded = 0

records = list(SOURCE_ROOT.rglob("*.hea"))

print(f"Found {len(records)} ECG records")

for hea_file in records:

    record_path = hea_file.with_suffix("")

    try:
        header = wfdb.rdheader(str(record_path))
    except:
        continue

    diagnosis = ""

    for c in header.comments:
        if c.startswith("Dx:"):
            diagnosis = c.replace("Dx:", "").strip()

    codes = set(code.strip() for code in diagnosis.split(","))

    if len(codes & TARGET_CODES):

        relative = hea_file.relative_to(SOURCE_ROOT)

        dest_folder = DEST_ROOT / relative.parent
        dest_folder.mkdir(parents=True, exist_ok=True)

        stem = hea_file.stem

        for ext in [".hea", ".mat"]:

            src = hea_file.with_suffix(ext)
            dst = dest_folder / (stem + ext)

            if src.exists():
                shutil.copy2(src, dst)

        kept += 1

    else:
        discarded += 1

print("="*60)
print("Finished")
print("="*60)
print("Kept ECGs      :", kept)
print("Discarded ECGs :", discarded)