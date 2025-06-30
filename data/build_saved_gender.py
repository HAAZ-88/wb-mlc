import pandas as pd
from pathlib import Path

# ---------------------- CONFIGURACIÓN ----------------------
COUNTRY_TAG = "ecu"  # Cambia por 'bol', 'ecu', 'per', etc.

BASE_DIR     = Path("data")
BASE_FILE    = f"micro_{COUNTRY_TAG}.csv"
OUTPUT_FILE  = f"micro_{COUNTRY_TAG}_saved_gender.csv"

# Columnas necesarias
COLUMNS_RAW = [
    "saved",
    "female",
    "age",
    "emp_in",
    "mobileowner",
    "internetaccess",
    "anydigpayment"
]
# -----------------------------------------------------------

def build_dataset():
    df = pd.read_csv(BASE_DIR / BASE_FILE)

    # Re-codificar female: 1 = hombre → 0, 2 = mujer → 1
    df["female_bin"] = df["female"].map({1: 0, 2: 1})

    # Verificar columnas requeridas
    required_cols = ["saved", "female_bin", "age", "emp_in", "mobileowner", "internetaccess", "anydigpayment"]
    missing = [col for col in required_cols if col not in df.columns and col not in COLUMNS_RAW]
    if missing:
        raise ValueError(f"❌ Faltan columnas: {missing}")

    # Subconjunto limpio
    df = df.dropna(subset=COLUMNS_RAW).copy()
    df["female_bin"] = df["female_bin"].astype(int)

    df_final = df[required_cols]

    # Guardar archivo limpio
    out_path = BASE_DIR / OUTPUT_FILE
    df_final.to_csv(out_path, index=False)
    print(f"✅ Archivo creado: {out_path} — {len(df_final):,} filas limpias")

if __name__ == "__main__":
    build_dataset()
