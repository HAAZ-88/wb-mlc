import pandas as pd
from pathlib import Path

# ---------------------- CONFIGURACIÓN ----------------------
COUNTRY_TAG = "ecu"  # Cambia por 'bol', 'ecu', 'per', etc.

BASE_DIR     = Path("data")
BASE_FILE    = f"micro_{COUNTRY_TAG}.csv"
OUTPUT_FILE  = f"micro_{COUNTRY_TAG}_saved_educlow.csv"

# Columnas finales deseadas
COLUMNS = [
    "saved",        # variable objetivo
    "educ_low",     # se construye a partir de 'educ'
    "female",
    "age",
    "educ",
    "emp_in",
    "mobileowner"
]

EDUC_COL_RAW = "educ"  # nivel educativo crudo
# -----------------------------------------------------------

def build_dataset():
    df = pd.read_csv(BASE_DIR / BASE_FILE)

    # Crear columna educ_low
    df["educ_low"] = df[EDUC_COL_RAW].apply(lambda x: 1 if x <= 2 else 0)

    # Asegurar que todas las columnas estén disponibles
    missing_cols = [col for col in COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Faltan columnas en el archivo: {missing_cols}")

    df = df[COLUMNS]

    # Eliminar observaciones con datos faltantes
    df = df.dropna().copy()

    # Mapear valores categóricos si fueran texto (por si acaso)
    bin_cols = ["female", "emp_in", "mobileowner"]
    for col in bin_cols:
        if df[col].dtype == "object":
            df[col] = df[col].map({"yes": 1, "no": 0})

    # Guardar archivo final limpio
    out_path = BASE_DIR / OUTPUT_FILE
    df.to_csv(out_path, index=False)
    print(f"✅ Archivo creado: {out_path} — {len(df):,} filas limpias")

if __name__ == "__main__":
    build_dataset()