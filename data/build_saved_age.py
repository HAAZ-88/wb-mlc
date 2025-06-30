import pandas as pd
from pathlib import Path

# ---------------------- CONFIGURACIÓN ----------------------
COUNTRY_TAG = "ecu"  # Cambia por 'ecu', 'per', etc., si deseas

BASE_DIR     = Path("data")
BASE_FILE    = f"micro_{COUNTRY_TAG}.csv"
OUTPUT_FILE  = f"micro_{COUNTRY_TAG}_saved_age.csv"

# Columnas necesarias
COLUMNS = [
    "saved",      # variable objetivo
    "age",        # para construir age_bin
    "educ",
    "emp_in",
    "mobileowner",
    "internetaccess",
    "anydigpayment"
]
# -----------------------------------------------------------

def build_dataset():
    df = pd.read_csv(BASE_DIR / BASE_FILE)

    # Validar columnas
    missing_cols = [col for col in COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Faltan columnas: {missing_cols}")

    # Eliminar observaciones con datos faltantes
    df = df[COLUMNS].dropna().copy()

    # Crear atributo protegido: 1 si edad < 30, si no 0
    df["age_bin"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

    # Crear DataFrame final
    final_cols = ["saved", "age_bin", "educ", "emp_in", "mobileowner", "internetaccess", "anydigpayment"]
    df_final = df[final_cols]

    # Guardar archivo limpio
    out_path = BASE_DIR / OUTPUT_FILE
    df_final.to_csv(out_path, index=False)
    print(f"✅ Archivo creado: {out_path} — {len(df_final):,} filas limpias")

if __name__ == "__main__":
    build_dataset()