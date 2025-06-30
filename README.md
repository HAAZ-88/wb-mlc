
# Welfare-Based Machine Learning Classifier (WB-MLC)

Este repositorio contiene una arquitectura modular para la evaluación y selección de clasificadores supervisados en contextos sensibles, incorporando criterios normativos explícitos de equidad y bienestar.

## Estructura del modelo
WB-MLC combina tres componentes:
1. Restricción mínima de equidad basada en la métrica $\text{TPR-gap}^2$.
2. Selección de modelos en la frontera de Pareto (Precisión vs Bienestar).
3. Función de bienestar prioritarista ajustable mediante $\lambda_{\text{norm}}$.

## Contenido

- `scripts/run_experiments.py`: Entrenamiento y evaluación del clasificador para distintas semillas y valores de $\lambda$.
- `scripts/select_pareto.py`: Aplicación del criterio normativo de selección en frontera.
- `src/`: Implementación del modelo, métricas y funciones de pérdida.
- `results/`: Archivos generados con resultados, tablas y figuras.
- `data/`: Archivos de entrada (preprocesados).

## Requisitos

```bash
pip install -r requirements.txt
