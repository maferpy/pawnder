# Pawnder

Plataforma para ayudar a encontrar mascota ideal y estimar el tiempo de adopción de perros y gatos.

## Funcionalidades

- Recomendador de mascotas según preferencias del usuario (tipo, edad, actividad, hogar, etc.).
- Predicción del tiempo de adopción usando señales tabulares (edad, raza, salud, costo, etc.), texto (descripción) e imagen (EfficientNet + PCA).
- Consejos generados con OpenAI para mejorar la publicación de adopción.
- Historia breve/emotiva para visualizar la vida con una mascota recomendada.

## Stack

- Python 3.12+
- Streamlit
- LightGBM
- TensorFlow / Keras (EfficientNetB0)
- Sentence Transformers
- OpenAI API
- `uv` para manejo de entorno y dependencias

## Requisitos

- Tener instalado `uv`: https://docs.astral.sh/uv/
- Python 3.12 (el proyecto usa `.python-version` y `pyproject.toml`)

## Instalación (con uv)

Desde la carpeta del proyecto (`pawnder/`):

```bash
uv sync
```

Esto crea/actualiza `.venv` e instala dependencias desde `pyproject.toml` y `uv.lock`.

## Configuración de API key (OpenAI)

La app busca `OPENAI_API_KEY` en `st.secrets`.

Opción recomendada:

1. Crea la carpeta `.streamlit/`
2. Crea el archivo `.streamlit/secrets.toml`
3. Agrega:

```toml
OPENAI_API_KEY = "tu_api_key_aqui"
```

Si no hay API key, la app sigue funcionando pero sin generación de historias/consejos.

## Ejecutar la app

Entry point principal:

```bash
uv run streamlit run main_principal.py
```

Luego abre la URL local que muestra Streamlit (normalmente `http://localhost:8501`).

## Estructura (resumen)

```text
pawnder/
├── main_principal.py      # App principal con navegación
├── main.py                # Flujo "Poner en adopción"
├── main2.py               # Flujo de recomendaciones
├── utils/
│   └── inference.py       # Predicción + tips OpenAI
├── pipelines/
│   ├── tabular.py
│   ├── text.py
│   └── image.py
├── models/                # Modelos entrenados (.pkl/.npy)
├── datasets/              # Datos para recomendaciones
├── embeddings/            # Embeddings precomputados
└── img/                   # Assets de interfaz
```

## Notas

- Este repositorio incluye artefactos de modelo y datos necesarios para inferencia local.
- Si quieres correr solo la parte de UI sin OpenAI, omite la API key.

## Licencia

Este proyecto incluye archivo `LICENSE` (MIT).
