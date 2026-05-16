
# Configurar Virtual Environment en Python

## Pasos de instalación

### 1. Crear el virtual environment
```bash
python -m venv .venv
```

### 2. Activar el virtual environment

**En macOS/Linux:**
```bash
source .venv/bin/activate
```

**En Windows:**
```bash
.venv\Scripts\activate
```

### 3. Verificar la activación
```bash
which python  # macOS/Linux
where python  # Windows
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Desactivar el virtual environment
```bash
deactivate
```
