### Proyecto para Inteligencia Artificial para Médicos

## Fase de entrenamiento con HAM10000

# Prerrequisitos

* Python 3.12 y librerias de Python

```bash
python3 -m pip install -r requirements.txt

```

* Descargar y extraer carpetas HAM10000 en el directorio

El sistema de archivos debe verse así

```text
├── root/
│   ├── IAMEDICOS/
│   └── dataverse_files/
|       ├── HAM10000_images_part_1/
|       ├── HAM10000_images_part_2/
|       └── HAM10000_metadata
```

# Entrenar RESNET 50

```bash
python3 training_resnet.py
```

# Entrenar EfficientNet B4

```bash
python3 training_e_net.py
```

## Fase de pruebas con DDI

* Descargar y extraer carpetas DDI en el directorio

El sistema de archivos debe verse así

```text
├── root/
│   ├── IAMEDICOS/
│   ├── dataverse_files/
|   |   ├── HAM10000_images_part_1/
|   |   ├── HAM10000_images_part_2/
|   |   └── HAM10000_metadata
|   └── ddidiversedermatologyimages/  
```

# Preparación de los datos

Ejecutar el notebook format_images.ipynb en el ambiente de su preferencia

Se deben crear los directorios 

```text
├── root/
│   ├── IAMEDICOS/
│   ├── dataverse_files/
│   ├── 
│   ├── dataverse_files/
|   |   ├── HAM10000_images_part_1/
|   |   ├── HAM10000_images_part_2/
|   |   └── HAM10000_metadata
|   └── ddidiversedermatologyimages/  
```

# Problemas


