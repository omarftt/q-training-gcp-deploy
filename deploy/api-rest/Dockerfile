# Usamos una imagen base de Python oficial. Puedes elegir una versión específica de Python si lo necesitas.
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos el archivo requirements.txt al contenedor para instalar las dependencias de Python.
COPY requirements.txt .

# Instalamos las dependencias de Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código fuente al contenedor.
COPY app/ .

CMD gunicorn app:app --workers 1 --threads 8 --worker-class uvicorn.workers.UvicornWorker --bind :8080 --timeout 0