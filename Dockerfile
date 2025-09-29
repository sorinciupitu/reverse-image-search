# Dockerfile pentru Reverse Image Search App
FROM python:3.11-slim

# Setează variabilele de mediu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Creează directorul de lucru
WORKDIR /app

# Instalează dependențele de sistem necesare
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiază fișierele de dependențe
COPY requirements.txt packages.txt ./

# Instalează dependențele Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalează dependențele de sistem din packages.txt (dacă există)
RUN if [ -f packages.txt ]; then \
    apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# Copiază codul aplicației
COPY . .

# Creează directoarele necesare pentru date și modele
RUN mkdir -p /app/data /app/models /app/chroma_db

# Setează permisiunile corecte
RUN chmod -R 755 /app

# Expune portul pentru Streamlit
EXPOSE 8501

# Health check pentru monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comandă pentru pornirea aplicației
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]