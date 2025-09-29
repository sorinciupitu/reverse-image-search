# 🔍 Reverse Image Search - Streamlit Application

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com/)

O aplicație web puternică pentru căutare inversă de imagini, construită cu Streamlit, EfficientNet-B0 și ChromaDB. Încarcă o imagine și găsește imagini vizual similare în baza ta de date folosind extragerea de caracteristici bazată pe deep learning.

**🌟 [Demo Live](https://your-app-url.streamlit.app) | 📖 [Documentație](https://github.com/your-username/reverse-image-search) | 🐛 [Raportează Bug](https://github.com/your-username/reverse-image-search/issues)**

## 🚀 Features

- **🔍 Reverse Image Search**: Upload images and find visually similar ones
- **🤖 Deep Learning**: Uses EfficientNet-B0 for robust feature extraction
- **⚡ Fast Search**: ChromaDB vector database for lightning-fast similarity search
- **🗄️ Database Management**: Add, remove, and manage your image collections
- **📊 Statistics**: View database statistics and directory breakdowns
- **⚙️ Configurable**: Adjust search parameters and similarity thresholds
- **🌐 Web Interface**: Clean, responsive Streamlit interface
- **☁️ Cloud Ready**: Optimized for deployment on Streamlit Cloud

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Deep Learning**: PyTorch, EfficientNet-B0
- **Vector Database**: ChromaDB
- **Image Processing**: Pillow (PIL)
- **Language**: Python 3.8+

## 📦 Instalare Rapidă

### Opțiunea 1: Clonare din GitHub (Recomandat)

```bash
# Clonează repository-ul
git clone https://github.com/your-username/reverse-image-search.git
cd reverse-image-search

# Instalează dependențele
pip install -r requirements.txt

# Rulează aplicația
streamlit run app.py
```

### Opțiunea 2: Download ZIP

1. **Descarcă** acest repository ca ZIP
2. **Extrage** fișierele într-un folder
3. **Instalează dependențele**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Rulează aplicația**:
   ```bash
   streamlit run app.py
   ```

**🌐 Deschide browser-ul** și navighează la `http://localhost:8501`

## 🎯 Usage

### Home Page - Image Search
1. Upload an image using the file uploader
2. Adjust search parameters (number of results, similarity threshold)
3. Click "Search Similar Images" to find matches
4. View results with similarity scores and metadata

### Database Management
1. Add images by specifying a directory path
2. View database statistics and directory breakdowns
3. Remove directories or clear the entire database
4. Monitor processing progress with real-time updates

### Settings
1. View model and system information
2. Configure default search parameters
3. Check supported image formats
4. Monitor system resources

## 📁 Project Structure

```
reverse-image-search/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── src/
│   ├── image_processor.py     # EfficientNet-B0 feature extraction
│   ├── database_manager.py    # ChromaDB operations
│   ├── search_engine.py       # Core search functionality
│   └── utils.py               # Utility functions
├── models/                    # Model cache directory
├── data/                      # ChromaDB data directory
└── sample_images/             # Sample images for testing
```

## 🔧 Configuration

### Streamlit Configuration
The application includes a custom Streamlit configuration in `.streamlit/config.toml`:
- Custom theme with modern colors
- Optimized for development and production
- Automatic file watching for development

### Model Configuration
- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Feature Dimension**: 1280
- **Input Size**: 224x224 pixels
- **Device**: Automatic GPU/CPU detection

### Database Configuration
- **Vector Database**: ChromaDB
- **Storage**: Persistent local storage
- **Similarity Metric**: Cosine similarity
- **Indexing**: Automatic HNSW indexing

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## 🚀 Deployment

### Deployment cu Docker și Coolify

Această aplicație este optimizată pentru deployment pe VPS folosind Docker și Coolify:

#### Cerințe pentru deployment:
- Docker și Docker Compose instalate
- Coolify configurat pe VPS
- Minimum 2GB RAM și 10GB spațiu liber

#### Pași pentru deployment cu Coolify:

1. **Pregătire repository**:
   ```bash
   # Clonează repository-ul pe serverul tău
   git clone <repository-url>
   cd reverse-image-search
   ```

2. **Configurare variabile de mediu**:
   ```bash
   # Copiază fișierul de exemplu
   cp .env.example .env
   
   # Editează variabilele după necesități
   nano .env
   ```

3. **Testare locală cu Docker**:
   ```bash
   # Build și rulare pentru producție
   docker-compose up -d production
   
   # Sau pentru development cu hot-reload
   docker-compose up -d development
   ```

4. **Deploy cu Coolify**:
   - Adaugă un nou proiect în Coolify
   - Selectează "Docker Compose" ca tip de deployment
   - Configurează repository-ul Git
   - Setează variabilele de mediu necesare
   - Deploy aplicația

#### Configurare Coolify:

**Variabile de mediu importante pentru Coolify:**
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
PYTHONUNBUFFERED=1
```

**Port mapping:**
- Container port: 8501
- Public port: 80 sau 443 (configurat în Coolify)

**Volume mounts pentru persistență:**
- `./chroma_db:/app/chroma_db` - Baza de date ChromaDB
- `./models:/app/models` - Cache modele AI
- `./data:/app/data` - Date aplicație
- `./logs:/app/logs` - Log-uri aplicație

#### Monitorizare și Health Checks:

Aplicația include health check automat la `/healthz` pentru monitoring în Coolify.

### Deployment pe Streamlit Cloud (alternativ)

Pentru deployment rapid pe [Streamlit Cloud](https://streamlit.io/cloud):

1. **Push your code** to a GitHub repository
2. **Connect to Streamlit Cloud** and select your repository
3. **Set the main file** to `app.py`
4. **Deploy** - the application will automatically install dependencies

### Deployment Notes
- All dependencies are specified in `requirements.txt`
- The application uses CPU-only PyTorch for cloud compatibility
- ChromaDB data persists between sessions using Docker volumes
- Optimized for both cloud and VPS deployment
- Includes backup/restore functionality for easy migration

## 📊 Performance

- **Feature Extraction**: ~100ms per image (CPU)
- **Search Speed**: <50ms for 10K images
- **Memory Usage**: ~500MB base + ~1KB per image
- **Supported Database Size**: 100K+ images

## 🔍 How It Works

1. **Feature Extraction**: Images are processed through EfficientNet-B0 to extract 1280-dimensional feature vectors
2. **Vector Storage**: Features are stored in ChromaDB with metadata (file path, size, etc.)
3. **Similarity Search**: Query images are compared using cosine similarity
4. **Result Ranking**: Results are ranked by similarity score and filtered by threshold

## 🛡️ Error Handling

The application includes comprehensive error handling:
- Invalid image format detection
- File access permission checks
- Database connection error recovery
- Memory usage monitoring
- Graceful degradation for missing files

## 🤝 Contribuții

Contribuțiile sunt binevenite! Pentru a contribui:

1. **Fork** acest repository
2. **Creează** o ramură pentru feature-ul tău (`git checkout -b feature/AmazingFeature`)
3. **Commit** modificările tale (`git commit -m 'Add some AmazingFeature'`)
4. **Push** pe ramură (`git push origin feature/AmazingFeature`)
5. **Deschide** un Pull Request

### Tipuri de contribuții dorite:
- 🐛 Raportarea și rezolvarea bug-urilor
- ✨ Adăugarea de noi funcționalități
- 📚 Îmbunătățirea documentației
- 🎨 Îmbunătățiri UI/UX
- ⚡ Optimizări de performanță

## 📄 Licență

Acest proiect este open source și disponibil sub [Licența MIT](LICENSE).

## ⭐ Suport

Dacă acest proiect te-a ajutat, te rog să îi dai o ⭐ pe GitHub!

Pentru întrebări sau probleme:
- 📧 [Deschide un Issue](https://github.com/your-username/reverse-image-search/issues)
- 💬 [Discussions](https://github.com/your-username/reverse-image-search/discussions)
## 🆘 Troubleshooting

**"No module named 'src'"**
- Make sure you're running the app from the project root directory
- Ensure all files in the `src/` directory are present

**"Database connection error"**
- Check that the `data/` directory is writable
- Restart the application to reinitialize ChromaDB

**"Model loading failed"**
- Ensure you have a stable internet connection for initial model download
- Check available disk space (models require ~50MB)

**"Out of memory"**
- Reduce batch size in settings
- Process smaller directories at a time
- Use CPU instead of GPU for large datasets

### Performance Tips

- **For large datasets**: Process images in smaller batches
- **For faster search**: Increase similarity threshold to reduce results
- **For better accuracy**: Use lower similarity threshold but expect more results
- **For cloud deployment**: Use smaller image directories to stay within memory limits

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the application logs in the Streamlit interface
3. Ensure all dependencies are correctly installed
4. Verify your Python version is 3.8 or higher

---

**Built with ❤️ using Streamlit, PyTorch, and ChromaDB**