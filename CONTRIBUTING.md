# 🤝 Ghid de Contribuție

Mulțumim pentru interesul tău de a contribui la proiectul Reverse Image Search! Acest ghid te va ajuta să începi.

## 🚀 Cum să contribui

### 1. Fork și Clone

```bash
# Fork repository-ul pe GitHub, apoi clonează-l local
git clone https://github.com/your-username/reverse-image-search.git
cd reverse-image-search

# Adaugă upstream remote
git remote add upstream https://github.com/original-owner/reverse-image-search.git
```

### 2. Configurare mediu de dezvoltare

```bash
# Creează un virtual environment
python -m venv venv
source venv/bin/activate  # Pe Windows: venv\Scripts\activate

# Instalează dependențele
pip install -r requirements.txt

# Rulează aplicația pentru testare
streamlit run app.py
```

### 3. Creează o ramură pentru feature

```bash
# Sincronizează cu upstream
git fetch upstream
git checkout main
git merge upstream/main

# Creează o ramură nouă
git checkout -b feature/numele-feature-ului
```

### 4. Fă modificările

- Respectă stilul de cod existent
- Adaugă comentarii pentru logica complexă
- Testează modificările tale local
- Asigură-te că aplicația rulează fără erori

### 5. Commit și Push

```bash
# Adaugă modificările
git add .

# Commit cu un mesaj descriptiv
git commit -m "feat: adaugă funcționalitatea X"

# Push pe ramura ta
git push origin feature/numele-feature-ului
```

### 6. Creează Pull Request

1. Mergi pe GitHub la repository-ul tău fork
2. Apasă "New Pull Request"
3. Completează template-ul de PR
4. Așteaptă review-ul

## 📝 Tipuri de contribuții

### 🐛 Bug Reports
- Folosește template-ul de issue pentru bug-uri
- Includeți pași de reproducere
- Adăugați screenshot-uri dacă e relevant
- Specificați versiunea Python și OS

### ✨ Feature Requests
- Descrieți feature-ul dorit
- Explicați de ce ar fi util
- Propuneți o implementare dacă aveți idei

### 📚 Documentație
- Îmbunătățiri la README
- Comentarii în cod
- Exemple de utilizare
- Ghiduri de deployment

### 🎨 UI/UX
- Îmbunătățiri la interfață
- Responsive design
- Accesibilitate
- Experiența utilizatorului

## 🔧 Standarde de cod

### Python
- Folosește PEP 8 pentru formatare
- Nume descriptive pentru variabile și funcții
- Docstrings pentru funcții complexe
- Type hints unde e posibil

### Streamlit
- Componentele să fie modulare
- Folosește session state pentru persistență
- Gestionează erorile elegant
- Optimizează pentru performanță

### Git
- Commit-uri atomice și descriptive
- Folosește conventional commits:
  - `feat:` pentru funcționalități noi
  - `fix:` pentru bug-uri
  - `docs:` pentru documentație
  - `style:` pentru formatare
  - `refactor:` pentru refactoring
  - `test:` pentru teste

## 🧪 Testare

Înainte de a face PR, testează:

1. **Funcționalitatea de bază**:
   - Upload de imagini
   - Căutare similară
   - Management bază de date

2. **Edge cases**:
   - Imagini corupte
   - Formate nesuportate
   - Directoare inexistente

3. **Performance**:
   - Imagini mari
   - Multe rezultate
   - Bază de date mare

## 📋 Checklist PR

Înainte de a submite PR-ul, verifică:

- [ ] Codul rulează fără erori
- [ ] Funcționalitatea nouă e testată
- [ ] Documentația e actualizată
- [ ] Commit-urile sunt clean
- [ ] Nu există conflicte cu main
- [ ] PR-ul are o descriere clară

## 🆘 Ajutor

Dacă ai întrebări:

- 💬 [GitHub Discussions](https://github.com/your-username/reverse-image-search/discussions)
- 📧 [Deschide un Issue](https://github.com/your-username/reverse-image-search/issues)
- 📖 Consultă documentația existentă

## 🎉 Recunoaștere

Toți contribuitorii vor fi adăugați în secțiunea Contributors din README.

Mulțumim pentru contribuția ta! 🙏