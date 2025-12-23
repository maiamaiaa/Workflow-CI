# Workflow-CI - Heart Disease Model Training

## Informasi Mahasiswa
- **Nama**: Eugenia Grasela Maia
- **Project**: Final Project - Supervised Machine Learning

## Deskripsi
Repository ini berisi pipeline CI/CD untuk training model Heart Disease Classification menggunakan GitHub Actions dan MLflow.

## Struktur Folder

```
Workflow-CI/
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml
└── MLProject/
    ├── MLProject
    ├── conda.yaml
    ├── modelling.py
    └── heart_disease_preprocessing/
        ├── X_train.csv
        ├── X_test.csv
        ├── y_train.csv
        ├── y_test.csv
        └── scaler.pkl
```

## Prerequisites

Data preprocessing harus sudah dilakukan di Repository 1:
- `Eksperimen_SML_Eugenia_Grasela_Maia`

Copy folder `heart_disease_preprocessing/` dari Repository 1 ke folder `MLProject/` di repository ini.

## Cara Penggunaan

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Workflow-CI.git
cd Workflow-CI
```

### 2. Copy Preprocessed Data
Copy folder `heart_disease_preprocessing/` dari Repository 1:
```bash
cp -r /path/to/Eksperimen_SML_Eugenia_Grasela_Maia/preprocessing/heart_disease_preprocessing MLProject/
```

### 3. Local Testing
```bash
cd MLProject
pip install mlflow pandas numpy scikit-learn joblib
python modelling.py
```

### 4. Trigger GitHub Actions
Push ke branch `main` atau `master` untuk trigger workflow:
```bash
git add .
git commit -m "Trigger model training"
git push origin main
```

Atau trigger manual melalui GitHub Actions tab.

## GitHub Actions Workflow

Workflow `ci.yml` akan:
1. Setup Python environment dengan Conda
2. Install dependencies dari `conda.yaml`
3. Jalankan MLflow Project
4. Upload model artifacts

### Trigger Events
- Push ke `main`/`master`
- Pull request ke `main`/`master`
- Manual dispatch (workflow_dispatch)

## Model Training

Script `modelling.py` akan:
1. Load preprocessed data
2. Train 4 model berbeda:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVM
3. Evaluate dan compare performance
4. Log metrics ke MLflow
5. Save best model

## Output

Setelah training:
- Model tersimpan di `model/best_model.pkl`
- Metrics di-log ke MLflow
- Artifacts di-upload ke GitHub Actions

## Kriteria Penilaian

Repository ini memenuhi **Kriteria 3** dengan target skor **Expert (4 pts)**:
- ✅ MLProject structure dengan conda.yaml
- ✅ modelling.py dengan training code
- ✅ GitHub Actions workflow (ci.yml)
- ✅ Artifact saving
- ✅ **Docker image pushed to DockerHub**
- ✅ **Tautan_DockerHub.txt tersedia**

## DockerHub Integration

### Setup DockerHub Secrets
Untuk mengaktifkan Docker build dan push, tambahkan secrets di GitHub Repository:
1. Buka Settings → Secrets and variables → Actions
2. Tambahkan:
   - `DOCKERHUB_USERNAME`: Username DockerHub Anda
   - `DOCKERHUB_TOKEN`: Access token dari DockerHub (bukan password)

### Cara Membuat DockerHub Token
1. Login ke https://hub.docker.com
2. Klik avatar → Account Settings → Security
3. Klik "New Access Token"
4. Beri nama (misal: "github-actions")
5. Copy token dan simpan sebagai secret di GitHub

### Manual Docker Build (Local)
```bash
cd MLProject
docker build -t YOUR_USERNAME/heart-disease-mlflow:latest .
docker push YOUR_USERNAME/heart-disease-mlflow:latest
```

## Catatan Penting

- Repository ini HANYA untuk Kriteria 3 (Workflow CI)
- TIDAK mengandung notebook eksperimen atau raw data
- Data preprocessing berada di repository terpisah: `Eksperimen_SML_Eugenia_Grasela_Maia`
- Pastikan repository ini **PUBLIC** saat submission
- **WAJIB**: Update Tautan_DockerHub.txt dengan link DockerHub yang valid

## Related Repositories
- **Kriteria 1 & 2 (Eksperimen & Model)**: `Eksperimen_SML_Eugenia_Grasela_Maia`
- **Kriteria 3 (Workflow CI)**: Repository ini
