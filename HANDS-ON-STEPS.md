# Project 1 — Hands-On Steps: Local ML Pipeline (Train → Serve)

> A step-by-step build log for the Heart Disease MLOps pipeline — cleaned up and
> **corrected to what actually works** on a modern stack (MLflow 3.x, Pydantic v2,
> Docker via Codespaces). Each task lists its **key topics**, the **exact working
> steps**, the **gotchas I hit and how I fixed them**, and the **mandatory learnings**
> to lock in before moving on.
>
> Companion to [`README.md`](README.md): the README is the *what & why*; this is the *how*.

**Author:** Srinivas Charan Mamidi · **Goal:** end-to-end ML pipeline, train → serve, as an MLOps interview portfolio piece.

---

## Environment & Ground Rules

| | |
|---|---|
| **Machine** | Windows laptop, 8 GB RAM (Docker step offloaded to GitHub Codespaces) |
| **Python** | 3.14 local · `python:3.11-slim` in the container |
| **Key libs** | pandas · scikit-learn 1.8 · mlflow 3.12 · fastapi 0.136 · uvicorn 0.47 · pydantic 2.13 |
| **Dataset** | [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) — file `processed.cleveland.data` |
| **Repo** | `mlops-heart-disease-pipeline` — run all commands **from the repo root** |

**Working rules I followed**
1. Each task is ~90 minutes. If it overruns, continue the *same* task next day — don't skip ahead.
2. Every task ends in something you can **see**: a file, an output, a running server, a screenshot.
3. When stuck, paste the error into Claude and understand the fix — that's how engineers work.
4. **One git commit per task**, minimum.

> **⚠️ Read this first — the #1 gotcha across the whole project:** pip installed packages
> as a *per-user* install, so tool CLIs (like `mlflow.exe`) are **not on PATH**. Always
> invoke them through Python: **`python -m mlflow …`** instead of bare `mlflow …`.

---

## Task 1 — Environment Setup

**Goal:** Python + tooling installed, project folder scaffolded, dependencies verified.

**Key topics:** Python install & PATH · `pip` · virtual/per-user installs · project structure · `git init`.

### Steps
1. Install **Python** from [python.org](https://www.python.org/) with **"Add to PATH"** ticked. Verify:
   ```bash
   python --version
   ```
2. Install **VS Code** + the Microsoft **Python** extension.
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn mlflow fastapi uvicorn jupyter
   ```
4. Scaffold the project:
   ```bash
   mkdir mlops-heart-disease-pipeline
   cd mlops-heart-disease-pipeline
   mkdir data models notebooks scripts screenshots
   git init
   ```
5. Create `requirements.txt`:
   ```
   pandas
   scikit-learn
   mlflow
   fastapi
   uvicorn
   jupyter
   ```
6. Create `test_setup.py` and run it:
   ```python
   import pandas as pd
   import sklearn
   import mlflow
   import fastapi
   print("pandas version:", pd.__version__)
   print("sklearn version:", sklearn.__version__)
   print("mlflow version:", mlflow.__version__)
   print("All good. Environment ready.")
   ```
   ```bash
   python test_setup.py
   ```
7. Commit:
   ```bash
   git add .
   git commit -m "Task 1: Environment setup complete"
   ```

**✅ Done when:** `test_setup.py` prints all versions with zero errors.

**Before you move on, you should know:**
- Why "Add to PATH" matters, and why `python -m <tool>` is the reliable way to run a library's CLI.
- What `requirements.txt` is for (reproducible installs).

---

## Task 2 — Data Loading & Exploration

**Goal:** Load the raw dataset, understand it, and save a clean version.

**Key topics:** pandas I/O · missing values (`na_values`, `dropna`) · target encoding · basic EDA.

### Steps
1. Download the dataset from the [UCI page](https://archive.ics.uci.edu/dataset/45/heart+disease). It arrives as a **zip** (`heart+disease.zip`) containing several files — **extract it and copy the single file `processed.cleveland.data` into `data/`**, so it lands as a *file* at `data/processed.cleveland.data` (not a folder — see the gotcha below).
2. Create `scripts/explore_data.py`:
   ```python
   import pandas as pd

   # Column names for this dataset (the file has no header row)
   columns = [
       'age', 'sex', 'cp', 'trestbps', 'chol',
       'fbs', 'restecg', 'thalach', 'exang',
       'oldpeak', 'slope', 'ca', 'thal', 'target'
   ]

   # Load data ('?' marks missing values in this dataset)
   df = pd.read_csv('data/processed.cleveland.data',
                    names=columns, na_values='?')

   print("Shape:", df.shape)
   print("\nFirst 5 rows:\n", df.head())
   print("\nMissing values:\n", df.isnull().sum())
   print("\nBasic stats:\n", df.describe())
   print("\nTarget distribution:\n", df['target'].value_counts())

   # Convert target to binary: 0 = no disease, 1 = disease
   df['target'] = (df['target'] > 0).astype(int)

   # Drop rows with missing values (only 6 rows — safe to drop)
   df = df.dropna()

   # Save clean version
   df.to_csv('data/heart_clean.csv', index=False)
   print("\nClean data saved to data/heart_clean.csv")
   print("Clean data shape:", df.shape)
   ```
3. Run it **from the repo root**:
   ```bash
   python scripts/explore_data.py
   ```

> **⚠️ Gotcha I hit — `PermissionError: [Errno 13]` on the CSV.** The dataset zip had
> extracted into a **folder** *named* `processed.cleveland.data`, so `open()` was
> pointed at a directory, not a file (Windows reports that as a PermissionError;
> Linux would say `IsADirectoryError`). **Fix:** move/rename so the actual data file
> sits at `data/processed.cleveland.data`. Lesson: a "permission" error on a path that
> exists often means it's a directory.

**📓 Concept note — Data cleaning in pandas** (prevents "Garbage In, Garbage Out"):
- **Missing data** → drop (`.dropna()`) or fill (`.fillna()` with a mean/median/logical value).
- **Duplicates** → `.drop_duplicates()`.
- **Type casting** → `.astype()`, `pd.to_datetime()` so numbers/dates aren't stored as strings.
- **Normalize text** → `.str.lower()`, `.str.strip()`, `.replace()` to unify categories.
- **Outliers** → boolean masking, e.g. `df[df['chol'] < 600]`.

**✅ Done when:** `data/heart_clean.csv` exists (303 rows → **297** after dropping 6 NA rows).

**Before you move on, you should know:**
- The dataset shape and what the `target` column means (0 = no disease, 1 = disease).
- Which columns had missing values (`ca`, `thal`) and why dropping 6 rows is acceptable here.

---

## Task 3 — Train Your First Model (no MLflow yet)

**Goal:** Train a Random Forest, print accuracy, save the model to disk.

**Key topics:** train/test split · `RandomForestClassifier` · accuracy vs. precision/recall/F1 · pickling.

### Steps
1. Create `scripts/train.py`:
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, classification_report
   import pickle

   df = pd.read_csv('data/heart_clean.csv')
   X = df.drop('target', axis=1)
   y = df['target']

   # 80% train / 20% test
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"\nModel Accuracy: {accuracy:.4f}")
   print("\n", classification_report(y_test, y_pred,
         target_names=['No Disease', 'Disease']))

   with open('models/heart_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   print("\nModel saved to models/heart_model.pkl")
   ```
2. Run it:
   ```bash
   python scripts/train.py
   ```
   **My result → `Model Accuracy: 0.8833`** (note this; you'll compare it in Task 4).
3. Commit:
   ```bash
   git add .
   git commit -m "Task 3: First model trained, accuracy noted"
   ```

**📓 Concept note — scikit-learn** is the standard Python ML library. Pandas *shapes* the
data; sklearn provides the *algorithms*. It covers **Classification** (predict a category),
**Regression** (predict a number), **Clustering** (group similar items), and **Preprocessing**
(scaling/transforming for models). `RandomForestClassifier` = an ensemble of decision trees
that vote — robust and a strong default. (Good primer: <https://www.youtube.com/watch?v=cIbj0WuK41w>)

**✅ Done when:** `models/heart_model.pkl` exists and accuracy is printed.

**🎯 Mandatory learnings — answer these before Task 4:**
1. What does **pandas** do in this project (vs. what sklearn does)?
2. What is the **shape** of `heart_clean.csv`? *(297 × 14)*
3. What does **`train_test_split`** do, and why is **`random_state`** important? *(reproducibility)*
4. What is **accuracy**, and what was yours? *(0.8833)*
5. What's the difference between **precision** and **recall**? *(precision = of predicted-positive, how many were right; recall = of actual-positive, how many we caught)*
6. What is **`heart_model.pkl`** and why does it exist? *(the serialized trained model, so you can reuse it without retraining)*
7. What does **`n_estimators=100`** mean in Random Forest? *(number of trees that vote)*

---

## Task 4 — MLflow Experiment Tracking

**Goal:** Log every training run (params + metrics + model) so runs are comparable; view them in the MLflow UI.

**Key topics:** experiments & runs · logging params/metrics/models · **MLflow backends** (file store vs. database).

> **⚠️ Two corrections to the naive tutorial — both essential:**
> 1. **Use a SQLite backend**, not the default `./mlruns` file store. The file store is
>    deprecated in MLflow 3.x and (critically) **cannot host the Model Registry** you need
>    in Task 5. Add `mlflow.set_tracking_uri("sqlite:///mlflow.db")` to the script.
> 2. **Don't drop the `pickle.dump`** from Task 3 when you add MLflow. The container in
>    Task 7 loads that pkl — losing it silently serves a stale model (**train/serve drift**).

### Steps
1. Update `scripts/train.py` to the tracked version (keeps the pkl dump):
   ```python
   import os
   import pickle
   import pandas as pd
   import mlflow
   import mlflow.sklearn
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, f1_score

   # Use a SQLite backend: the default ./mlruns file store is deprecated in
   # MLflow 3.x and can't host the Model Registry that Task 5 needs.
   mlflow.set_tracking_uri("sqlite:///mlflow.db")
   mlflow.set_experiment("heart-disease-prediction")

   # ── Change these between runs to compare ──
   N_ESTIMATORS = 100      # run 2: 200
   MAX_DEPTH = None        # run 2: 10
   TEST_SIZE = 0.2

   df = pd.read_csv('data/heart_clean.csv')
   X = df.drop('target', axis=1)
   y = df['target']
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=TEST_SIZE, random_state=42
   )

   with mlflow.start_run():
       mlflow.log_param("n_estimators", N_ESTIMATORS)
       mlflow.log_param("max_depth", MAX_DEPTH)
       mlflow.log_param("test_size", TEST_SIZE)

       model = RandomForestClassifier(
           n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42
       )
       model.fit(X_train, y_train)

       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred)
       mlflow.log_metric("accuracy", accuracy)
       mlflow.log_metric("f1_score", f1)

       # Log to the MLflow registry...
       mlflow.sklearn.log_model(model, "model")

       # ...AND keep a plain pickle for the Docker image (Task 7).
       os.makedirs('models', exist_ok=True)
       with open('models/heart_model.pkl', 'wb') as f:
           pickle.dump(model, f)

       print(f"Run complete. Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
       print(f"Run ID: {mlflow.active_run().info.run_id}")
   ```
2. Run it once (100 / None), then change to **`N_ESTIMATORS=200`, `MAX_DEPTH=10`** and run again:
   ```bash
   python scripts/train.py      # run 1
   # edit the two params, then:
   python scripts/train.py      # run 2
   ```
   *(A harmless `artifact_path is deprecated` warning may print when the model logs — ignore it; the run still records correctly.)*
3. Launch the MLflow UI **in a second terminal** (note the corrected command):
   ```bash
   python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
   # open http://127.0.0.1:5000
   ```
   > `python -m mlflow` (PATH fix) **and** `--backend-store-uri sqlite:///mlflow.db` — without
   > the flag the UI reads the empty default store and shows **nothing**.
4. Screenshot the two runs → `screenshots/task4_mlflow.png`.
5. Commit: `git add . && git commit -m "Task 4: MLflow tracking added, 2 runs logged"`

**✅ Done when:** you can see **2 runs** with their params and metrics in the MLflow UI.

**Before you move on, you should know:**
- The difference between a **file store** and a **database backend**, and why the registry needs a DB.
- What MLflow **logs** (params, metrics, artifacts/model) and why run comparison matters.

---

## Task 5 — Register the Best Model & Load It (Registry + Aliases)

**Goal:** Register your best run in the Model Registry, mark it as the model to serve, and load it back by a stable reference.

**Key topics:** Model Registry · model **versions** · **aliases** (the modern replacement for stages).

> **⚠️ Biggest correction in the whole project — stages are gone.** The tutorial says
> "change Stage to Production" and load `models:/HeartDiseaseModel/Production`. **MLflow 3.x
> removed stages.** Use an **alias** instead.

**📓 Interview-ready note — Stages → Aliases:** Old MLflow promoted models through *stages*
(None → Staging → Production → Archived). MLflow 3.x **removed stages** and replaced them with
**aliases** — named, movable pointers you attach to a version (`champion`, `challenger`, …).
Same concept as "Production" (a label for *the model to serve*), just more flexible.
> Say this in an interview: *"I promote models using registry aliases, since stages were deprecated in MLflow 3."*

### Steps
1. In the MLflow UI (`python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`), open your
   best run → **Register Model** → name it **`HeartDiseaseModel`**.
2. Set the **`champion` alias** on that version (this replaces the old "set stage to Production").
   **Easiest in the UI:** Models → **HeartDiseaseModel** → click **version 1** → **Aliases** → add `champion`.
   Or via code — save it as `set_alias.py` and run `python set_alias.py` (use **your** version number — it's `1` for a first registration):
   ```python
   import mlflow
   from mlflow import MlflowClient

   mlflow.set_tracking_uri("sqlite:///mlflow.db")
   MlflowClient().set_registered_model_alias("HeartDiseaseModel", "champion", "1")
   print("Alias 'champion' set.")
   ```
3. Create `scripts/load_model.py` (note the tracking URI + the `@champion` alias):
   ```python
   import mlflow
   import mlflow.sklearn
   import pandas as pd

   # The registry lives in the same SQLite store train.py wrote to.
   mlflow.set_tracking_uri("sqlite:///mlflow.db")

   # MLflow 3.x: reference the version carrying the "champion" alias (not a stage).
   model_uri = "models:/HeartDiseaseModel@champion"
   model = mlflow.sklearn.load_model(model_uri)
   print("Model loaded successfully from MLflow registry.")

   sample = pd.DataFrame([{
       'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
       'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
       'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
   }])
   prediction = model.predict(sample)
   probability = model.predict_proba(sample)
   print(f"Prediction: {'Disease' if prediction[0] == 1 else 'No Disease'}")
   print(f"Probability: {probability[0][1]:.4f}")
   ```
4. Run it: `python scripts/load_model.py` → should print a prediction.
5. Commit: `git add . && git commit -m "Task 5: Model registered with champion alias, loaded from registry"`

**✅ Done when:** `load_model.py` prints a prediction with no errors.

**Before you move on, you should know:**
- The difference between a **version URI** (`models:/HeartDiseaseModel/3`, pins a version) and an
  **alias URI** (`models:/HeartDiseaseModel@champion`, a movable "which model is live" pointer).
- Why the registry needs the SQLite backend from Task 4.

---

## Task 6 — FastAPI Serving Endpoint

**Goal:** Serve predictions over HTTP with an interactive `/docs` page.

**Key topics:** REST API · **Pydantic v2** request schema · loading the model once at startup · Swagger UI.

> **⚠️ Two corrections:** (1) the registry URI must use the **`@champion` alias** and the script
> must **set the tracking URI** (same as Task 5). (2) Pydantic **v2** replaced `.dict()` with
> **`.model_dump()`** — the tutorial's `.dict()` is deprecated.

### Steps
1. Create `scripts/serve.py` (registry-loading version, for local dev):
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   import mlflow
   import mlflow.sklearn
   import pandas as pd
   import uvicorn

   app = FastAPI(title="Heart Disease Prediction API",
                 description="MLOps Portfolio Project 1", version="1.0")

   mlflow.set_tracking_uri("sqlite:///mlflow.db")
   MODEL_URI = "models:/HeartDiseaseModel@champion"
   model = mlflow.sklearn.load_model(MODEL_URI)
   print("Model loaded and ready.")

   class PatientData(BaseModel):
       age: float
       sex: float
       cp: float
       trestbps: float
       chol: float
       fbs: float
       restecg: float
       thalach: float
       exang: float
       oldpeak: float
       slope: float
       ca: float
       thal: float

   @app.get("/")
   def root():
       return {"message": "Heart Disease Prediction API is running", "status": "healthy"}

   @app.get("/health")
   def health():
       return {"status": "ok", "model": "HeartDiseaseModel@champion"}

   @app.post("/predict")
   def predict(patient: PatientData):
       data = pd.DataFrame([patient.model_dump()])   # Pydantic v2 (not .dict())
       prediction = model.predict(data)[0]
       probability = model.predict_proba(data)[0][1]
       return {
           "prediction": int(prediction),
           "result": "Disease Detected" if prediction == 1 else "No Disease",
           "confidence": round(float(probability), 4),
       }

   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```
2. Run it and open the docs:
   ```bash
   python scripts/serve.py
   # http://127.0.0.1:8000/docs
   ```
   > Note: `serve.py` reads `mlflow.db` **directly** — you do **not** need `mlflow ui` running for the API.
3. In `/docs` → `POST /predict` → **Try it out** → paste the sample body → **Execute**:
   ```json
   { "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
     "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1 }
   ```
4. Screenshot the result → `screenshots/task6_api.png`.
5. Commit: `git add . && git commit -m "Task 6: FastAPI serving endpoint working"`

**✅ Done when:** `POST /predict` returns a prediction in the browser.

**Before you move on, you should know:**
- Why the model is loaded **once at startup** (not per request).
- How FastAPI generates `/docs` automatically from the Pydantic schema.
- The Pydantic v1 → v2 change (`.dict()` → `.model_dump()`).

---

## Task 7 — Containerise with Docker (via GitHub Codespaces)

**Goal:** Package the API into a Docker image and run it as a container — built and run in **GitHub Codespaces** (Linux, in the browser), because an 8 GB laptop can't run Docker Desktop comfortably. You still write the Dockerfile, build the image, and run the container — 100% of the skill, in the cloud.

**Key topics:** Dockerfile · image layers · slim images · **decoupling serving from the tracking server** · Codespaces port forwarding.

### Part 1 — Prepare files locally
1. Create `Dockerfile` (no extension) at the repo root:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY scripts/serve.py .
   COPY models/ ./models/
   EXPOSE 8000
   CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
2. Slim `requirements.txt` (the container loads the pkl, so **no mlflow**):
   ```
   pandas
   scikit-learn
   fastapi
   uvicorn
   pydantic
   ```
3. **Switch `serve.py` to load the pkl** instead of the registry — and this is where the
   tutorial trips you up:

   Replace the model-loading block:
   ```python
   # remove:  mlflow.set_tracking_uri(...) ; MODEL_URI = ... ; model = mlflow.sklearn.load_model(...)
   import pickle
   with open('models/heart_model.pkl', 'rb') as f:
       model = pickle.load(f)
   print("Model loaded from pkl file.")
   ```

   > **⚠️ Gotcha the tutorial misses — also delete the mlflow *imports*.** It says "leave the
   > rest of serve.py as is," but the top of the file still has `import mlflow` /
   > `import mlflow.sklearn`. Since you removed mlflow from `requirements.txt`, the container
   > crashes at startup with **`ModuleNotFoundError: No module named 'mlflow'`**. It never
   > failed locally because mlflow was installed globally on your laptop — the classic
   > **"works on my machine"** trap that containers exist to catch. **Fix:** remove those two
   > import lines. Final `serve.py` imports: `fastapi`, `pydantic`, `pandas`, `uvicorn`, `pickle`.

   Your final container-ready `serve.py` should look **exactly** like this:
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   import pandas as pd
   import uvicorn
   import pickle

   app = FastAPI(title="Heart Disease Prediction API",
                 description="MLOps Portfolio Project 1", version="1.0")

   # Container loads the model straight from the pickle — no mlflow, no tracking server.
   with open('models/heart_model.pkl', 'rb') as f:
       model = pickle.load(f)
   print("Model loaded from pkl file.")

   class PatientData(BaseModel):
       age: float
       sex: float
       cp: float
       trestbps: float
       chol: float
       fbs: float
       restecg: float
       thalach: float
       exang: float
       oldpeak: float
       slope: float
       ca: float
       thal: float

   @app.get("/")
   def root():
       return {"message": "Heart Disease Prediction API is running", "status": "healthy"}

   @app.get("/health")
   def health():
       return {"status": "ok", "model": "heart_model.pkl"}

   @app.post("/predict")
   def predict(patient: PatientData):
       data = pd.DataFrame([patient.model_dump()])
       prediction = model.predict(data)[0]
       probability = model.predict_proba(data)[0][1]
       return {
           "prediction": int(prediction),
           "result": "Disease Detected" if prediction == 1 else "No Disease",
           "confidence": round(float(probability), 4),
       }

   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

4. Make sure `models/heart_model.pkl` exists (re-run `python scripts/train.py` if unsure).
5. Create `.gitignore`:
   ```gitignore
   # Python
   __pycache__/
   *.pyc
   .ipynb_checkpoints/
   # Data (download separately) and local MLflow store (regenerated by train.py)
   data/
   mlruns/
   mlartifacts/
   mlflow.db
   # NOTE: models/heart_model.pkl is intentionally NOT ignored — the Docker COPY needs it.
   ```
   > If you keep a broad `*.pkl` ignore, add an exception `!models/*.pkl` so the served model
   > is still tracked — otherwise a fresh clone has no `models/` folder and `COPY models/` fails.

### Part 2 — Push to GitHub
6. Commit and create a **public** repo, then push:
   ```bash
   git add .
   git commit -m "Task 7: Dockerfile, gitignore, serve.py loads from pkl"
   git remote add origin https://github.com/<your-username>/mlops-heart-disease-pipeline.git
   git branch -M main
   git push -u origin main
   ```
   Confirm `Dockerfile` **and** `models/heart_model.pkl` are on GitHub.

### Part 3 — Build & run in Codespaces
7. Repo → green **`< > Code`** → **Codespaces** tab → **Create codespace on main** (a browser VS Code / Linux box; Docker is pre-installed).
8. In the Codespace terminal:
   ```bash
   docker --version                              # confirm Docker is there
   ls                                            # confirm your files came across
   docker build -t heart-disease-api:v1 .        # first build ~2-4 min
   docker run -p 8000:8000 heart-disease-api:v1  # should print "Model loaded from pkl file."
   ```

### Part 4 — Test the running API
9. Codespaces auto-forwards port 8000 — click the pop-up **"Open in Browser"** (or **PORTS** tab → globe icon on 8000). The URL looks like `https://<codespace>-8000.app.github.dev`.
10. Add `/docs`, run `POST /predict` with the sample body → you get a prediction **from a container running in the cloud**.
11. Screenshot the `/docs` result + the terminal showing the running container → `screenshots/task7_docker.png`.

### Part 5 — Clean up
12. Stop the container (`Ctrl+C`) and **stop the Codespace** (bottom-left → Stop Current Codespace) so you don't burn free hours (free tier ≈ 60 h/month; this task ≈ 1 h).
13. If you edited anything in the Codespace: `git add . && git commit -m "Task 7: containerised and tested in Codespaces" && git push`.

**✅ Done when:** the container runs in Codespaces and `POST /predict` returns a prediction via the forwarded URL.

> **Interview talking point:** *"I containerised the model API with Docker — wrote the Dockerfile,
> built the image, and ran it serving predictions over HTTP, in a Linux cloud environment that
> mirrors how images are built and run in production."*

**Common issues:**
- `docker: command not found` → the Codespace didn't load Docker; recreate it on `main`.
- Build fails on `COPY models/` → the pkl wasn't pushed (caught by `.gitignore`); fix the ignore, recommit, push.
- `ModuleNotFoundError: mlflow` → you didn't remove the mlflow imports from `serve.py` (see the gotcha above).
- Port 8000 "not found" → the container isn't running; re-run `docker run`.
- `/predict` validation error → your JSON body must have all **13** fields exactly.

**Before you move on, you should know:**
- What each Dockerfile line does, and why we install `requirements.txt` **before** copying code (layer caching).
- Why the container loads a **pkl** instead of the registry (small, standalone, no tracking server).

---

## Task 8 — README & Publish

**Goal:** A README a hiring manager can skim and understand in two minutes.

**Key topics:** documentation as an interface · reproducibility framing · portfolio presentation.

### Steps
1. Write `README.md` — see the finished one in this repo. It covers: what the project
   demonstrates, an architecture diagram, tech stack, dataset, **results** (Accuracy **88.33%**,
   F1 **0.857**), project structure, run instructions, API usage, and a per-stage learnings table.
   > Use your **real** numbers and the **corrected** commands (SQLite `--backend-store-uri`,
   > `@champion`, `.model_dump()`) — not the tutorial's placeholders.
2. Ensure the repo is **public** and push:
   ```bash
   git add .
   git commit -m "Task 8: README complete, project published"
   git push
   ```
3. Add the repo link to your **LinkedIn Featured** section.

**✅ Done when:** the repo is public and the README renders (diagram + screenshots visible).

---

## Appendix A — Pitfalls & Fixes (quick reference)

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `PermissionError` reading the CSV | zip extracted into a *folder* named like the file | point the loader at the real file |
| 2 | MLflow runs "missing" / registry unavailable | using the deprecated file store | `mlflow.set_tracking_uri("sqlite:///mlflow.db")` |
| 3 | MLflow UI empty | UI defaulted to `./mlruns` | launch with `--backend-store-uri sqlite:///mlflow.db` |
| 4 | `mlflow: not recognized` | per-user install not on PATH | run `python -m mlflow …` |
| 5 | `models:/…/Production` won't load | stages removed in MLflow 3.x | use alias `models:/HeartDiseaseModel@champion` |
| 6 | `.dict()` deprecated | Pydantic v2 | use `.model_dump()` |
| 7 | API serves a stale model | MLflow rewrite dropped the `pickle.dump` | keep the pkl dump in `train.py` (train/serve sync) |
| 8 | Container: `ModuleNotFoundError: mlflow` | unused `import mlflow` + slim requirements | remove the dead imports |

## Appendix B — One-liners for interviews

- *"End-to-end MLOps pipeline: ingestion → training → MLflow tracking & registry → FastAPI serving → Docker."*
- *"I promote models with registry **aliases** (champion/challenger) — MLflow 3 deprecated stages."*
- *"I **decouple serving from the tracking server** by baking a model artifact into the image — small and independently deployable."*
- *"I prevent **model drift** by regenerating the served artifact on every retrain."*

---

*Built as MLOps Portfolio Project 1 — the local train-to-serve pipeline.*
