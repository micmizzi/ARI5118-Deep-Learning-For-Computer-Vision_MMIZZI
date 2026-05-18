# ARI5118 – Deep Learning for Computer Vision
## Interactive CNN Convolution Simulator

> **Course:** ARI5118 – Deep Learning for Computer Vision  
> **Author:** M. Mizzi  
> **Platform:** Google Colab (runs in-browser on any OS, including Windows — no local Python install required)  
> **Tool:** Interactive Gradio web app for visualising and experimenting with convolutional operations on MNIST images.

---

## What This App Does

This simulator lets you explore the core building block of Convolutional Neural Networks (CNNs) — the **convolution operation** — interactively, without writing any code.

You can:

- Pick any of 5 randomly sampled MNIST handwritten-digit images as input.
- Configure **two independent convolution setups** side by side (kernel type, kernel size, padding, stride, dilation).
- Instantly compare the feature maps produced by each setup.
- Inspect the raw kernel matrix and output shape / receptive field for each setup.

Supported kernels: **Identity**, **Edge Detection (Laplacian)**, **Sharpen**, **Blur (Box)**, **Emboss** — in 3×3 and 5×5 variants.

---

## Why Google Colab?

Google Colab gives you a free, ready-to-use Python environment in your browser with no installation required on Windows. It comes with TensorFlow, NumPy, Pandas, and Matplotlib pre-installed. You only need a Google account.

| Requirement | Notes |
|---|---|
| Google account | Free — [sign up](https://accounts.google.com) if needed |
| Kaggle account | Free — required for dataset download |
| Browser | Chrome or Firefox recommended |
| Local Python / CUDA | **Not required** |

> The Colab CPU runtime is used throughout. No GPU tier is needed for this app.

---

## Quick Start (Windows + Google Colab)

### Step 1 — Get your Kaggle API token

The app downloads the MNIST dataset automatically via `kagglehub`. You need a free Kaggle API token.

1. Sign in at [https://www.kaggle.com](https://www.kaggle.com).
2. Click your profile picture (top right) → **Settings**.
3. Scroll to the **API** section → click **Create New Token**. Copy the key!!


---

### Step 2 — Open Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.

Create a new notebook: **File → New notebook**.

> Alternatively, open this repository directly in Colab via **File → Open notebook → GitHub** and paste the repository URL.

---

### Step 3 — Clone the repository

In the first Colab cell, run:

```python
!git clone https://github.com/micmizzi/ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI.git
%cd ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI/Simulator
```

---

### Step 4 — Install missing dependencies

```python
!pip install -r requirements.txt -q
```

This installs only `gradio` and `kagglehub`. All other packages (TensorFlow, NumPy, Pandas, Matplotlib) are already available in Colab.

---

### Step 5 — Set up Kaggle credentials

The safest method in Colab uses the built-in **Secrets** manager (no file uploads needed).

#### Method A — Colab Secrets (recommended)

1. In Colab, click the **🔑 key icon** in the left sidebar → **Add new secret**.
2. Add two secrets:
   - Name: `KAGGLE_USERNAME` → Value: your Kaggle username
   - Name: `KAGGLE_KEY` → Value: paste the `key` value from your Kaggle notification window 
3. Enable the toggle **"Notebook access"** for both secrets.
4. In a Colab cell, run this **once** before launching the app:

```python
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = userdata.get("KAGGLE_KEY")
```

### Step 6 — Launch the app

```python
!python Simulator_gradio_app.py
```

After a few seconds you will see output like:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxx.gradio.live
```

**Click the public URL** (`gradio.live` link) to open the interactive app in a new browser tab. The local URL does not work in Colab — always use the public one.


---

### Complete Colab Cell Sequence (copy-paste reference)

Run these cells in order in a single Colab notebook:

```python
# Cell 1 — Clone repo
!git clone https://github.com/micmizzi/ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI.git
%cd ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI/Simulator

# Cell 2 — Install dependencies
!pip install -r requirements.txt -q

# Cell 3 — Kaggle credentials (Secrets method)
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = userdata.get("KAGGLE_KEY")

# Cell 4 — Launch the app
!python Simulator_gradio_app.py
```

---

## Project Structure

```
ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI/
│
├── Simulator_gradio_app.py   # Main Gradio application
├── requirements.txt          # Colab pip dependencies
└── README.md                 # This file
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'gradio'`** — Run Cell 2 again. If the error persists, restart the Colab runtime (**Runtime → Restart session**) and re-run all cells from Cell 1.

**`401 – Unauthorized` or `kaggle.json not found`** — Your Kaggle credentials were not picked up. Re-run Cell 3 (Secrets method) or re-upload `kaggle.json` (Method B). Make sure "Notebook access" is toggled on for both secrets.

**`Invalid magic number` error on startup** — The MNIST file downloaded incompletely. Run the cell below to clear the cache, then re-run Cell 4:

```python
import shutil, os
cache = os.path.expanduser("~/.cache/kagglehub")
if os.path.exists(cache):
    shutil.rmtree(cache)
    print("Cache cleared.")
```

**App shows a blank page or spinner that never resolves** — Make sure you opened the `gradio.live` public URL, not the `127.0.0.1` local URL. The local URL is unreachable from outside the Colab VM.

**App loads but images are blank on first open** — This is a known Gradio rendering delay. Move any slider slightly; the plots will update immediately.

**`tensorflow` CPU warnings (AVX/FMA)** — These are informational messages from TensorFlow and do not affect results. Suppress them by adding the following line at the top of Cell 4:

```python
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
!python Simulator_gradio_app.py
```

**Colab runtime disconnects mid-session** — Free Colab sessions time out after ~90 minutes of inactivity. Re-run all four cells to restart. The Kaggle dataset re-downloads quickly from cache if it still exists in the session.


## License

For academic use only as part of ARI5118 coursework.
