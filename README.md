# ARI5118 – Deep Learning for Computer Vision

**Course:** ARI5118 – Deep Learning for Computer Vision  
**Author:** M. Mizzi  
**Platform:** Google Colab (no local Python install required)

MSc assignment covering CNN foundations — convolution arithmetic, spatial reasoning, and receptive fields — supported by an interactive simulator, a worked-notebook, study materials, and a quiz.

---

## Repository Contents

| Item | Description |
|---|---|
| `Simulator/` | Interactive Gradio app for visualising CNN convolution operations on MNIST images |
| `walkthrough.ipynb` | Step-by-step Jupyter notebook covering convolution, spatial arithmetic, and receptive field computations |
| `study_notes.pdf` | Compiled study notes on CNN foundations |
| `slides.pdf` | Course slide deck |
| `quiz_with_rationale.pdf` | Multiple-choice quiz with explanations |
| `Further reading/` | Seminal papers (LeCun 1998, AlexNet, and others) + paper summary |
| `ai_journal.pdf` | AI usage journal used throughout the project |

---

## Simulator – Quick Start (Google Colab)

The simulator lets you compare two convolution setups side-by-side on MNIST images, choosing kernel type, size, padding, stride, and dilation.

**Requirements:** Google account · Free Kaggle account (for dataset)

```python
# Cell 1 — Clone repo
!git clone https://github.com/micmizzi/ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI.git
%cd ARI5118-Deep-Learning-For-Computer-Vision_MMIZZI/Simulator

# Cell 2 — Install dependencies
!pip install -r requirements.txt -q

# Cell 3 — Kaggle credentials (via Colab Secrets)
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = userdata.get("KAGGLE_KEY")

# Cell 4 — Launch the app
!python Simulator_gradio_app.py
```

Open the `gradio.live` public URL printed in the output (the local `127.0.0.1` URL does not work in Colab).

For full setup instructions and troubleshooting, see [`Simulator/README.md`](Simulator/README.md).

---

## License

For academic use only as part of ARI5118 coursework.
