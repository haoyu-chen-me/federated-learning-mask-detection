# federated-learning-mask-detection

Federated learning project exploring the impact of training configurations and non-IID data on mask detection using CIFAR-10 and the Flower framework.

---

## 📁 Project Structure

```
federated-learning-mask-detection/
│
├── code/
│   ├── impact of training configurations/
│   │   ├── client.py
│   │   ├── server_e1r20.py
│   │   ├── server_e5r4.py
│   │   ├── server_e10r2.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── evaluate_all.py
│   │   ├── predict_gui.py
│   │   └── plot.py, e1r20_plot.py ...
│   │
│   └── non-IID data on mask detection using CIFAR-10 and the Flower framework/
│       ├── client.py
│       ├── server.py
│       ├── main.py
│       ├── run_all.py
│       ├── model.py
│       ├── dataset.py
│       ├── train_utils.py
│       ├── visualize.py
│       └── Untitled.ipynb
│
├── figures and data tables/
│   ├── accuracy plot.png
│   ├── gui demo.png
│   ├── non_iid_distribution.png
│   ├── global_client_accuracy.png
│   ├── accuracy_log_CIFAR.csv
│   ├── accuracy_log_E1R20.csv
│   ├── accuracy_log_E5R4.csv
│   └── accuracy_log_E10R2.csv
│
└── README.md
```

---

## 📝 Project Description

This project investigates the efficiency of federated learning applied to a face mask detection task using both IID and non-IID data scenarios. The experiments compare different training configurations (E1R20, E5R4, E10R2) and their influence on model convergence, performance, and communication efficiency. Additionally, the impact of non-IID distribution is explored through CIFAR-10 and a custom dataset.

---

## 🔗 Dataset Access

Due to GitHub's 25MB file limit, the dataset is hosted externally.

👉 **Download dataset here**:  
[https://drive.google.com/drive/folders/1JtPd9YWX46-C-KGhxys-9XeRTl9wQ6O9?usp=drive_link](https://drive.google.com/drive/folders/1JtPd9YWX46-C-KGhxys-9XeRTl9wQ6O9?usp=drive_link)

After download, place the dataset in the appropriate folder, e.g.:

```
data/mask/
data/nomask/
```

---

## 📦 Requirements

This project requires:

- Python 3.9+
- PyTorch
- Flower (`flwr`)
- numpy
- pandas
- matplotlib
- tkinter (for GUI)

To install:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Training with Different Configurations (E1R20, E5R4, E10R2)

Go to `code/impact of training configurations/` and run:

```bash
python server_e1r20.py
# or
python server_e5r4.py
# or
python server_e10r2.py
```

Then run clients:

```bash
python client.py
```

To visualize results:

```bash
python e1r20_plot.py
```

### Option 2: Non-IID Data Experiment with CIFAR-10

Go to `code/non-IID data on mask detection using CIFAR-10 and the Flower framework/` and run:

```bash
python server.py
python client.py
```

Or run all at once:

```bash
python run_all.py
```

---

## 🖼️ GUI Demo

To launch the GUI for mask detection:

```bash
python predict_gui.py
```

The GUI interface allows the user to test trained models on custom images.

---

## 📊 Results

Training results are visualized in:

- `accuracy plot.png`
- `non_iid_distribution.png`
- `global_client_accuracy.png`

Raw accuracy logs are available in:

- `accuracy_log_CIFAR.csv`
- `accuracy_log_E1R20.csv`
- `accuracy_log_E5R4.csv`
- `accuracy_log_E10R2.csv`

---

## 📄 License

This project is for academic and non-commercial use only.

---

## 👤 Author

**Chenhao Yu**  
Computer Engineering Major, Stony Brook University

---

## 🙏 Acknowledgements

This project was developed as part of a group study on distributed learning, under the course  
**Distributed Machine Learning: Foundations and Algorithms**.
