AI Health Model – Lifestyle Persona Classification

This project builds a machine learning model that classifies users into 5 lifestyle personas based on daily health indicators.

The model is trained using a Keras Conv1D (CNN) neural network and deployed through a Streamlit web application for interactive use.

What This Project Does

The AI model uses daily aggregated data to classify users into one of the following personas:

    high_workout

    healthy

    low_activity

    lazy_obese

    over_trained

Input features

    Steps per day

    Average stress level

    Daily average heart rate (heart_rate_per_point)

Output

    Predicted lifestyle persona

    Model confidence

    Positive, human-readable feedback

    Probability for each class

📁 Project Structure
    ai-health-model/
    │
    ├── data/                    # CSV datasets (steps, heart rate, stress)
    │
    ├── notebooks/               # Jupyter notebooks (training & experiments)
    │   ├── 01_explore_data.ipynb
    │   ├── 02_Keras_Neural_Network.ipynb
    │   └── 03_Keras_HR_Per_Point.ipynb
    │
    ├── saved/                   # Trained models & preprocessing artifacts
    │   ├── activity_cnn_5classes_v2.keras
    │   ├── scaler_activity_v2.pkl
    │   └── label_encoder_activity_v2.pkl
    │
    ├── streamlit_app.py         # Original Streamlit app
    ├── streamlit_v2.py          # Final Streamlit app (recommended)
    │
    ├── requirements.txt
    ├── README.md
    └── .gitignore

⚠️ Requirements (IMPORTANT)
🐍 Python Version

    This project requires Python 3.10.

    TensorFlow does not support Python 3.12+ (including Python 3.14).
    If you use a newer Python version, TensorFlow will not install.

    Recommended version:

    Python 3.10.x

▶️ How to Run the Project
1️⃣ Clone the repository

    git clone https://github.com/RobbenLe/AI_lifestyle_Project.git
    cd ai-health-model

2️⃣ Create and activate a virtual environment (project root)
Windows (recommended)

    py -3.10 -m venv .venv_tf
    .venv_tf\Scripts\activate


You should see:

    (.venv_tf)

macOS / Linux

    python3.10 -m venv .venv_tf
    source .venv_tf/bin/activate

3️⃣ Install dependencies

    pip install --upgrade pip
    pip install -r requirements.txt


If TensorFlow is not included in requirements.txt, install it manually:

    pip install tensorflow==2.15.0



    ---------------------------------------------------------------------------

💻 Using Jupyter Notebooks in Visual Studio Code (IMPORTANT)

Jupyter notebooks in VS Code do not automatically use your virtual environment.
You must explicitly select the correct Python interpreter and kernel.

4️⃣ Install required VS Code extensions

    In VS Code, open Extensions and install:

    Python (by Microsoft)

    Jupyter (by Microsoft)

5️⃣ Open the project in VS Code

    In VS Code:

    File → Open Folder → ai-health-model


Make sure you open the project root folder.

6️⃣ Select the correct Python interpreter

    Press Ctrl + Shift + P

    Type:

    Python: Select Interpreter


Choose the interpreter pointing to:

    .venv_tf\Scripts\python.exe


Check the bottom-right corner of VS Code. You should see:

    Python 3.10 (.venv_tf)

7️⃣ Install Jupyter kernel support (inside the virtual environment)

Open the VS Code terminal (with .venv_tf activated):

    pip install notebook ipykernel

8️⃣ Select the correct kernel in a notebook

Open any .ipynb file in the notebooks/ folder

In the top-right corner, click the kernel selector

Choose:

    Python 3.10 (.venv_tf)

9️⃣ Verify the kernel is correct

Run this cell inside the notebook:

    import sys
    print(sys.executable)


Expected output should point to:

    .../ai-health-model/.venv_tf/...


Verify TensorFlow:

    import tensorflow as tf
    print(tf.__version__)


Expected output:

    2.15.0

🌐 Running the Streamlit App (Recommended)
    Go to your Project root, copy the path
    Open terminal:

      C:\Windows\System32>cd "paste your project root here"


4️⃣ Start the app
    "..\.venv\Scripts\python.exe" -m streamlit run streamlit_v2.py


Open your browser at:

    http://localhost:8501

🧪 Using the Application

Input:

    Steps per day

    Average stress level (0–100)

    Average heart rate per day (70–140 bpm)

    Click Classify

The app displays:

    Predicted lifestyle persona

    Model confidence

    Positive feedback

    Probability per class

🧠 Training the Model (Optional)

If you want to retrain or inspect the model:

Open notebooks in the notebooks/ folder

Recommended order:

    01_explore_data.ipynb

    02_Keras_Neural_Network.ipynb

    03_Keras_HR_Per_Point.ipynb

The trained model and preprocessing files are saved in the saved/ directory.

💾 Saved Artifacts (Used by Streamlit)

The Streamlit app depends on these files:

    saved/
    ├── activity_cnn_5classes_v2.keras
    ├── scaler_activity_v2.pkl
    └── label_encoder_activity_v2.pkl


⚠️ Do not rename or delete these files unless you retrain the model.

🛠️ Common Issues
❌ TensorFlow not found

Error example:

ModuleNotFoundError: No module named 'tensorflow'


Cause: Unsupported Python version.

Fix:

Install Python 3.10

Create a new virtual environment

Reinstall dependencies

ℹ️ Notes

PostgreSQL is not required to run the model or the Streamlit app.

Some database packages may exist in requirements.txt but are not used for inference.

This project is intended for educational, research, and prototyping purposes.

👤 Author

Robben Le
Internship Project – AI Lifestyle Classification
Inholland University
