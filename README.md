## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/RobbenLe/AI_lifestyle_Project.git
cd ai-health-model


### 2. Create and activate a virtual environment

#Windows

python -m venv .venv
.venv\Scripts\activate


#macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


### 3. Install Dependencies 
pip install -r requirements.txt


### Install Streamlit 
pip install streamlit
### Run the streamlin app
streamlit run streamlit_app.py

#If you only wants to run the Streamlit app + models, yoi do not need PostgreSQL running.
Those packages will just be installed but not used.