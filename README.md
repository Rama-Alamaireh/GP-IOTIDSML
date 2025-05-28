# IDS for IoT Networks Using Machine Learning

## About the Project
This project presents a comprehensive study on the use of Machine Learning (ML) algorithms to build an efficient Intrusion Detection System (IDS) tailored for Internet of Things (IoT) networks. By leveraging the recent and realistic CICIoT2023 dataset, the study evaluates multiple supervised ML models under various preprocessing and balancing strategies. The goal is to identify lightweight and accurate models that can effectively detect different types of cyberattacks in resource-constrained IoT environments.
Among the evaluated models, XGBoost achieved the highest performance, making it the final selected model for reliable and scalable intrusion detection in IoT settings.

## Technologies Used
- Python
- Pandas (data handling)
- Scikit-learn (model evaluation and preprocessing)
- XGBoost (final ML model)
- Streamlit (for interactive web app)


## How to Run the App
Online Access:
You can try the app directly without installation from this link:
https://idsiotml.streamlit.app

## How to Run Locally
1. Clone the repository
```bash
git clone https://github.com/Rama-Alamaireh/GP-IOTIDSML.git
cd GP-IOTIDSML
```
2. Create a virtual environment and activate it (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
3. Install required packages
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app
```bach
streamlit run streamlit_app.py
```
Open your browser and go to the local link shown (usually: http://localhost:8501)
## How to Use
 1. Upload your IoT network dataset file in CSV format via the Streamlit app.
 2. The app will predict possible cyberattacks on the network data using the trained XGBoost model.
 3. Results, including detected attacks and risk scores, will be displayed interactively.

## Dataset
This project uses the CICIoT2023 dataset for training and evaluation.
You can download the dataset from the following sources:

UNB CIC IoT Dataset on Kaggle
https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset
