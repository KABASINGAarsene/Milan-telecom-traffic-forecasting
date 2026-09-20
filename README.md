# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## Project Overview
This repository contains an empirical research project evaluating the predictive capabilities of advanced deep learning architectures for urban telecommunications forecasting. Using a massive 20GB dataset of cellular activity across Milan, Italy, the study implements a 3x3 experimental grid to compare Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and Bidirectional LSTMs (Bi-LSTM) architectures for one-step-ahead traffic prediction.

The primary objective is to balance predictive capacity during extreme traffic spikes against computational efficiency and memory constraints. 

## Dataset
* **Source:** A multi-source dataset of urban life in the city of Milan and the Province of Trentino.
* **Download:** (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
* **Characteristics:** Over 2 months of Call Detail Records (CDRs) recorded at 10-minute intervals across 10,000 spatial grid squares.

## Hardware & Environment Setup
This pipeline was engineered and tested in a cloud-based Linux environment. Due to the dataset's size (20GB uncompressed) and the complexity of bidirectional recurrent layers, the following hardware is recommended:
* **Accelerator:** Dual NVIDIA Tesla T4 GPUs (`GPU T4 x2`)
* **RAM:** 16 GB VRAM per GPU

## Installation
Ensure you have Python 3.8+ installed. You can install all necessary dependencies via the provided requirements file:

```bash
pip install -r requirements.txt
```
## Reproducibility & Execution Instructions

To replicate the main findings, follow these exact steps to avoid Out-Of-Memory (OOM) kernel crashes:

1. **Clone the Repository:**
   ```bash
   git clone Repo URL
   cd  Repo Directory Name
   ```

1. **Download the Data:**  
   Place the raw Milan telecommunications `.txt` or `.csv` files into a local `/data` directory.

2. **Open the Notebook:**  
   Launch `predicting-telecom-traffic.ipynb` in your preferred environment (Kaggle and Google Colab are recommended for T4 GPU access).

3. **Data Ingestion (Critical Step):**  
   Run the data processing cells. The script uses a chunked ingestion strategy (`chunksize=100000`) and a vector accumulator to process the 20GB dataset. It will automatically isolate the target zones (e.g., Square 5059 and Square 5161) and save them to a lightweight `filtered_target_squares.csv` file.

4. **Model Training:**  
   Execute the 3x3 experimental grid cells. The script will dynamically apply MinMax scaling, construct 24-hour look-back sliding windows, and train all 9 model iterations.   