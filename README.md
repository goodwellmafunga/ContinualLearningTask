# ROW Algorithm for Continual Learning

This project demonstrates the implementation of the ROW (Replay-On-Write) algorithm for continual learning. The experiments focus on applying this algorithm to the CIFAR-10 and CIFAR-100 datasets using Jupyter notebooks.

## Project Overview

### Main Components
- **Continual Learning with Replay Buffer**: Implemented to simulate continual learning by replaying stored data batches from previous tasks.
- **Within-Distribution (WP) and Out-of-Distribution (OOD) Tasks**: The project aims to distinguish between within-distribution accuracy and OOD detection performance.
- **Experiment Logging**: Performance metrics such as accuracy, precision, recall, and F1-score are logged for each epoch of training.

### Notebooks:
1. `final_CLandOOD.ipynb`: Main notebook demonstrating continual learning and OOD detection using CIFAR-10 and CIFAR-100 datasets.
2. `ROW_Algorithm_for_CIFAR10.ipynb`: Notebook focused on CIFAR-10 dataset experiments.
3. `ROW_Algorithm_for_CIFAR100.ipynb`: Notebook focused on CIFAR-100 dataset experiments.

### Key Features:
- **Metrics Logging**: Logs accuracy, precision, recall, and F1-score for each experiment.
- **Confusion Matrix Visualisation**: Plots confusion matrices for both WP and OOD predictions to provide a visual insight into model performance.
- **Results Comparison**: Compare results across different datasets (CIFAR-10 and CIFAR-100).

## Getting Started

### Installation

To run the notebooks, ensure you have Python and the following dependencies installed:

1. Clone the repository:

    ```bash
    git clone https://github.com/goodwellmafunga/ROW_Algorithm_Continual_Learning.git
    cd ROW_Algorithm_Continual_Learning
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Lab or Jupyter Notebook:

    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```

4. Open and run the `final_CLandOOD.ipynb` notebook for a complete walkthrough of the experiment.

## Requirements
- Python 3.8+
- Jupyter Lab or Jupyter Notebook
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- seaborn
- tqdm
- scikit-learn

## Running the Experiments
To reproduce the results:
- Open the `final_CLandOOD.ipynb` notebook.
- Run all the cells to observe the training, evaluation, and visualisation of the experiment.



