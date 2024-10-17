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



Interpretation and Analysis of the Visual Results
1. CIFAR-10 Loss Over Epochs
The first graph shows the model's loss decreasing as the training progresses over 50 epochs. Starting from around 2.0, the loss rapidly declines, stabilising after 20 epochs. The sharp decline in loss indicates that the model is learning effectively in the early stages, quickly reducing prediction errors. By epoch 20, the loss reaches a low level close to 0.05, showing that the model has successfully minimised classification errors on the CIFAR-10 dataset. This trend suggests that the model's convergence is solid and the learning process is smooth.

2. t-SNE Visualization of Extracted Features
The second visualisation employs t-SNE to project high-dimensional features extracted by the model into a 2D space. Different colours represent various classes from the CIFAR-10 dataset, showing a clear separation between the classes, meaning that the feature extractor has learned to differentiate between the 10 classes. The clusters indicate that the model's feature extractor is working effectively, mapping similar inputs close to each other, which helps in accurate classification. There is some overlap between a few clusters, which could suggest room for further improvement, but overall, the representation is robust.

3. CIFAR-10 OOD Precision, Recall, and F1 Score Over Epochs
The third graph tracks the performance metrics for out-of-distribution (OOD) detection, including precision, recall, and F1-score. These metrics improve rapidly and stabilise around epoch 20, where they all hover near 1.0, indicating excellent OOD detection performance. The model becomes adept at recognising whether a given input is from a new, unseen task or belongs to the already learned tasks. The stability of these metrics throughout the remaining epochs suggests that the model consistently performs well in OOD detection once it has learned the data distribution.

4. CIFAR-10 WP Precision, Recall, and F1 Score Over Epochs
Similar to the OOD metrics, this graph shows the within-project (WP) task metrics. Precision, recall, and F1 scores also reach close to 1.0 after epoch 20, indicating that the model is effectively classifying within-task data. The convergence of all three metrics around a high value suggests that the model has learned to make precise and balanced predictions for data belonging to the current task without suffering from catastrophic forgetting.

5. CIFAR-10 WP and OOD Accuracy Over Epochs
The fifth graph compares WP and OOD accuracy across epochs, with both metrics following a similar upward trajectory. WP accuracy starts at around 70% and reaches close to 100% by epoch 20, while OOD accuracy follows a comparable path. This indicates that the model balances learning new tasks (WP) and distinguishing old tasks (OOD), with both accuracies reaching near-optimal levels. This balance demonstrates the effectiveness of the dual-head architecture (OOD and WP) used in the ROW algorithm.