# Advanced Machine Learning Frameworks and Analyses

## Project Overview

This repository is a comprehensive showcase of my work in the machine learning space, featuring everything from foundational neural network implementations to in-depth analyses of substantial datasets such as the Million Song Dataset. Through this project, I aim to demonstrate the versatility and depth of machine learning applications across various domains including materials science, music analysis, and unsupervised learning techniques.

### Key Projects

- **BandGapPrediction.ipynb**: Predict the band gaps of materials, facilitating the discovery of new materials for electronics and renewable energy solutions.
- **Clustering.ipynb**: Employ clustering algorithms to unveil hidden patterns within complex datasets, highlighting the practical utility of unsupervised learning.
- **Million_song.ipynb**: Analyze the Million Song Dataset to predict song attributes and classify songs. This analysis was particularly challenging due to the vast amount of data, and was conducted on Amazon EMR (Elastic MapReduce) to leverage distributed computing capabilities, showcasing the power of machine learning in processing and understanding vast datasets.
- **pytorch_nn.py**: Validate the practical use of PyTorch in constructing and training neural networks, emphasizing ease of use and efficiency.
- **self_implement_nn.py**: Build a neural network from the ground up, demonstrating a fundamental understanding of neural network mechanics and underlying mathematical principles.

### Dataset Sizes and Distributed Computing Experience

The datasets employed in these projects vary greatly in size, from smaller datasets processed locally to larger ones necessitating distributed computing. Notably, the `Million Song Dataset`, which is approximately 300GB in size, stands out for its volume. To efficiently process and analyze this substantial dataset, Amazon EMR (Elastic MapReduce) was leveraged, demonstrating my capability to utilize cloud-based distributed computing platforms effectively. This approach not only facilitated the handling of large-scale data but also underscored the versatility and scalability of the computational strategies employed in my machine learning projects.

I have extensive experience in designing, building, and training machine learning models within distributed computing environments. Utilizing platforms like Apache Spark for data processing and PyTorch distributed for model training has enabled efficient management and analysis of large datasets.

## Installation and Usage

Ensure Python 3.8+, PyTorch, Pandas, NumPy, and Scikit-learn are installed. Refer to `requirements.txt` for detailed installation instructions.

```bash

pip install -r requirements.txt
```

### Running Notebooks

Open and run the `.ipynb` files using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook BandGapPrediction.ipynb
```

## Executing Python Scripts

To run the Python scripts included in this project, use the following commands in your terminal or command prompt. Ensure you are in the project's root directory and have activated your virtual environment if you are using one.

```bash
python pytorch_nn.py
python self_implement_nn.py
```
