# t-SNE Analysis of Human Activity Recognition Dataset

This project implements a comprehensive t-SNE (t-Distributed Stochastic Neighbor Embedding) analysis pipeline for the Human Activity Recognition dataset, following the methodology outlined in the research requirements.

## Dataset Overview

- **Source**: Human Activity Recognition dataset (Kaggle)
- **Size**: 7,352 samples × 562 features + target variable
- **Classes**: 6 activity types
  - LAYING (1,407 samples)
  - STANDING (1,374 samples)
  - SITTING (1,286 samples)
  - WALKING (1,226 samples)
  - WALKING_UPSTAIRS (1,073 samples)
  - WALKING_DOWNSTAIRS (986 samples)
- **Features**: Accelerometer and gyroscope sensor data with various statistical measures

## Methodology

The analysis follows a systematic approach as specified in the research requirements:

### a. Dataset Selection
- Selected Human Activity Recognition dataset (~7K samples, comparable to MNIST size)
- Features extracted from smartphone accelerometer and gyroscope data
- 562 numerical features representing time and frequency domain variables

### b. Hyperparameter Configuration
Tested t-SNE with the following hyperparameters:
- **Perplexity**: [5, 30, 50, 100]
- **Early Exaggeration**: [4, 12]
- **Total Combinations**: 8 different parameter settings

### c. Dimensionality Reduction with PCA
- Applied PCA to reduce dimensions from 562 → 50
- Retained 87.42% of total variance
- First component explains 50.69% of variance
- Used PCA-reduced data as input to t-SNE

### d. t-SNE Implementation
- Applied t-SNE to map 50D PCA space to 2D visualization
- Used 5,000 randomly sampled points for computational efficiency
- Fixed random seed for reproducibility
- 1,000 iterations with early stopping

### e. Hyperparameter Analysis
Systematic evaluation of all parameter combinations to examine:
- Visual clustering quality
- KL divergence convergence
- Neighborhood preservation

### f. Quantitative Evaluation
Implemented three key evaluation metrics:

1. **Shepard Correlation**: Measures correlation between original and embedded distances
2. **Continuity**: Evaluates preservation of local neighborhoods (k=7)
3. **Mean Local Error**: Quantifies ranking errors in neighborhood preservation

## Key Results

### Best Configuration
**Perplexity=100, Early Exaggeration=4** achieved the best overall performance:
- Shepard Correlation: 0.7593
- Continuity: 0.4170
- Local Error: 30.74
- KL Divergence: 1.1108

### Key Findings

1. **Perplexity Impact**:
   - Higher perplexity (100) generally produced better results
   - Lower perplexity (5) led to more fragmented clusters
   - Optimal range: 50-100 for this dataset

2. **Early Exaggeration Effect**:
   - Lower early exaggeration (4) slightly outperformed higher values (12)
   - Effect less pronounced than perplexity changes

3. **PCA Preprocessing**:
   - Successfully reduced dimensionality while retaining 87% variance
   - Improved t-SNE computational efficiency
   - First 10 components capture 70% of variance

4. **Class Separation**:
   - Clear separation between static (LAYING, SITTING, STANDING) and dynamic activities
   - Walking activities show intermediate clustering
   - Some overlap between similar activities (SITTING/STANDING)

## Generated Visualizations

1. **`eda_plots.png`**: Exploratory data analysis including:
   - Target class distribution
   - Feature correlation heatmap
   - Feature variance analysis
   - Sample feature distributions

2. **`pca_analysis.png`**: PCA results showing:
   - Explained variance by component
   - Cumulative explained variance

3. **`tsne_experiments.png`**: t-SNE results for all parameter combinations:
   - 2×4 grid showing all perplexity/early exaggeration combinations
   - Color-coded by activity class
   - Visual comparison of clustering quality

4. **`evaluation_summary.png`**: Quantitative evaluation metrics:
   - Bar charts comparing all configurations
   - Shepard correlation, continuity, local error, and KL divergence

## Project Structure

```
tsne/
├── train.csv                  # Dataset
├── tsne_analysis.ipynb       # Main analysis notebook
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── tsne_env/                 # Virtual environment
├── eda_plots.png            # Exploratory data analysis
├── pca_analysis.png         # PCA variance analysis
├── tsne_experiments.png     # t-SNE parameter comparison
└── evaluation_summary.png   # Quantitative evaluation
```

## Usage

### Setup
```bash
# Create virtual environment
python3 -m venv tsne_env
source tsne_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```bash
# Launch Jupyter notebook
jupyter notebook tsne_analysis.ipynb
```

### Notebook Features
The analysis notebook (`tsne_analysis.ipynb`) provides:
- **Interactive Analysis**: Each function is in a separate cell for step-by-step execution
- **Complete Pipeline**: Single cell execution for full analysis
- **Automated EDA**: Comprehensive exploratory data analysis with visualizations
- **Data preprocessing**: Normalization and feature preparation
- **PCA dimensionality reduction**: With variance analysis
- **Systematic t-SNE parameter exploration**: Testing all hyperparameter combinations
- **Quantitative evaluation**: Multiple metrics (Continuity, Local Error, Shepard Correlation)
- **Comprehensive visualizations**: All plots generated and saved automatically

### Usage Options:

1. **Complete Pipeline**: Run the main execution cell for full automated analysis
2. **Step-by-Step**: Uncomment and run individual cells for interactive exploration
3. **Customization**: Easily modify parameters and re-run specific analysis steps

## Technical Details

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- scipy >= 1.9.0
- jupyter >= 1.0.0

### Implementation Notes
- Uses scikit-learn's TSNE implementation
- StandardScaler for feature normalization
- Random sampling for computational efficiency
- Comprehensive error handling and progress reporting

## Evaluation Metrics Explained

1. **Shepard Correlation**: 
   - Measures how well pairwise distances are preserved
   - Range: [-1, 1], higher is better
   - Values > 0.7 indicate good preservation

2. **Continuity**:
   - Fraction of k-nearest neighbors preserved in embedding
   - Range: [0, 1], higher is better
   - Measures local neighborhood preservation

3. **Mean Local Error**:
   - Average ranking error in neighborhood preservation
   - Lower values indicate better preservation
   - Sensitive to local structure distortions

## Conclusions

The analysis demonstrates that t-SNE can effectively visualize the Human Activity Recognition dataset:

- **Optimal Configuration**: Perplexity=100, Early Exaggeration=4
- **PCA Preprocessing**: Essential for computational efficiency, retains 87% variance
- **Class Separation**: Clear distinction between static and dynamic activities
- **Evaluation**: Quantitative metrics confirm visual observations

The results provide insights into human activity patterns and validate t-SNE's effectiveness for high-dimensional sensor data visualization.

## Future Work

- Experiment with other dimensionality reduction techniques (UMAP, Isomap)
- Investigate feature importance for activity classification
- Explore time-series analysis of activity transitions
- Apply clustering algorithms to discover sub-activities # Tsne_Human_Activity_Recognition
