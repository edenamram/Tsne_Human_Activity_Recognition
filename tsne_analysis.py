#!/usr/bin/env python3
"""
t-SNE Analysis of Human Activity Recognition Dataset

This script implements a comprehensive t-SNE analysis pipeline following the research methodology:

1. Dataset Selection: Human Activity Recognition dataset (~7K samples)
2. Hyperparameter Testing: perplexity=[5,30,50,100], early_exaggeration=[4,12]
3. PCA Preprocessing: Dimensionality reduction to 50 components
4. t-SNE Implementation: 2D embedding with various parameters
5. Quantitative Evaluation: Continuity, Mean Local Error, Shepard Correlations

Author: AI Assistant
Date: 2024
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TSNEAnalysis:
    def __init__(self, data_path):
        """Initialize the t-SNE analysis with dataset path"""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_pca = None
        self.results = {}
        
    def load_and_explore_data(self):
        """Load data and perform EDA"""
        print("="*50)
        print("LOADING AND EXPLORING DATA")
        print("="*50)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Dataset size: {self.data.size:,} total values")
        
        # Basic info
        print("\nFirst few rows:")
        print(self.data.head())
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nColumn names:")
        print(self.data.columns.tolist())
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print(f"\nMissing values: {missing_values.sum()}")
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        return self.data
    
    def prepare_data(self):
        """Prepare data for analysis - separate features and target"""
        print("\n" + "="*50)
        print("PREPARING DATA")
        print("="*50)
        
        # Assuming the last column is the target (common in many datasets)
        # Let's identify categorical columns first
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        print(f"Categorical columns: {list(categorical_cols)}")
        print(f"Numerical columns: {list(numerical_cols)}")
        
        # If there are categorical columns, assume the last one is target
        if len(categorical_cols) > 0:
            target_col = categorical_cols[-1]
            feature_cols = [col for col in self.data.columns if col != target_col]
        else:
            # If all numerical, assume last column is target
            target_col = self.data.columns[-1]
            feature_cols = self.data.columns[:-1].tolist()
        
        print(f"Target column: {target_col}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Separate features and target
        self.X = self.data[feature_cols]
        self.y = self.data[target_col]
        
        # Encode target if categorical
        if self.y.dtype == 'object':
            le = LabelEncoder()
            self.y_encoded = le.fit_transform(self.y)
            self.label_names = le.classes_
            print(f"Target classes: {self.label_names}")
            print(f"Class distribution:\n{pd.Series(self.y).value_counts()}")
        else:
            self.y_encoded = self.y
            self.label_names = None
            
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target vector shape: {self.y_encoded.shape}")
        
        return self.X, self.y_encoded
    
    def _count_high_correlations(self):
        """Count highly correlated feature pairs"""
        if self.X.shape[1] > 100:  # Sample for large datasets
            sample_X = self.X.sample(n=100, axis=1)
        else:
            sample_X = self.X
            
        corr_matrix = sample_X.corr()
        high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix.abs() < 1.0)
        return high_corr.sum().sum() // 2  # Divide by 2 to avoid double counting
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Create figure for EDA plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Exploratory Data Analysis', fontsize=16)
        
        # 1. Target distribution
        if self.label_names is not None:
            labels = self.label_names
            counts = pd.Series(self.y).value_counts()
        else:
            labels = pd.Series(self.y_encoded).value_counts().index
            counts = pd.Series(self.y_encoded).value_counts()
            
        axes[0,0].pie(counts.values, labels=labels, autopct='%1.1f%%')
        axes[0,0].set_title('Target Distribution')
        
        # 2. Feature correlation heatmap (sample of features if too many)
        if self.X.shape[1] > 20:
            sample_features = self.X.iloc[:, :20]
            axes[0,1].set_title('Correlation Heatmap (First 20 features)')
        else:
            sample_features = self.X
            axes[0,1].set_title('Feature Correlation Heatmap')
            
        corr_matrix = sample_features.corr()
        sns.heatmap(corr_matrix, ax=axes[0,1], cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        
        # 3. Feature statistics
        feature_stats = pd.DataFrame({
            'mean': self.X.mean(),
            'std': self.X.std(),
            'min': self.X.min(),
            'max': self.X.max()
        }).head(20)  # Show first 20 features
        
        axes[1,0].bar(range(len(feature_stats)), feature_stats['std'])
        axes[1,0].set_title('Feature Standard Deviations (First 20)')
        axes[1,0].set_xlabel('Feature Index')
        axes[1,0].set_ylabel('Standard Deviation')
        
        # 4. Data quality summary
        quality_info = [
            f"Total samples: {self.X.shape[0]:,}",
            f"Total features: {self.X.shape[1]:,}",
            f"Missing values: {self.data.isnull().sum().sum()}",
            f"High correlations: {self._count_high_correlations()}",
            f"Target classes: {len(np.unique(self.y_encoded))}"
        ]
        
        axes[1,1].text(0.1, 0.9, '\n'.join(quality_info), transform=axes[1,1].transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1,1].set_title('Data Quality Summary')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ EDA completed and saved as 'eda_plots.png'")
        return fig
    
    def normalize_data(self):
        """Normalize the feature data"""
        print("\n" + "="*50)
        print("NORMALIZING DATA")
        print("="*50)
        
        # Standardize features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        print(f"Features normalized using StandardScaler")
        print(f"Original data range: [{self.X.min().min():.3f}, {self.X.max().max():.3f}]")
        print(f"Scaled data range: [{self.X_scaled.min():.3f}, {self.X_scaled.max():.3f}]")
        print(f"Scaled data mean: {self.X_scaled.mean():.6f}")
        print(f"Scaled data std: {self.X_scaled.std():.6f}")
        
        return self.X_scaled
    
    def apply_pca(self, n_components=50):
        """Apply PCA for dimensionality reduction"""
        print("\n" + "="*50)
        print("APPLYING PCA")
        print("="*50)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        self.X_pca = pca.fit_transform(self.X_scaled)
        
        # Calculate variance explained
        variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        print(f"PCA applied: {self.X_scaled.shape} → {self.X_pca.shape}")
        print(f"Variance explained by {n_components} components: {cumulative_variance[-1]:.4f}")
        print(f"Variance explained by first 10 components: {cumulative_variance[9]:.4f}")
        
        # Create PCA analysis plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot explained variance
        axes[0].plot(range(1, len(variance_ratio) + 1), variance_ratio, 'bo-', markersize=4)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Individual Explained Variance')
        axes[0].grid(True)
        
        # Plot cumulative variance
        axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', markersize=4)
        axes[1].axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='95% threshold')
        axes[1].axhline(y=cumulative_variance[-1], color='g', linestyle='--', alpha=0.7, 
                       label=f'{n_components} components: {cumulative_variance[-1]:.3f}')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store PCA results
        self.pca_results = {
            'n_components': n_components,
            'variance_explained': cumulative_variance[-1],
            'individual_variance': variance_ratio,
            'cumulative_variance': cumulative_variance
        }
        
        print(f"✓ PCA analysis completed and saved as 'pca_analysis.png'")
        return self.X_pca
    
    def run_tsne_experiments(self):
        """Run t-SNE with different hyperparameters"""
        print("\n" + "="*50)
        print("RUNNING t-SNE EXPERIMENTS")
        print("="*50)
        
        # Define hyperparameters
        perplexities = [5, 30, 50, 100]
        early_exaggerations = [4, 12]
        
        # Store results
        self.tsne_results = {}
        
        # Create subplot grid
        fig, axes = plt.subplots(len(early_exaggerations), len(perplexities), 
                                figsize=(20, 10))
        
        experiment_count = 0
        total_experiments = len(perplexities) * len(early_exaggerations)
        
        for i, early_exag in enumerate(early_exaggerations):
            for j, perp in enumerate(perplexities):
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}: "
                      f"perplexity={perp}, early_exaggeration={early_exag}")
                
                # Run t-SNE
                tsne = TSNE(
                    n_components=2,
                    perplexity=perp,
                    early_exaggeration=early_exag,
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
                
                X_tsne = tsne.fit_transform(self.X_pca)
                
                # Store results
                experiment_key = f"perp_{perp}_exag_{early_exag}"
                self.tsne_results[experiment_key] = {
                    'embedding': X_tsne,
                    'perplexity': perp,
                    'early_exaggeration': early_exag,
                    'kl_divergence': tsne.kl_divergence_,
                    'n_iter': tsne.n_iter_
                }
                
                # Plot results
                scatter = axes[i, j].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                           c=self.y_encoded, cmap='tab10', 
                                           alpha=0.7, s=1)
                axes[i, j].set_title(f'Perp={perp}, EarlyExag={early_exag}\n'
                                    f'KL={tsne.kl_divergence_:.4f}')
                axes[i, j].set_xlabel('t-SNE 1')
                axes[i, j].set_ylabel('t-SNE 2')
                
                print(f"  KL divergence: {tsne.kl_divergence_:.4f}")
                print(f"  Iterations: {tsne.n_iter_}")
        
        plt.tight_layout()
        plt.savefig('tsne_experiments.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ t-SNE experiments completed and saved as 'tsne_experiments.png'")
        print(f"✓ Total experiments: {len(self.tsne_results)}")
        
        return self.tsne_results
    
    def _calculate_continuity(self, X_high, X_low, k=7):
        """Calculate continuity metric"""
        # High-dimensional neighbors
        high_dist = pairwise_distances(X_high)
        high_neighbors = np.argsort(high_dist, axis=1)[:, 1:k+1]
        
        # Low-dimensional neighbors  
        low_dist = pairwise_distances(X_low)
        low_neighbors = np.argsort(low_dist, axis=1)[:, 1:k+1]
        
        # Calculate continuity
        continuity_sum = 0
        n = X_high.shape[0]
        
        for i in range(n):
            # Find intersection of neighbor sets
            intersection = np.intersect1d(high_neighbors[i], low_neighbors[i])
            continuity_sum += len(intersection) / k
            
        return continuity_sum / n
    
    def _calculate_local_error(self, X_high, X_low, k=7):
        """Calculate mean relative rank error"""
        high_dist = pairwise_distances(X_high)
        low_dist = pairwise_distances(X_low)
        
        n = X_high.shape[0]
        local_errors = []
        
        for i in range(n):
            # Get k nearest neighbors in high-dimensional space
            high_neighbors_idx = np.argsort(high_dist[i])[1:k+1]
            
            # Calculate rank errors
            rank_errors = []
            for neighbor_idx in high_neighbors_idx:
                # Rank in high-dimensional space
                high_rank = np.where(np.argsort(high_dist[i]) == neighbor_idx)[0][0]
                # Rank in low-dimensional space
                low_rank = np.where(np.argsort(low_dist[i]) == neighbor_idx)[0][0]
                
                # Relative rank error
                if high_rank > 0:
                    rank_error = abs(low_rank - high_rank) / high_rank
                    rank_errors.append(rank_error)
            
            if rank_errors:
                local_errors.append(np.mean(rank_errors))
        
        return np.mean(local_errors) * 100  # Convert to percentage
    
    def _calculate_shepard_correlation(self, X_high, X_low):
        """Calculate Shepard diagram correlation"""
        # Calculate pairwise distances
        high_dist = pdist(X_high)
        low_dist = pdist(X_low)
        
        # Calculate Spearman correlation
        correlation, p_value = spearmanr(high_dist, low_dist)
        return correlation
    
    def evaluate_tsne_results(self):
        """Evaluate t-SNE results using quality metrics"""
        print("\n" + "="*50)
        print("EVALUATING t-SNE RESULTS")
        print("="*50)
        
        evaluation_results = {}
        
        for exp_name, exp_data in self.tsne_results.items():
            print(f"\nEvaluating {exp_name}...")
            
            X_embed = exp_data['embedding']
            
            # Calculate metrics
            continuity = self._calculate_continuity(self.X_pca, X_embed)
            local_error = self._calculate_local_error(self.X_pca, X_embed)
            shepard_corr = self._calculate_shepard_correlation(self.X_pca, X_embed)
            
            evaluation_results[exp_name] = {
                'perplexity': exp_data['perplexity'],
                'early_exaggeration': exp_data['early_exaggeration'],
                'kl_divergence': exp_data['kl_divergence'],
                'continuity': continuity,
                'local_error': local_error,
                'shepard_correlation': shepard_corr
            }
            
            print(f"  Continuity: {continuity:.4f}")
            print(f"  Local Error: {local_error:.2f}%")
            print(f"  Shepard Correlation: {shepard_corr:.4f}")
            print(f"  KL Divergence: {exp_data['kl_divergence']:.4f}")
        
        # Create evaluation summary plot
        df_eval = pd.DataFrame(evaluation_results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('t-SNE Evaluation Metrics', fontsize=16)
        
        # Plot metrics
        metrics = ['continuity', 'local_error', 'shepard_correlation', 'kl_divergence']
        titles = ['Continuity (higher better)', 'Local Error % (lower better)', 
                 'Shepard Correlation (higher better)', 'KL Divergence (lower better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx//2, idx%2]
            bars = ax.bar(range(len(df_eval)), df_eval[metric])
            ax.set_title(title)
            ax.set_xticks(range(len(df_eval)))
            ax.set_xticklabels([f"P{row.perplexity}_E{row.early_exaggeration}" 
                               for _, row in df_eval.iterrows()], rotation=45)
            
            # Color bars based on performance
            if metric in ['continuity', 'shepard_correlation']:
                # Higher is better
                best_idx = df_eval[metric].idxmax()
            else:
                # Lower is better
                best_idx = df_eval[metric].idxmin()
            
            for i, bar in enumerate(bars):
                if df_eval.index[i] == best_idx:
                    bar.set_color('green')
                else:
                    bar.set_color('lightblue')
        
        plt.tight_layout()
        plt.savefig('evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store evaluation results
        self.evaluation_results = evaluation_results
        
        print(f"\n✓ Evaluation completed and saved as 'evaluation_summary.png'")
        return evaluation_results
    
    def generate_report(self):
        """Generate final analysis report"""
        print("\n" + "="*70)
        print("FINAL ANALYSIS REPORT")
        print("="*70)
        
        # Dataset summary
        print("\n📊 DATASET SUMMARY:")
        print(f"   • Dataset: {self.data_path}")
        print(f"   • Samples: {self.data.shape[0]:,}")
        print(f"   • Original Features: {self.X.shape[1]:,}")
        print(f"   • Target Classes: {len(np.unique(self.y_encoded))}")
        if self.label_names is not None:
            print(f"   • Class Names: {', '.join(self.label_names)}")
        
        # PCA summary
        print(f"\n🔄 PCA PREPROCESSING:")
        print(f"   • Dimensions: {self.X.shape[1]} → {self.X_pca.shape[1]}")
        print(f"   • Variance Retained: {self.pca_results['variance_explained']:.4f}")
        
        # t-SNE experiments summary
        print(f"\n🎯 t-SNE EXPERIMENTS:")
        print(f"   • Total Experiments: {len(self.tsne_results)}")
        print(f"   • Perplexities Tested: {sorted(set(r['perplexity'] for r in self.tsne_results.values()))}")
        print(f"   • Early Exaggerations: {sorted(set(r['early_exaggeration'] for r in self.tsne_results.values()))}")
        
        # Best results
        print(f"\n🏆 BEST RESULTS:")
        
        eval_df = pd.DataFrame(self.evaluation_results).T
        
        # Find best performer for each metric
        best_continuity = eval_df.loc[eval_df['continuity'].idxmax()]
        best_shepard = eval_df.loc[eval_df['shepard_correlation'].idxmax()]
        best_local = eval_df.loc[eval_df['local_error'].idxmin()]
        best_kl = eval_df.loc[eval_df['kl_divergence'].idxmin()]
        
        print(f"   • Best Continuity: P={best_continuity['perplexity']}, E={best_continuity['early_exaggeration']} "
              f"(Score: {best_continuity['continuity']:.4f})")
        print(f"   • Best Shepard Corr: P={best_shepard['perplexity']}, E={best_shepard['early_exaggeration']} "
              f"(Score: {best_shepard['shepard_correlation']:.4f})")
        print(f"   • Best Local Error: P={best_local['perplexity']}, E={best_local['early_exaggeration']} "
              f"(Score: {best_local['local_error']:.2f}%)")
        print(f"   • Best KL Divergence: P={best_kl['perplexity']}, E={best_kl['early_exaggeration']} "
              f"(Score: {best_kl['kl_divergence']:.4f})")
        
        # Overall recommendation
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Calculate overall score (normalize and weight metrics)
        eval_df_norm = eval_df.copy()
        eval_df_norm['continuity_norm'] = (eval_df['continuity'] - eval_df['continuity'].min()) / (eval_df['continuity'].max() - eval_df['continuity'].min())
        eval_df_norm['shepard_norm'] = (eval_df['shepard_correlation'] - eval_df['shepard_correlation'].min()) / (eval_df['shepard_correlation'].max() - eval_df['shepard_correlation'].min())
        eval_df_norm['local_norm'] = 1 - (eval_df['local_error'] - eval_df['local_error'].min()) / (eval_df['local_error'].max() - eval_df['local_error'].min())
        eval_df_norm['kl_norm'] = 1 - (eval_df['kl_divergence'] - eval_df['kl_divergence'].min()) / (eval_df['kl_divergence'].max() - eval_df['kl_divergence'].min())
        
        eval_df_norm['overall_score'] = (eval_df_norm['continuity_norm'] + eval_df_norm['shepard_norm'] + 
                                        eval_df_norm['local_norm'] + eval_df_norm['kl_norm']) / 4
        
        best_overall = eval_df.loc[eval_df_norm['overall_score'].idxmax()]
        
        print(f"   • Best Overall: Perplexity={best_overall['perplexity']}, Early Exaggeration={best_overall['early_exaggeration']}")
        print(f"     - Shepard Correlation: {best_overall['shepard_correlation']:.4f}")
        print(f"     - Continuity: {best_overall['continuity']:.4f}")  
        print(f"     - Local Error: {best_overall['local_error']:.2f}%")
        print(f"     - KL Divergence: {best_overall['kl_divergence']:.4f}")
        
        # Files generated
        print(f"\n📁 FILES GENERATED:")
        print(f"   • eda_plots.png - Exploratory data analysis")
        print(f"   • pca_analysis.png - PCA variance analysis")
        print(f"   • tsne_experiments.png - All t-SNE experiments")
        print(f"   • evaluation_summary.png - Quantitative evaluation")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
    
    def run_complete_analysis(self):
        """Run the complete t-SNE analysis pipeline"""
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Prepare data
        self.prepare_data()
        
        # Step 3: Perform EDA
        self.perform_eda()
        
        # Step 4: Normalize data
        self.normalize_data()
        
        # Step 5: Apply PCA
        self.apply_pca()
        
        # Step 6: Run t-SNE experiments
        self.run_tsne_experiments()
        
        # Step 7: Evaluate results
        self.evaluate_tsne_results()
        
        # Step 8: Generate final report
        self.generate_report()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED!")
        print("="*50)

def main():
    """Main function to run the analysis"""
    # Create analysis instance
    analysis = TSNEAnalysis('train.csv')
    
    # Run complete analysis
    analysis.run_complete_analysis()

if __name__ == "__main__":
    main() 