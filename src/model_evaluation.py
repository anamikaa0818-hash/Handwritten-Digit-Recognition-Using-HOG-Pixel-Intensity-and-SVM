"""
Model Evaluation Module: Metrics, Confusion Matrix, Comparisons
Authors: Anamika, Ardra, Anugopal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pickle

class ModelEvaluator:
    """
    Evaluate and compare multiple trained models.
    """
    
    def __init__(self):
        self.results = {}
        self.confusion_matrices = {}
        self.per_digit_metrics = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Evaluate single model on test set.
        
        Parameters:
        -----------
        model : trained sklearn model
        X_test : numpy array
        y_test : numpy array
        model_name : str
        
        Returns:
        --------
        metrics_dict : dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-digit metrics
        cm = confusion_matrix(y_test, y_pred, labels=range(10))
        per_digit_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Store results
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'confusion_matrix': cm,
            'per_digit_accuracy': per_digit_acc
        }
        
        self.results[model_name] = metrics
        self.confusion_matrices[model_name] = cm
        self.per_digit_metrics[model_name] = per_digit_acc
        
        print(f"\n=== {model_name} ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, models_dict, X_test, y_test):
        """
        Evaluate multiple models.
        """
        print("=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)
        
        for model_name, model in models_dict.items():
            self.evaluate_model(model, X_test, y_test, model_name)
    
    def compare_models(self):
        """
        Create comparison table of all models.
        """
        results_list = []
        for model_name, metrics in self.results.items():
            results_list.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
        
        df_comparison = pd.DataFrame(results_list)
        df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def plot_confusion_matrices(self, figsize=(15, 10)):
        """
        Plot confusion matrices for all models.
        """
        n_models = len(self.confusion_matrices)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, cm) in enumerate(self.confusion_matrices.items()):
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, xticklabels=range(10), yticklabels=range(10))
            ax.set_title(f"{model_name}\nAccuracy: {self.results[model_name]['accuracy']:.4f}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        
        # Hide extra axes
        for idx in range(len(self.confusion_matrices), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('../data/confusion_matrices.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_per_digit_accuracy(self):
        """
        Plot per-digit accuracy for all models.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(10)
        width = 0.2
        
        for idx, (model_name, accuracies) in enumerate(self.per_digit_metrics.items()):
            offset = (idx - len(self.per_digit_metrics) / 2) * width
            ax.bar(x + offset, accuracies, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel("Digit Class")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Digit Accuracy Across Models")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('../data/per_digit_accuracy.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison_bar(self):
        """
        Plot model comparison as bar chart.
        """
        df_comparison = self.compare_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            colors = ['green' if i == df_comparison[metric].idxmax() else 'steelblue' 
                     for i in range(len(df_comparison))]
            ax.barh(df_comparison['Model'], df_comparison[metric], color=colors, alpha=0.8)
            ax.set_xlabel(metric)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, v in enumerate(df_comparison[metric]):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig('../data/model_comparison_bars.png', dpi=100, bbox_inches='tight')
        plt.show()
