"""
Data Loading and Preprocessing Module
Author: Anamika, Ardra, Anugopal
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataLoader:
    """
    Handles MNIST digit dataset loading, preprocessing, and splitting.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
    
    def load_data(self):
        """Load MNIST digits dataset from sklearn."""
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def normalize_pixels(self, X, method='standard'):
        """
        Normalize pixel values.
        
        Parameters:
        -----------
        X : numpy array
            Pixel data
        method : str
            'standard' (0 mean, 1 std) or 'minmax' (0-1 range)
        
        Returns:
        --------
        X_normalized : numpy array
        """
        if method == 'standard':
            X_normalized = self.scaler.fit_transform(X)
        elif method == 'minmax':
            X_normalized = X / X.max()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        print(f"✓ Pixels normalized using {method}")
        return X_normalized
    
    def train_test_split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets.
        
        Parameters:
        -----------
        X : numpy array
        y : numpy array
        test_size : float
        val_size : float
        
        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train + val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"✓ Data split: Train={self.X_train.shape[0]}, "
              f"Val={self.X_val.shape[0]}, Test={self.X_test.shape[0]}")
        
        return self.X_train, self.X_val, self.X_test, \
               self.y_train, self.y_val, self.y_test
    
    def prepare_data(self, normalize_method='standard'):
        """
        Complete pipeline: load, normalize, split.
        """
        X, y = self.load_data()
        X_normalized = self.normalize_pixels(X, method=normalize_method)
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.train_test_split_data(X_normalized, y)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
