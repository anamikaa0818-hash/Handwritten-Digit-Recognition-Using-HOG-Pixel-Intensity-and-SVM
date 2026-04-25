"""
Data Loading and Preprocessing Module
Author: Anamika, Ardra, Anugopal
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        """
        Load MNIST dataset (70,000 samples, 28x28 images)
        """

        print("Loading MNIST dataset...")

        mnist = fetch_openml('mnist_784', version=1, as_frame=False)

        X = mnist.data
        y = mnist.target.astype(np.int32)

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
            'standard' -> mean 0 std 1
            'minmax'   -> values between 0 and 1
        """

        if method == 'standard':
            X_normalized = self.scaler.fit_transform(X)

        elif method == 'minmax':
            X_normalized = X / 255.0

        else:
            raise ValueError("method must be 'standard' or 'minmax'")

        print(f"✓ Pixels normalized using {method}")

        return X_normalized

    def train_test_split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets.
        """

        # Train+Validation and Test split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Train and Validation split
        val_size_adjusted = val_size / (1 - test_size)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        print(
            f"✓ Data split:\n"
            f"   Train = {self.X_train.shape[0]}\n"
            f"   Validation = {self.X_val.shape[0]}\n"
            f"   Test = {self.X_test.shape[0]}"
        )

        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test
        )

    def prepare_data(self, normalize_method='standard'):
        """
        Complete pipeline:
        Load → Normalize → Split
        """

        X, y = self.load_data()

        X_normalized = self.normalize_pixels(
            X,
            method=normalize_method
        )

        return self.train_test_split_data(X_normalized, y)


# Example Usage
if __name__ == "__main__":

    loader = DataLoader()

    X_train, X_val, X_test, y_train, y_val, y_test = \
        loader.prepare_data(normalize_method='minmax')

    print("\nDataset Ready!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

