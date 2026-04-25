"""
Feature Extraction Module: HOG, Pixel Intensity, Zernike Moments, and PCA
Authors: Anamika, Ardra, Anugopal
"""

import numpy as np
from skimage import feature
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extract HOG features, pixel intensity vectors,
    and Zernike moment descriptors from MNIST images.
    """

    def __init__(self, image_shape=(28, 28)):
        """
        Initialize feature extractor.

        Parameters:
        -----------
        image_shape : tuple
            Shape of MNIST images (28x28)
        """

        self.image_shape = image_shape
        self.pca = None
        self.n_components = None

    # =========================================================
    # HOG FEATURES
    # =========================================================
    def extract_hog_features(
        self,
        X_flat,
        image_shape=(28, 28),
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2)
    ):
        """
        Extract Histogram of Oriented Gradients (HOG) features.

        Parameters:
        -----------
        X_flat : numpy array
            Flattened images

        image_shape : tuple
            Original image dimensions

        pixels_per_cell : tuple
            Size of each HOG cell

        cells_per_block : tuple
            Number of cells per block

        Returns:
        --------
        hog_features : numpy array
        """

        hog_features = []

        print("Extracting HOG features...")

        for i, img_flat in enumerate(X_flat):

            # Reshape flattened vector into image
            img = img_flat.reshape(image_shape)

            # Extract HOG feature vector
            hog_feature = feature.hog(
                img,
                orientations=9,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,          # Faster for large datasets
                block_norm='L2-Hys'
            )

            hog_features.append(hog_feature)

            # Progress update
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1} images")

        hog_features = np.array(hog_features)

        print(f"✓ HOG features extracted")
        print(f"  Shape: {hog_features.shape}")

        return hog_features

    # =========================================================
    # RAW PIXEL FEATURES
    # =========================================================
    def extract_pixel_intensity(self, X_flat):
        """
        Use raw pixel intensities as features.

        Parameters:
        -----------
        X_flat : numpy array

        Returns:
        --------
        pixel_features : numpy array
        """

        print(f"✓ Pixel intensity features extracted")
        print(f"  Shape: {X_flat.shape}")

        return X_flat

    # =========================================================
    # ZERNIKE MOMENTS
    # =========================================================
    def extract_zernike_moments(
        self,
        X_flat,
        image_shape=(28, 28),
        order=8
    ):
        """
        Extract simplified Zernike moment descriptors.

        Parameters:
        -----------
        X_flat : numpy array

        image_shape : tuple

        order : int
            Order of Zernike moments

        Returns:
        --------
        zernike_features : numpy array
        """

        zernike_features = []

        print("Extracting Zernike moments...")

        # Precompute valid indices
        moment_indices = [
            (n, m)
            for n in range(order + 1)
            for m in range(n + 1)
            if (n - m) % 2 == 0
        ]

        for i, img_flat in enumerate(X_flat):

            img = img_flat.reshape(image_shape)

            # Normalize image to [0,1]
            img_norm = (
                (img - img.min()) /
                (img.max() - img.min() + 1e-8)
            )

            moments = []

            for n, m in moment_indices:
                moment = self._zernike_moment(
                    img_norm,
                    n,
                    m
                )

                moments.append(np.abs(moment))

            zernike_features.append(moments)

            # Progress update
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1} images")

        zernike_features = np.array(zernike_features)

        print(f"✓ Zernike features extracted")
        print(f"  Shape: {zernike_features.shape}")

        return zernike_features

    # =========================================================
    # SINGLE ZERNIKE MOMENT
    # =========================================================
    @staticmethod
    def _zernike_moment(image, n, m):
        """
        Compute simplified Zernike moment.
        """

        h, w = image.shape

        cy, cx = h / 2, w / 2

        max_r = np.sqrt(h**2 + w**2) / 2

        y, x = np.ogrid[:h, :w]

        r = np.sqrt((x - cx)**2 + (y - cy)**2) / max_r

        theta = np.arctan2(y - cy, x - cx)

        # Simplified basis function
        v = np.cos(m * theta) * (2 * r - 1)**n

        moment = np.sum(image * v)

        return moment

    # =========================================================
    # PCA DIMENSIONALITY REDUCTION
    # =========================================================
    def apply_pca(
        self,
        X_features,
        n_components=100,
        fit=True
    ):
        """
        Apply Principal Component Analysis (PCA).

        Parameters:
        -----------
        X_features : numpy array

        n_components : int

        fit : bool

        Returns:
        --------
        X_pca : numpy array
        """

        if fit:

            self.pca = PCA(
                n_components=n_components,
                random_state=42
            )

            X_pca = self.pca.fit_transform(X_features)

            self.n_components = n_components

            explained_var = (
                self.pca.explained_variance_ratio_.sum()
            )

            print(f"✓ PCA applied")
            print(f"  Components: {n_components}")
            print(
                f"  Explained Variance: "
                f"{explained_var:.4f} "
                f"({explained_var * 100:.2f}%)"
            )

        else:

            if self.pca is None:
                raise ValueError(
                    "PCA not fitted. Set fit=True first."
                )

            X_pca = self.pca.transform(X_features)

        return X_pca

    # =========================================================
    # COMBINE FEATURES
    # =========================================================
    def combine_features(
        self,
        hog_features,
        pixel_features,
        zernike_features,
        apply_pca=True,
        n_components=100
    ):
        """
        Combine all feature types into one vector.

        Parameters:
        -----------
        hog_features : numpy array

        pixel_features : numpy array

        zernike_features : numpy array

        apply_pca : bool

        n_components : int

        Returns:
        --------
        combined_features : numpy array
        """

        print("Combining feature vectors...")

        combined = np.hstack([
            hog_features,
            pixel_features,
            zernike_features
        ])

        print(f"✓ Combined feature shape: {combined.shape}")

        if apply_pca:

            combined = self.apply_pca(
                combined,
                n_components=n_components
            )

        return combined

