import numpy as np
from skimage import feature
from sklearn.decomposition import PCA


class FeatureExtractor:
    """
    Feature extraction class for handwritten digit recognition.

    This class provides methods to:
    - Extract Histogram of Oriented Gradients (HOG) features
    - Extract raw pixel intensity features
    - Extract Zernike moment shape descriptors
    - Combine all features
    - Apply PCA for dimensionality reduction
    """

    def __init__(self, image_shape=(28, 28)):
        """
        Initialize feature extractor.

        Parameters:
        -----------
        image_shape : tuple
            Shape of MNIST images (default: (28, 28))
        """
        self.image_shape = image_shape
        self.pca = None
        self.n_components = None

    # =========================================================================
    # HOG FEATURES EXTRACTION
    # =========================================================================
    def extract_hog_features(
        self,
        X_flat,
        image_shape=(28, 28),
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2)
    ):
        """
        Extract Histogram of Oriented Gradients (HOG) features.

        HOG captures the distribution of gradient orientations in local regions
        of an image. It's effective for detecting edges and shape features in
        handwritten digits.

        Parameters:
        -----------
        X_flat : numpy array, shape (n_samples, 784)
            Flattened image data

        image_shape : tuple, default (28, 28)
            Original image dimensions

        pixels_per_cell : tuple, default (4, 4)
            Size of each cell in pixels

        cells_per_block : tuple, default (2, 2)
            Number of cells per block for normalization

        Returns:
        --------
        hog_features : numpy array
            HOG feature vectors
        """

        hog_features = []

        print("Extracting HOG features...")

        for i, img_flat in enumerate(X_flat):

            # Reshape flattened vector back into 2D image
            img = img_flat.reshape(image_shape)

            # Extract HOG feature vector
            hog_feature = feature.hog(
                img,
                orientations=9,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,
                block_norm='L2-Hys'
            )

            hog_features.append(hog_feature)

            # Progress update
            if (i + 1) % 5000 == 0:
                print(f"  ✓ Processed {i + 1} images")

        # Convert to numpy array
        hog_features = np.array(hog_features)

        print("✓ HOG features extraction complete")
        print(f"  Shape: {hog_features.shape}")
        print(f"  Features per sample: {hog_features.shape[1]}")

        return hog_features

    # =========================================================================
    # RAW PIXEL INTENSITY FEATURES
    # =========================================================================
    def extract_pixel_intensity(self, X_flat):
        """
        Use raw pixel intensities as features.

        Parameters:
        -----------
        X_flat : numpy array
            Flattened pixel data

        Returns:
        --------
        pixel_features : numpy array
            Raw pixel intensity features
        """

        print("✓ Pixel intensity features extracted")
        print(f"  Shape: {X_flat.shape}")
        print(f"  Features per sample: {X_flat.shape[1]}")

        return X_flat

    # =========================================================================
    # ZERNIKE MOMENT FEATURES
    # =========================================================================
    def extract_zernike_moments(
        self,
        X_flat,
        image_shape=(28, 28),
        order=8
    ):
        """
        Extract simplified Zernike moment shape descriptors.

        Parameters:
        -----------
        X_flat : numpy array
            Flattened pixel data

        image_shape : tuple
            Original image dimensions

        order : int
            Maximum order of Zernike moments

        Returns:
        --------
        zernike_features : numpy array
            Zernike feature vectors
        """

        zernike_features = []

        print("Extracting Zernike moments...")

        # Precompute valid (n, m) indices
        moment_indices = [
            (n, m)
            for n in range(order + 1)
            for m in range(n + 1)
            if (n - m) % 2 == 0
        ]

        print(f"  Computing {len(moment_indices)} Zernike moments...")

        for i, img_flat in enumerate(X_flat):

            # Reshape to 2D image
            img = img_flat.reshape(image_shape)

            # Normalize image
            img_min = img.min()
            img_max = img.max()

            img_norm = (
                (img - img_min) /
                (img_max - img_min + 1e-8)
            )

            moments = []

            # Compute moments
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
                print(f"  ✓ Processed {i + 1} images")

        # Convert to numpy array
        zernike_features = np.array(zernike_features)

        print("✓ Zernike moments extraction complete")
        print(f"  Shape: {zernike_features.shape}")
        print(f"  Moments per sample: {zernike_features.shape[1]}")

        return zernike_features

    # =========================================================================
    # COMPUTE SINGLE ZERNIKE MOMENT
    # =========================================================================
    @staticmethod
    def _zernike_moment(image, n, m):
        """
        Compute a single Zernike moment.

        Parameters:
        -----------
        image : numpy array
            Normalized image

        n : int
            Radial order

        m : int
            Angular order

        Returns:
        --------
        moment : complex
            Zernike moment value
        """

        h, w = image.shape

        # Image center
        cy, cx = h / 2.0, w / 2.0

        # Maximum radius
        max_r = np.sqrt(h**2 + w**2) / 2.0

        # Coordinate grids
        y, x = np.ogrid[:h, :w]

        # Polar coordinates
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_r
        theta = np.arctan2(y - cy, x - cx)

        # Clamp radius
        r = np.minimum(r, 1.0)

        # Simplified Zernike basis
        radial_basis = (2 * r - 1) ** n
        angular_basis = np.cos(m * theta) + 1j * np.sin(m * theta)

        basis = radial_basis * angular_basis

        # Compute moment
        moment = np.sum(image * basis)

        return moment

    # =========================================================================
    # PCA DIMENSIONALITY REDUCTION
    # =========================================================================
    def apply_pca(
        self,
        X_features,
        n_components=100,
        fit=True
    ):
        """
        Apply PCA for dimensionality reduction.

        Parameters:
        -----------
        X_features : numpy array
            Combined feature matrix

        n_components : int
            Number of PCA components

        fit : bool
            Whether to fit PCA or only transform

        Returns:
        --------
        X_pca : numpy array
            Reduced feature matrix
        """

        if fit:

            # Ensure valid PCA components
            n_components = min(
                n_components,
                X_features.shape[0],
                X_features.shape[1]
            )

            self.pca = PCA(
                n_components=n_components,
                random_state=42
            )

            X_pca = self.pca.fit_transform(X_features)

            self.n_components = n_components

            explained_var = (
                self.pca.explained_variance_ratio_.sum()
            )

            print("✓ PCA fitted and applied")
            print(f"  Input shape: {X_features.shape}")
            print(f"  Output shape: {X_pca.shape}")
            print(f"  Components: {n_components}")
            print(
                f"  Explained variance: "
                f"{explained_var:.4f} ({explained_var * 100:.2f}%)"
            )

        else:

            if self.pca is None:
                raise ValueError(
                    "PCA not fitted. Set fit=True first."
                )

            X_pca = self.pca.transform(X_features)

        return X_pca

    # =========================================================================
    # COMBINE ALL FEATURES
    # =========================================================================
    def combine_features(
        self,
        hog_features,
        pixel_features,
        zernike_features,
        apply_pca=False,
        n_components=100,
        pca_object=None
    ):
        """
        Combine HOG, Pixel, and Zernike features.

        Parameters:
        -----------
        hog_features : numpy array
            HOG features

        pixel_features : numpy array
            Pixel intensity features

        zernike_features : numpy array
            Zernike features

        apply_pca : bool
            Whether to apply PCA

        n_components : int
            Number of PCA components

        pca_object : PCA object
            Pre-fitted PCA object

        Returns:
        --------
        combined_features : numpy array
            Final feature matrix
        """

        # Concatenate features
        combined = np.hstack([
            hog_features,
            pixel_features,
            zernike_features
        ])

        print("✓ Features combined")
        print(f"  HOG shape: {hog_features.shape}")
        print(f"  Pixel shape: {pixel_features.shape}")
        print(f"  Zernike shape: {zernike_features.shape}")
        print(f"  Combined shape: {combined.shape}")

        if apply_pca:

            if pca_object is not None:

                # Use existing PCA
                combined = pca_object.transform(combined)

                print("✓ Pre-fitted PCA applied")
                print(f"  Output shape: {combined.shape}")

            else:

                # Fit new PCA
                combined = self.apply_pca(
                    combined,
                    n_components=n_components,
                    fit=True
                )

        return combined

    # =========================================================================
    # GET PCA OBJECT
    # =========================================================================
    def get_pca_object(self):
        """
        Return fitted PCA object.
        """

        return self.pca

    # =========================================================================
    # GET PCA STATISTICS
    # =========================================================================
    def get_pca_stats(self):
        """
        Return PCA statistics.

        Returns:
        --------
        stats : dict
            PCA information
        """

        if self.pca is None:
            return None

        return {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'total_variance': self.pca.explained_variance_ratio_.sum(),
            'singular_values': self.pca.singular_values_
        }
        
  