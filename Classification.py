from Helper import Helper

import numpy as np
import pandas as pd
from scipy import signal

import holoviews as hv
hv.notebook_extension('bokeh')


class Classification:

    @staticmethod
    def classify_nn(train_features, train_labels, test_features, k=1):

        helper = Helper()
        helper.check_type(train_features, np.ndarray, "Train Features")
        helper.check_type(train_labels, np.ndarray, "Train Labels")
        helper.check_type(test_features, np.ndarray, "Test Features")
        helper.check_type(k, int, "k")

        # checking if arrays are correct
        helper.check_ndim(train_features, 2, "Train Features")
        helper.check_ndim(test_features, 2, "Test Features")

        num_labels = len(train_labels)
        label_shape = train_labels.shape
        num_train_features, train_feature_len = train_features.shape
        num_test_features, test_feature_len = test_features.shape

        if label_shape != (num_labels,):
            raise ValueError(f"Train Labels: Expected array shape ({num_labels},) but got {label_shape}.")

        if train_feature_len != test_feature_len:
            raise ValueError("Train and Test Features: The sample length is not consistent over the entire dataset!")

        if num_train_features != num_labels:
            raise ValueError(f"Train Features and Labels: Unequal ammount of samples and labels!")

        if k <= 0:
            raise ValueError("The value of k cannot be smaller or equal to 0!")

        # casting the arrays as float64 to avoid overflow during the later calculations
        train_features = train_features.astype(np.float64)
        test_features = test_features.astype(np.float64)

        train_normed = train_features / np.linalg.norm(train_features, axis=1)[:, np.newaxis]
        test_normed = test_features / np.linalg.norm(test_features, axis=1)[:, np.newaxis]

        cosine_distances = 1 - np.einsum("ik,jk->ij", test_normed, train_normed)

        sample_labels = []
        for distances in cosine_distances:

            closest_labels = train_labels[np.argsort(distances)[:k]]
            labels, counts = np.unique(closest_labels, return_counts=True)

            most_common = labels[np.where(counts == max(counts))]

            # In case there are multiple most common labels, I choose the most common label with the lowest
            # cosine distance
            for label in closest_labels:
                if label in most_common:
                    chosen_label = label
                    break

            sample_labels.append(chosen_label)

        sample_labels = np.array(sample_labels)

        return sample_labels

    @staticmethod
    def accuracy_scores(actual, predicted):
        helper = Helper()
        helper.check_type(actual, np.ndarray, "Actual Labels")
        helper.check_type(predicted, np.ndarray, "Predicted Labels")

        helper.check_ndim(actual, 1, "Actual Labels")
        helper.check_ndim(predicted, 1, "Predicted Labels")

        if actual.shape != predicted.shape:
            raise ValueError("The label arrays do not have the same shape!")

        labels = pd.DataFrame({"Actual": actual, "Predicted": predicted})
        labels["Correct"] = labels["Actual"] == labels["Predicted"]

        global_accuracy = labels["Correct"].mean()
        label_accuracy = labels.groupby(["Actual"])["Correct"].mean().to_numpy()

        return global_accuracy, label_accuracy

    @staticmethod
    def confusion_matrix(actual, predicted):
        helper = Helper()
        helper.check_type(actual, np.ndarray, "True Labels")
        helper.check_type(predicted, np.ndarray, "Predicted Labels")

        helper.check_ndim(actual, 1, "Actual Labels")
        helper.check_ndim(predicted, 1, "Predicted Labels")

        if actual.shape != predicted.shape:
            raise ValueError("The label arrays do not have the same shape!")

        # how many classes are there
        highest_label = max(np.unique(actual)) + 1

        confusion = np.zeros((highest_label, highest_label))

        # print(confusion)

        for act, pred in zip(actual, predicted):
            confusion[act][pred] += 1

        confusion = np.array(confusion, dtype="int")
        return confusion

    @staticmethod
    def pca_transform(train_features, test_features, num_dims=20):
        helper = Helper()
        helper.check_type(train_features, np.ndarray, "Train Features Matrix")
        helper.check_type(test_features, np.ndarray, "Test Features Matrix")
        helper.check_type(num_dims, int, "Number of Feature Vectors")

        helper.check_ndim(train_features, 2, "Train Features")
        helper.check_ndim(test_features, 2, "Test Features")

        num_train_features, train_feature_len = train_features.shape
        num_test_features, test_feature_len = test_features.shape

        if train_feature_len != test_feature_len:
            raise ValueError("The Train and Test Features do not have the same sample length!")

        if num_dims <= 20:
            raise ValueError("The number of dimensions chosen cannot be lower than 1!")

        train_centered = train_features - np.mean(train_features, axis=0)
        test_centered = test_features - np.mean(train_features, axis=0)

        covariance_matrix = np.cov(train_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        principal_components = eigenvectors[:, np.argsort(eigenvalues)[-num_dims:]]

        train_transformed = np.real(np.einsum("ij,jk->ik", train_centered, principal_components))
        test_transformed = np.real(np.einsum("ij,jk->ik", test_centered, principal_components))

        return train_transformed, test_transformed

    @staticmethod
    def dft_transform(train_features, test_features):
        helper = Helper()
        helper.check_type(train_features, np.ndarray, "Train Features")
        helper.check_type(test_features, np.ndarray, "Test Features")

        helper.check_ndim(train_features, 2, "Train Features")
        helper.check_ndim(test_features, 2, "Test Features")

        num_train_features, train_feature_len = train_features.shape
        num_test_features, test_feature_len = test_features.shape

        if train_feature_len != test_feature_len:
            raise ValueError("The Train and Test Features do not have the same sample length!")

        train_dft = abs(np.fft.fft(train_features, axis=1))
        test_dft = abs(np.fft.fft(test_features, axis=1))

        num_train_samples, len_train_freq = train_dft.shape
        num_test_samples, len_test_freq = test_dft.shape

        train_dft = train_dft[:, :len_train_freq // 2] / len_train_freq
        test_dft = test_dft[:, :len_test_freq // 2] / len_test_freq

        return train_dft, test_dft

    @staticmethod
    def variance_analysis(features, labels):
        helper = Helper()
        helper.check_type(features, np.ndarray, "Features Matrix")
        helper.check_type(labels, np.ndarray, "Labels Vector")

        helper.check_ndim(features, 2, "Features Matrix")
        helper.check_ndim(labels, 1, "Label Vector")

        if len(features) != len(labels):
            raise ValueError("Sample amount in the Features Matrix and length of the Labels Vector do not add up!")

        unique_labels = set(labels)

        label_variances = []
        for current_label in unique_labels:
            label_features = features[np.where(labels == current_label)[0]]
            label_variance = np.var(label_features, axis=0)

            variance_maximum = label_variance.max()
            label_variance *= 255.0 / variance_maximum

            # To avoid hardcoding, Itake the first sample, get its length and then calculate the sqrt of that to get
            # the height/width of the image,cast it as an int for good measure
            image_size = int(np.sqrt(len(label_features[0])))

            variance_image = hv.Image(label_variance.reshape((image_size, image_size))).opts(tools=["hover"],
                                                                                             title=f"{current_label}",
                                                                                             xaxis=None, yaxis=None)
            label_variances.append(variance_image)

        all_variances = hv.Layout(tuple(label_variances))

        return all_variances

    @staticmethod
    def stft_analysis(features, labels, fs=8000, nperseg=100):
        helper = Helper()
        helper.check_type(features, np.ndarray, "Features Matrix")
        helper.check_type(labels, np.ndarray, "Labels Vector")
        helper.check_type(fs, int, "Sampling Frequency")
        helper.check_type(nperseg, int, "STFT Segment length")

        helper.check_ndim(features, 2, "Features Matrix")
        helper.check_ndim(labels, 1, "Label Vector")

        if len(features) != len(labels):
            raise ValueError("The feature matrix and the labels vector do not have the same length!")

        if fs <= 0:
            raise ValueError("Sampling frequency: Please enter a valid value larger than 0!")

        if nperseg <= 0:
            raise ValueError("Segment length: Please enter a valid value larger than 0!")

        unique_labels = np.unique(labels)

        layouts = []
        for current_label in unique_labels:
            label_features = features[np.where(labels == current_label)[0]]
            feature_mean = np.mean(label_features, axis=0)

            f, t, zxx = signal.stft(feature_mean, fs=fs, nperseg=nperseg)
            stft_plot = hv.QuadMesh((t, f, np.log10(np.abs(zxx)))).opts(title=f"{current_label}", cmap="Jet",
                                                                        ylabel="Frequencies in Hz", xlabel="Time in s",
                                                                        tools=["hover"])

            layouts.append(stft_plot)

        final_layout = hv.Layout(tuple(layouts))

        return final_layout
