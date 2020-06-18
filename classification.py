import numpy as np
import pandas as pd
from scipy import signal

import wave
import PIL

import warnings
import copy
import pathlib

import holoviews as hv
from IPython.display import Audio as ipy_Audio, display
hv.notebook_extension('bokeh')


class Helper:

    @staticmethod
    def check_type(parameter, intended_type, name):
        """Checks if the input parameter is of the intended type. Raises a ValueError if this is not the case."""

        if intended_type == callable and not callable(parameter):
            error = str(type(parameter))[8:-2]
            raise TypeError(f"Parameter {name}: Expected callable, got {error} instead!")

        elif not isinstance(parameter, intended_type):
            error = str(type(parameter))[8:-2]

            if isinstance(intended_type, tuple):

                error = str(type(parameter))[8:-2]

                type_list = [str(item)[8:-2] for item in intended_type]
                displayed_types = " or ".join(type_list)

            else:
                displayed_types = str(intended_type)[8:-2]

            raise TypeError(f"Parameter {name}: Expected {displayed_types}, got {error} instead!")

    @staticmethod
    def check_ndim(array, dim, name):
        """Checks if the passed array has the appropriate dimension. Only works on the data type numpy.ndarray."""

        if array.ndim != dim:
            raise ValueError(f"{name}: Expected dim of {dim}, got {array.ndim} instead!")

    @staticmethod
    def below_zero(parameter, name):
        """Checks if the passed parameter is equal or lower to 0 and raised a ValueError if not"""

        if parameter <= 0:
            raise ValueError(f"{name}: please enter a valid inout greater than 0!")

    @staticmethod
    def verify_filename(filename, name, return_ext=False):
        """Verifies if the passed parameter is a valid, raises a ValueError if not. \n
        If 'return_ext' is set to true, the extension is returned including the dot in front of it.
        Else it returns None.
        """

        if "." not in filename:
            raise ValueError(f"{name}: Filename is not valid!")

        if return_ext:
            return "." + filename.split(".")[-1]

        else:
            return None


class Dataset(Helper):

    _default_plot_kwargs = {"width": 1000, "height": 400}

    def __init__(self, ds_name, names, features, labels, train_size=0.8, display_func=None, plot_kwargs=None):
        Helper.__init__(self)

        self.check_type(names, (tuple, list, np.ndarray), "Sample Names")
        self.check_type(features, np.ndarray, "Features")
        self.check_type(labels, np.ndarray, "Labels")

        self.check_ndim(features, 2, "Features Matrix")
        self.check_ndim(labels, 1, "Label Vector")

        for index, element in enumerate(names):
            self.check_type(element, str, f"Array Element #{index}")

        if any(len(names) != len(item) for item in [features, labels]):
            raise ValueError("The length of Sample Names, Sample Features and Labels are not the same!")

        self.ds_name = ds_name
        self._names = names
        self._features = features
        self._labels = labels
        self.train_size = train_size

        self._display_func = None
        self._plot_kwargs = copy.deepcopy(self.__class__._default_plot_kwargs)

        if display_func is not None:
            self.display_func = display_func

        if plot_kwargs is not None:
            self.check_type(plot_kwargs, dict, "Plot Kwargs")
            self.set_plot_kwargs(**plot_kwargs)

    @property
    def ds_name(self):
        return self._ds_name

    @ds_name.setter
    def ds_name(self, new_name):
        self.check_type(new_name, str, "Dataset Name")
        self._ds_name = new_name

    @property
    def names(self):
        return self._names

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def train_size(self):
        return self._train_size

    @train_size.setter
    def train_size(self, new_size):
        self.check_type(new_size, float, "Train Size")
        self.below_zero(new_size, "Train size")

        if new_size > 1:
            raise ValueError("The train size must smaller or equal to 1!")

        self._train_size = new_size

    @property
    def display_func(self):
        return self._display_func

    @display_func.setter
    def display_func(self, new_display):
        if not callable(new_display):
            raise TypeError("This display function is not callable!")
        self._display_func = new_display

    @property
    def plot_kwargs(self):
        return self._plot_kwargs

    @property
    def num_samples(self):
        return len(self.names)

    def set_plot_kwargs(self, **kwargs):
        self.plot_kwargs.update(**kwargs)

    def generate_sets(self, method="random", num_samples=None):
        """Returns a data set for classification: ((train_samples, train_labels), (test_samples, test_labels)) \n
        By default, 'num_samples' is None, in which case all samples a taken. Choose from three modes:
        - random: The samples a chosen randomly
        - equal: An equal amount of samples is chosen from every class of digits
        - stratified: The distribution of digits is upheld in the returned data set

        If the method restricts the amount of samples that can be chosen, an appropriate warning is raised.
        The samples are shuffled and then split by the ratio of the objects' train_size property, then returned."""

        if num_samples is None:
            num_samples = self.num_samples

        self.check_type(method, str, "Method")
        self.check_type(num_samples, int, "Number of samples")

        if method not in ["random", "stratified", "equal"]:
            raise ValueError("This method is not defined!")

        elif method == "random":
            sample_indices = np.random.choice(range(self.num_samples), replace=False, size=num_samples)

        else:
            unique_classes, class_counts = np.unique(self._labels, return_counts=True)
            num_classes = len(unique_classes)
            class_distribution = class_counts / self.num_samples
            sample_indices = []

            if method == "equal":

                if num_classes > num_samples:
                    raise ValueError("There are more unique classes than samples called for!")

                equal_ammount = num_samples / num_classes

                if all(class_counts >= equal_ammount):
                    num_samples_taken = int(equal_ammount)

                else:
                    num_samples_taken = min(class_counts)

                for sample_class in unique_classes:
                    fitting_indices = np.where(self._labels == sample_class)[0]
                    sample_indices.extend(np.random.choice(fitting_indices, replace=False,
                                                           size=num_samples_taken))

            # -->stratified
            else:
                num_samples_taken = class_distribution * num_samples

                # casts the numbers as ints, which is needed for later functions, and implicitly uses floor() on them
                num_samples_taken = np.array([int(number) for number in num_samples_taken])

                if 0 in num_samples_taken:
                    warnings.warn("Method 'stratified': Some classes have 0 samples!")

                for sample_class, sample_ammount in zip(unique_classes, num_samples_taken):
                    fitting_indices = np.where(self._labels == sample_class)[0]
                    sample_indices.extend(np.random.choice(fitting_indices, replace=False, size=sample_ammount))

            sample_indices = np.array(sample_indices)

        actual_len = len(sample_indices)

        if actual_len != num_samples:
            warnings.warn(f"Due to constraints {actual_len} samples were taken (requested ammount: {num_samples})")

        np.random.shuffle(sample_indices)

        last_train_index = int(np.ceil(actual_len * self.train_size))
        train_indices = sample_indices[:last_train_index]
        test_indices = sample_indices[last_train_index:]

        train_data = (self._features[train_indices], self._labels[train_indices])
        test_data = (self._features[test_indices], self._labels[test_indices])

        dataset = (train_data, test_data)

        return dataset

    def visualize(self, sample_indices):
        """Visualizes the samples with the passed indices using the objects display function."""

        if self._display_func is None:
            raise NotImplementedError("Warning: there is no display function available!")

        if isinstance(sample_indices, (int, np.integer)) and sample_indices in range(self.num_samples):
            return self._display_func(self._names[sample_indices], self._features[sample_indices],
                                      self._labels[sample_indices], **self._plot_kwargs)

        elif isinstance(sample_indices, (int, np.integer)):
            warnings.warn(f"Sample #{sample_indices} couldnot be displayed!")
            return

        self.check_type(sample_indices, (tuple, list, np.ndarray), "Sample interable")

        for index, sample_number in enumerate(sample_indices):
            self.check_type(sample_number, (int, np.integer), f"Element #{index}")

        layout = []

        for index in sample_indices:
            if index not in range(self.num_samples):
                warnings.warn(f"Sample #{index} could not be displayed!")
                continue

            else:
                layout.append(self._display_func(self._names[index], self._features[index],
                                                 self._labels[index], **self._plot_kwargs))
        return layout

    def table(self, num_samples=None):
        """Returns a table of size 'num_samples' containing sample names, labels and dtypes."""

        if num_samples is not None:

            if num_samples not in range(1, self.num_samples + 1):
                raise ValueError("The number of samples is either too high or too low!")

            else:
                self.check_type(num_samples, int, "Number of samples")

        key_names = ["name", "label", "dtype"]
        name_array = np.asarray(self._names[:num_samples])
        label_array = self._labels[:num_samples]
        dtype_array = np.array([str(feature.dtype) for feature in self._features[:num_samples]])

        table_data = {key: value for key, value in zip(key_names, [name_array, label_array, dtype_array])}

        table = hv.Table(table_data, key_names).opts(**self.plot_kwargs)

        display(table)

        return table

    def __repr__(self):
        return f"Info2 Dataset: {self.ds_name}"


def create_ds_audio(path):
    """Extracts all .wav files in the target directory into a Dataset object."""

    helper = Helper()
    helper.check_type(path, str, "File directory")

    if not pathlib.Path(path).is_dir():
        raise OSError("No such directory found!")

    audio_files = [element for element in pathlib.Path(path).iterdir() if element.suffix == ".wav"]

    names = np.array([element.name for element in audio_files])
    labels = np.array([int(item[0]) for item in names], dtype="O")

    frames = []
    framerate = None
    for element in audio_files:

        with wave.open(str(element), "r") as wave_file:
            raw_signal = wave_file.readframes(-1)
            current_framerate = wave_file.getframerate()

        if framerate is None:
            framerate = current_framerate

        if framerate != current_framerate:
            raise ValueError("The framerate is not consistent over the entire dataset!")

        signal_data = np.frombuffer(raw_signal, dtype=np.int16)

        new_length = int(framerate * 0.5)
        resampled_signal = signal.resample(signal_data, new_length)

        frames.append(resampled_signal)

    frames = np.array(frames)

    def audio_plotter(name, data, label, **kwargs):
        """Nested audio plotter function with is later passed into the Dataset object."""

        helper.check_type(name, str, "Sample name")
        helper.check_type(data, np.ndarray, "Sample data")
        helper.check_type(label, int, "Sample label")

        title = f"Sample: {name}, Label: {label}"

        signal_curve = hv.Curve((np.arange(0, 0.5, 0.5 / new_length), data)).opts(xlabel="time in s",
                                                                                  ylabel="amplitude",
                                                                                  shared_axes=False,
                                                                                  title=title,
                                                                                  **kwargs)

        absolute_values = np.abs(np.fft.fft(data))[:len(data) // 2] / len(data)
        frequencies = np.fft.fftfreq(len(data), 1 / framerate)[:len(data) // 2]
        frequency_curve = hv.Curve((frequencies, absolute_values)).opts(xlabel="frequencies in Hz",
                                                                        ylabel="amplitude",
                                                                        title="Frequency transform of the sample",
                                                                        **kwargs)

        layout = hv.Layout(signal_curve + frequency_curve).opts(shared_axes=False)

        display(layout)
        display(ipy_Audio(data=data, rate=framerate))

        return layout

    dataset = Dataset(f"Spoken audio samples of numbers taken from {path}", names, frames, labels,
                      display_func=audio_plotter)

    return dataset


def create_ds_image(path):
    """Extracts all .png files in the target directory into a Dataset object."""

    helper = Helper()
    helper.check_type(path, str, "Sample directory")

    if not pathlib.Path(path).is_dir():
        raise OSError("This directory does not exist!")

    image_files = [element for element in pathlib.Path(path).iterdir() if element.suffix == ".png"]

    names = np.array([sample.name for sample in image_files])
    labels = np.array([int(name[0]) for name in names], dtype="O")

    frames = []

    for element in image_files:
        image = PIL.Image.open(element).resize((28, 28))
        inverted_image = 255 - np.array(image.getdata(), dtype=np.uint8)

        frames.append(inverted_image)

    frames = np.array(frames)

    def img_plotter(name, data, label, **kwargs):
        """Nested image plotter function with is later passed into the Dataset object."""

        helper.check_type(name, str, "Sample name")
        helper.check_type(data, np.ndarray, "Sample data")
        helper.check_type(label, int, "Sample label")

        label_image = hv.Image(data.reshape((28, 28))).opts(cmap="Greys",
                                                            xaxis=None,
                                                            yaxis=None,
                                                            title=f"Sample: {name}, Label: {label}")

        histogram = hv.Histogram(np.histogram(data, bins=15)).opts(shared_axes=False,
                                                                   xlabel="color scales",
                                                                   ylabel="frequency",
                                                                   title="Histogram of the sample")

        layout = hv.Layout(label_image + histogram).opts(shared_axes=False, **kwargs)

        display(layout)

        return layout

    dataset = Dataset(f"Handwritten image samples of numbers taken from {path}", names, frames, labels,
                      display_func=img_plotter)

    return dataset


class Captcha(Helper):

    def __init__(self, loaded_handlers):
        Helper.__init__(self)

        self.check_type(loaded_handlers, list, "Loaded Handlers")

        self._loaded_handlers = {}

        for handler in loaded_handlers:
            self.load_handler(handler)

    @property
    def loaded_handlers(self):
        handler_keys = self._loaded_handlers.keys()
        # the set orders the keys alphabetically, then i cast it as tuple as required
        return tuple(sorted(handler_keys))

    def load_handler(self, handler):
        """Verifies a handler, if it does it is added to the 'loaded handlers' dict of the Captcha object.

        If the handler doesn't pass this, an appropriate warning is raised."""

        if not hasattr(handler, "implements"):
            warnings.warn("Handler implements is missing!")
            return

        if not hasattr(handler, "read"):
            warnings.warn("Handler read is missing!")
            return

        if not hasattr(handler, "write"):
            warnings.warn("Handler write is missing!")
            return

        self._loaded_handlers[handler.implements] = handler

    def read_captcha(self, filename, display_captcha=False):
        """Reads a Captcha and returns it in the form of an array for later classification. \n
        Uses duck typing to use the appropriate handler. Raises an NotImplementedError if no fitting handler can
         be found.
        Set display_captcha to True to display the individual Captchas."""

        self.check_type(filename, str, "File Name")
        self.check_type(display_captcha, bool, "display captcha")

        file_ext = self.verify_filename(filename, "Method Read Captcha", return_ext=True)

        if file_ext not in self._loaded_handlers:
            raise NotImplementedError(f"No handler for {file_ext} implemented!")

        return self._loaded_handlers[file_ext].read(filename, display_captcha=display_captcha)

    def write_captcha(self, sample, filename, display_captcha=False):
        """Writes a Captcha from the passed sample array and saves to 'filename'.
        Uses duck typing to use the appropriate handler. Raises an NotImplementedError if no fitting handler can
        be found.
        Set display_captcha to True to display the completed Captcha."""

        self.check_type(sample, np.ndarray, "Sample Data")
        self.check_type(filename, str, "File Name")
        self.check_type(display_captcha, bool, "display captcha")
        self.check_ndim(sample, 2, "write sample")

        file_ext = self.verify_filename(filename, "Method Write Captcha", return_ext=True)

        if file_ext not in self._loaded_handlers:
            raise NotImplementedError(f"No handler for {file_ext} implemented!")

        return self._loaded_handlers[file_ext].write(sample, filename, display_captcha=display_captcha)


class WavHandler(Helper):

    @property
    def implements(self):
        return ".wav"

    def read(self, filename, display_captcha=False):
        """Reads a .wav Captcha and returns it in the form of an array for later classification. \n
        Uses duck typing to use the appropriate handler. Raises an NotImplementedError if there is no fitting handler.
        Set display_captcha to True to display the individual audio samples.\n"""

        self.check_type(filename, str, "File Name")
        self.check_type(display_captcha, bool, "display captcha")
        self.verify_filename(filename, "Method Read")

        with wave.open(filename, "r") as wave_file:
            all_data = np.frombuffer(wave_file.readframes(-1), dtype=np.int16)
            frame_rate = wave_file.getframerate()

        individual_length = int(frame_rate / 2)
        data_length = len(all_data)

        num_samples = int(data_length // individual_length)

        split_data = np.array(np.split(all_data, num_samples))

        if display_captcha:
            for audio_sample in split_data:
                display(ipy_Audio(data=audio_sample, rate=frame_rate))

        return split_data

    def write(self, sample, filename, display_captcha=False):
        """Writes a .wav Captcha from the passed sample array and saves to 'filename'. \n
        Set display_captcha to True to display the stitched together audio track."""

        self.check_type(sample, np.ndarray, "Sample Data")
        self.check_type(filename, str, "File Name")
        self.check_type(display_captcha, bool, "display captcha")
        self.verify_filename(filename, "Method Read")

        full_data = np.concatenate(sample)

        with wave.open(filename, "w") as wave_file:
            # nchannels, sampwidth, framerate, nframes, comptype, compname
            wave_file.setparams((1, 2, 8000, len(full_data), "NONE", "not compressed"))
            wave_file.writeframes(full_data.astype(np.int16))

        if display_captcha:
            display(ipy_Audio(data=full_data, rate=8000))
        return


class PngHandler(Helper):

    @property
    def implements(self):
        return ".png"

    def read(self, filename, display_captcha=False):
        """Reads a .png Captcha and returns it in the form of an array for later classification. \n
        Uses duck typing to use the appropriate handler. Raises an NotImplementedError if there is no fitting handler.
        Set display_captcha to True to display the individual audio samples.\n"""

        self.check_type(filename, str, "File Name")
        self.check_type(display_captcha, bool, "display captcha")
        self.verify_filename(filename, "Method Read")

        image = PIL.Image.open(filename)

        # taking advantage of the knowledge
        num_digits = int(image.width / 28)

        digit_images = []
        digit_data = []
        for index in range(num_digits):
            current_digit = image.crop((28 * index, 0, 28 * (index + 1), 28))
            digit_images.append(current_digit)
            current_digit_data = 255 - np.array(current_digit.getdata(), dtype=np.uint8)

            digit_data.append(current_digit_data)

        if display_captcha:
            for image in digit_images:
                # hv Image too?
                display(image)

        return np.array(digit_data)

    def write(self, sample, filename, display_captcha=False):
        """Writes a .png Captcha from the passed sample array and saves to 'filename'. \n
        Set display_captcha to True to display the stitched together audio track."""

        self.check_type(sample, np.ndarray, "Sample Data")
        self.check_type(filename, str, "File Name")
        self.check_type(display_captcha, bool, "display captcha")
        self.verify_filename(filename, "Method Read")

        num_samples, sample_len = sample.shape
        single_len = int(np.sqrt(sample_len))

        new_width = single_len * num_samples
        # since it's a square picture:
        new_height = single_len
        patched_image = PIL.Image.new("1", (new_width, new_height))

        sample = 255 - sample

        for index, digit in enumerate(sample):
            current_digit = PIL.Image.fromarray(digit.reshape((single_len, single_len)).astype("uint8"))
            patched_image.paste(im=current_digit, box=(index * single_len, 0))

        patched_image.save(filename)

        if display_captcha:
            display(patched_image)

        return


class Classification:

    @staticmethod
    def classify_nn(train_features, train_labels, test_features, k=1):
        """kNN: Predicts the labels of the test features using the train data and cosine distance. \n
        Predictions for each sample are chosen from the most common label among k labels with the lowest cosine
        distances. By default, k=1. If there are multiple most common labels, the one with the lowest distance is
        chosen."""

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
        """Compares predicted labels to true labels and returns the accuracies. \n
        This function returns a tuple, the first being the accuracy of all samples"""

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
        """Compares predicted labels to true labels and returns the confusion matrix."""

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
        """Image only: Calculates the PCA-Transform of train and test samples to a dimension of 'num_dims'. \n
        The reduction in dimensions leads to shorter run times since less calculations have to take place.
        However, higher values for 'num_dims' mean less information is lost due to the transformation."""

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
        """Audio only: Calculates the DFT transform of a train and test sample matrix. \n
        This step is important because in order to properly classify audio samples, one has to look at the frequency
        domain. Trying to classify using the wave form of an audio signal has a very low accuracy."""

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
        """Image only: Calculates the variance of every single pixel over all samples of each individual digit. \n
        Dark pixels correlate with low variance where as bright pixels correlate with high variance."""

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
        """Audio only: Calculates the STFT of the mean of all samples of every individual sample. \n
        It provides frequency information over time, dark pixels correlate with high amplitudes (power)."""
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
