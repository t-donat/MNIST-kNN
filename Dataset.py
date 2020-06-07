from IPython.display import Audio as ipy_Audio, display
from Helper import Helper

import numpy as np
from scipy import signal

import wave
import PIL

import warnings
import copy
import pathlib

import holoviews as hv

hv.notebook_extension('bokeh')


class Dataset:
    _default_plot_kwargs = {"width": 1000, "height": 400}

    def __init__(self, ds_name, names, features, labels, train_size=0.8, display_func=None, plot_kwargs=None):
        helper = Helper()
        helper.check_type(names, (tuple, list, np.ndarray), "Sample Names")
        helper.check_type(features, np.ndarray, "Features")
        helper.check_type(labels, np.ndarray, "Labels")

        helper.check_ndim(features, 2, "Features Matrix")
        helper.check_ndim(labels, 1, "Label Vector")

        for index, element in enumerate(names):
            helper.check_type(element, str, f"Array Element #{index}")

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
            helper.check_type(plot_kwargs, dict, "Plot Kwargs")
            self.set_plot_kwargs(**plot_kwargs)

    @property
    def ds_name(self):
        return self._ds_name

    @ds_name.setter
    def ds_name(self, new_name):
        helper = Helper()
        helper.check_type(new_name, str, "Dataset Name")
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
        helper = Helper()
        helper.check_type(new_size, float, "Train Size")
        helper.below_zero(new_size, "Train size")

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
        helper = Helper()

        if num_samples is None:
            num_samples = self.num_samples

        helper.check_type(method, str, "Method")
        helper.check_type(num_samples, int, "Number of samples")

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
        helper = Helper()
        if self._display_func is None:
            raise NotImplementedError("Warning: there is no display function available!")

        if isinstance(sample_indices, (int, np.integer)) and sample_indices in range(self.num_samples):
            return self._display_func(self._names[sample_indices], self._features[sample_indices],
                                      self._labels[sample_indices], **self._plot_kwargs)

        elif isinstance(sample_indices, (int, np.integer)):
            warnings.warn(f"Sample #{sample_indices} couldnot be displayed!")
            return

        helper.check_type(sample_indices, (tuple, list, np.ndarray), "Sample interable")

        for index, sample_number in enumerate(sample_indices):
            helper.check_type(sample_number, (int, np.integer), f"Element #{index}")

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
        helper = Helper()
        if num_samples is not None:

            if num_samples not in range(1, self.num_samples + 1):
                raise ValueError("The number of samples is either too high or too low!")

            else:
                helper.check_type(num_samples, int, "Number of samples")

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
