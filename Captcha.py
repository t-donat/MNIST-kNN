from IPython.display import Audio as ipy_Audio, display
from Helper import Helper
import warnings

import numpy as np
import wave
import PIL

import holoviews as hv

hv.notebook_extension('bokeh')


class Captcha:

    def __init__(self, loaded_handlers):
        helper = Helper()
        helper.check_type(loaded_handlers, list, "Loaded Handlers")

        self._loaded_handlers = {}

        for handler in loaded_handlers:
            self.load_handler(handler)

    @property
    def loaded_handlers(self):
        handler_keys = self._loaded_handlers.keys()
        # the set orders the keys alphabetically, then i cast it as tuple as required
        return tuple(sorted(handler_keys))

    def load_handler(self, handler):

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
        helper = Helper()
        helper.check_type(filename, str, "File Name")
        helper.check_type(display_captcha, bool, "display captcha")

        if "." not in filename:
            raise ValueError("Method Read Captcha: Filename is not valid!")

        file_ext = "." + filename.split(".")[-1]

        if file_ext not in self._loaded_handlers:
            raise NotImplementedError(f"No handler for {file_ext} implemented!")

        return self._loaded_handlers[file_ext].read(filename, display_captcha=display_captcha)

    def write_captcha(self, sample, filename, display_captcha=False):
        helper = Helper()
        helper.check_type(sample, np.ndarray, "Sample Data")
        helper.check_type(filename, str, "File Name")
        helper.check_type(display_captcha, bool, "display captcha")

        if "." not in filename:
            raise ValueError("Method Write Captcha: Filename is not valid!")

        if sample.ndim != 2:
            raise ValueError("The sample does not have 2 dimensions")

        file_ext = "." + filename.split(".")[-1]

        if file_ext not in self._loaded_handlers:
            raise NotImplementedError(f"No handler for {file_ext} implemented!")

        return self._loaded_handlers[file_ext].write(sample, filename, display_captcha=display_captcha)


class WavHandler:

    @property
    def implements(self):
        return ".wav"

    def read(self, filename, display_captcha=False):
        helper = Helper()
        helper.check_type(filename, str, "File Name")
        helper.check_type(display_captcha, bool, "display captcha")

        if "." not in filename:
            raise ValueError("Method Read: Filename is not valid!")

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
        helper = Helper()
        helper.check_type(sample, np.ndarray, "Sample Data")
        helper.check_type(filename, str, "File Name")
        helper.check_type(display_captcha, bool, "display captcha")

        if "." not in filename:
            raise ValueError("Method Write: Filename is not valid!")

        full_data = np.concatenate(sample)

        with wave.open(filename, "w") as wave_file:
            # nchannels, sampwidth, framerate, nframes, comptype, compname
            wave_file.setparams((1, 2, 8000, len(full_data), "NONE", "not compressed"))
            wave_file.writeframes(full_data.astype(np.int16))

        if display_captcha:
            display(ipy_Audio(data=full_data, rate=8000))
        return


class PngHandler:

    @property
    def implements(self):
        return ".png"

    def read(self, filename, display_captcha=False):
        helper = Helper()
        helper.check_type(filename, str, "File Name")
        helper.check_type(display_captcha, bool, "display captcha")

        if "." not in filename:
            raise ValueError("Method Read: Filename is not valid!")

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
            for picture in digit_images:
                # hv Image too?
                display(picture)

        return np.array(digit_data)

    def write(self, sample, filename, display_captcha=False):
        helper = Helper()
        helper.check_type(sample, np.ndarray, "Sample Data")
        helper.check_type(filename, str, "File Name")
        helper.check_type(display_captcha, bool, "display captcha")

        if "." not in filename:
            raise ValueError("Method Write: Filename is not valid!")

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
