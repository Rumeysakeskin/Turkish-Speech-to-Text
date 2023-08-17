from ruamel.yaml import YAML
import numpy as np
import importlib
from jetson_voice_utils.trt_model import TRTModel
from util import int2float
import torch
import soxr
from typing import List


class QuartznetInferencer():
    def __init__(self):

        self.stt_config_path = "configs/quartznet15x5.yaml"
        self.rate = 16000
        DYNAMIC_SHAPES = {"min": (1, 64, 1), "max": (1, 64, 1024)}

        yaml = YAML(typ='safe')
        with open(self.stt_config_path, encoding="utf-8") as f:
            self.params = yaml.load(f)

        self.trt_model = TRTModel(self.params, DYNAMIC_SHAPES)
        self.preprocessor = self.create_preprocessor()

    def create_preprocessor(self):
        preprocessor_name = params['model']['preprocessor']['cls'].rsplit(".", 1)
        preprocessor_class = getattr(importlib.import_module(preprocessor_name[0]), preprocessor_name[1])
        preprocessor_config = params['model']['preprocessor'].copy()
        preprocessor_config.pop('cls')
        preprocessor_config = preprocessor_config["params"]
        preprocessor = preprocessor_class(**preprocessor_config)
        return preprocessor

    def ctc_decoder_predictions_tensor(self, predictions: torch.Tensor, lang) -> List[str]:
        """
        Decodes a sequence of labels to words
        """
        vocabulary = params["labels"]
        blank_id = len(vocabulary)
        labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        hypotheses = []
        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            # CTC decoding procedure
            decoded_prediction = []
            previous = blank_id
            for p in prediction:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
            hypotheses.append(hypothesis)
        return hypotheses

    def inference(self, stt_bytes):

        audio_int16 = np.fromstring(stt_bytes, dtype=np.int16)  # convert bytes to numpy array for stt prediction
        audio_data = int2float(audio_int16)  # audio_float32#
        if self.rate != 16000:
            audio_data = soxr.resample(audio_data, 48000, 16000, quality=soxr.VHQ)

        signal = np.expand_dims(audio_data, 0)  # add a batch dimension
        signal = torch.from_numpy(signal)  # converts the NumPy array to a PyTorch tensor

        processed_signal, _ = self.preprocessor(input_signal=signal, length=torch.tensor(audio_data.size).unsqueeze(0), )

        processed_signal = processed_signal.cpu().numpy()

        logits = self.trt_model.execute(processed_signal)

        probabilities = logits[0]
        a = np.array([np.argmax(x) for x in probabilities])
        a = np.expand_dims(a, 0)
        a = torch.from_numpy(a)

        prediction = self.ctc_decoder_predictions_tensor(a, lang)

        return prediction[0]


quartznet_inferencer = QuartznetInferencer()
quartznet_inferencer.inference(stt_bytes=stt_bytes)