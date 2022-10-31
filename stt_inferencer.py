import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
from omegaconf import DictConfig
import numpy as np
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
import onnxruntime
from nemo.collections.asr.metrics import wer
from util import softmax, get_nemo_dataset
import tempfile
import os
import json
import torch
from nemo.utils import logging
logging.set_verbosity(logging.ERROR)
from pathlib import Path
import pyaudio
import wave


class QuartznetInferencer():
    def __init__(self, inference_file):

        self.stt_config_path = "configs/quartznet5x5.yaml"

        self.inference_file_location = inference_file

        yaml = YAML(typ='safe')
        with open(self.stt_config_path, encoding="utf-8") as f:
            self.params = yaml.load(f)

        self.load_onnx_model()
        self.inference()

    def load_onnx_model(self):

        model_to_load = "model/turkish_fine-tuned_model.onnx"

        # create preprocessor
        preprocessor_cfg = DictConfig(self.params['model']).preprocessor
        self.preprocessor = EncDecCTCModel.from_config_dict(preprocessor_cfg)

        # create onnx session with model
        self.sess = onnxruntime.InferenceSession(model_to_load)
        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name

        self.wer = wer.WER(
            vocabulary=self.params["labels"],
            batch_dim_index=0,
            use_cer=False,
            ctc_decode=True,
            dist_sync_on_step=True,
        )


    def inference(self):

        files = [self.inference_file_location]
        # prediction here is in the form ["some string"]
        prediction = self._onnx_prediction(files)
        print("STT prediction:{}".format(prediction))

        prediction = [[(-1.0, prediction[0])]]

        del files

        return prediction

    def _onnx_prediction(self, files, return_logits_only=False):
        """
        Makes a prediction using onnx model.
        :Return: list of strings if return_logits_only is False, else logits
        """

        # try:
        with tempfile.TemporaryDirectory() as dataloader_tmpdir:
            with open(os.path.join(dataloader_tmpdir, 'manifest.json'), 'w') as fp:
                for audio_file in files:
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                    fp.write(json.dumps(entry) + '\n')
            out_batch = []
            config = {'paths2audio_files': files, 'batch_size': 1, 'temp_dir': dataloader_tmpdir}
            temporary_datalayer = get_nemo_dataset(config, self.params["labels"], 16000)
            for test_batch in temporary_datalayer:
                out_batch.append(test_batch)

        processed_signal, processed_signal_length = self.preprocessor(input_signal=out_batch[0][0],
                                                                      length=out_batch[0][1], )
        processed_signal = processed_signal.cpu().numpy()

        # inference
        logits = self.sess.run([self.label_name], {self.input_name: processed_signal})
        if return_logits_only:
            return logits[0][0]

        probabilities = logits[0][0]
        a = np.array([np.argmax(x) for x in probabilities])
        a = np.expand_dims(a, 0)
        a = torch.from_numpy(a)
        prediction = self.wer.ctc_decoder_predictions_tensor(a)

        return prediction


inference_file = "test_files/test0.wav"

quartznet_inferencer = QuartznetInferencer(inference_file)
quartznet_inferencer