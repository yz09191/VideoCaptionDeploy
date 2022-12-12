import base64
import io
import os
import pickle

import numpy as np
import torch
import torchvision
from ts.torch_handler.base_handler import BaseHandler
from NAVC import load_model_and_opt, get_model
import logging

logger = logging.getLogger(__name__)
__all__ = (
    'NAVCHandler',
)


class NAVCHandler(BaseHandler):
    def __init__(self):
        super(NAVCHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        data = pickle.load(open('info_corpus.pkl', 'rb'))
        info = data['info']
        self.idx_to_word = info['itow']

        checkpoint = torch.load(model_pt_path)
        self.opt = checkpoint['settings']
        self.model = get_model(self.opt)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        print("********* files in temp direcotry that .mar file got extracted *********", os.listdir(model_dir))

    def preprocess(self, requests):
        """ Preprocessing, based on processor defined for MMF model.
            """
        feats = []
        print(len(requests))
        print(requests[0].keys())
        feats_i, feats_m = requests[0].get("feats_i"), requests[0].get("feats_m")
        feats_i, feats_m = np.frombuffer(feats_i, dtype=np.float32), np.frombuffer(feats_m, dtype=np.float32)
        i_feats, m_feats = torch.FloatTensor(feats_i.copy()), torch.FloatTensor(feats_m.copy())
        # video = io.BytesIO(data['data'])
        # video_tensor, audio_tensor, info = torchvision.io.read_video(video)
        feats.append(i_feats)
        feats.append(m_feats)
        return feats


    def inference(self, feats, *args, **kwargs):
        if torch.cuda.is_available():
            with torch.cuda.device(feats.get_device()):
                results = self.model(feats)
        else:
            results = self.model(feats)

        all_hyp = results['tgt_word_logprobs']
        caption = ''
        # while isinstance(all_hyp, list):
        all_hyp = all_hyp[0][0]
        for i in all_hyp:
            if i == 3:
                break
            caption += self.idx_to_word[i] + ' '
        print("************** caption *********", caption)
        return caption


    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return [inference_output]







