import base64
import pickle

import numpy as np
import torch
import torchvision
from ts.torch_handler.base_handler import BaseHandler
from NAVC import load_model_and_opt, get_model
import logging

from Translator import Translator


class Handler(BaseHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self.initialized = False

    def initialize(self):
        model_pt_path = './best.pth.tar'
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.map_location if torch.cuda.is_available() else self.map_location)
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

    def preprocess(self, requests):
        """ Preprocessing, based on processor defined for MMF model.
            """
        feats = []

        feats_i, feats_m = requests[0].get("feats_i"), requests[0].get("feats_m")
        feats_i, feats_m = np.frombuffer(feats_i, dtype=np.float64).reshape(26, 2048), np.frombuffer(feats_m, dtype=np.float64).reshape(26,2048)
        i_feats, m_feats = torch.FloatTensor(feats_i.copy()).unsqueeze(dim=0), torch.FloatTensor(feats_m.copy()).unsqueeze(dim=0)
        # video = io.BytesIO(data['data'])
        # video_tensor, audio_tensor, info = torchvision.io.read_video(video)
        print(m_feats.size())
        print(i_feats.size())
        feats.append(m_feats)
        feats.append(i_feats)
        return feats


    def inference(self, feats, *args, **kwargs):

        results = self.model.encode(feats=feats)
        encoder_outputs = results
        teacher_encoder_outputs = None
        translator = Translator(model=self.model, opt=self.opt, device=self.device, teacher_model=None,
                                dict_mapping=None)

        category = torch.zeros(1, 1).long().to(self.device)
        # labels = torch.randint(1, 1000, (1, 20)).to(device)
        all_hyp, all_scores = translator.translate_batch(encoder_outputs, category, tgt_tokens=None, tgt_vocab=self.idx_to_word,
                                                         teacher_encoder_outputs=teacher_encoder_outputs)
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




handle = Handler()
feats_m = np.random.normal(0, 1, (26, 2048))
feats_i = np.random.normal(0, 1, (26, 2048))
feats_m, feats_i = feats_m.tobytes(), feats_i.tobytes()

data = {
        'feats_i': feats_m,
        'feats_m': feats_i,
        }

handle.initialize()
feats = handle.preprocess([data])
category = torch.zeros(1, 1).long().to(handle.device)
labels = torch.randint(1, 1000, (1, 20)).to(handle.device)
caption = handle.inference(feats, category, labels)





