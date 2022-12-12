import base64
from io import BytesIO

import requests
import json
import torch
import numpy as np
import savc_handle
''' preprocessing steps to select a video
clip from the full video needs to be done
to make the video ready for request.
'''
feats_m = np.random.normal(0, 1, (1, 26, 2048))
feats_i = np.random.normal(0, 1, (1, 26, 2048))

feats_m, feats_i = feats_m.tobytes(), feats_i.tobytes()

data = {
        'feats_i': feats_m,
        'feats_m': feats_i,
        }

response = requests.post('http://114.212.87.252:3000/predictions/navc', data)
print(response.content)
# with open("response.txt", "wb") as response_handler:
#     response_handler.write(data)

