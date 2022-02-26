import argparse
import json

import librosa
import torch

from config import Config
from speechset.utils.normalizer import TextNormalizer
from taco import Tacotron

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--ckpt')
parser.add_argument('--text')

args = parser.parse_args()

with open(args.config) as f:
    config = Config.load(json.load(f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(args.ckpt, map_location='cpu')

tts = Tacotron(config.model)
tts.load(ckpt)
tts.eval()
tts.to(device)

vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
vocoder.eval()
vocoder.to(device)

# [S]
label = torch.tensor(
    TextNormalizer().labeling(args.text), dtype=torch.long, device=device)
# S
textlen = torch.tensor(len(label), dtype=torch.long, device=device)

with torch.no_grad():
    # [1, T, M]
    mel, _, _ = tts(label[None], textlen[None])
    # [1, T x H]
    audio = vocoder.inference(mel.transpose(1, 2)) / 32768.
    # [T x H], mono-channel
    audio = audio.squeeze(dim=0).cpu().numpy()

librosa.output.write_wav('output.wav', audio, sr=22050)
