import os
import torch
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from utils.mlp_utilities import MLP

Tk().withdraw()
filename = askopenfilename(initialdir = '.')
print(f"Loading Model: {filename}")

model_loader = torch.load(filename)

model = MLP(input_size=model_loader['input_size'], num_layers=model_loader['num_layers'], layers_size=model_loader['layer_size'], output_size=model_loader['output_size'])
model.load_state_dict(model_loader['model_state_dict'])