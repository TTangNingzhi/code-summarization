import warnings
import random
import numpy as np
import torch
import nltk

warnings.filterwarnings("ignore")
language = 'python'

seed = 612
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# nltk.download('wordnet')


print("Torch version:", torch.__version__)
if torch.__version__ >= '2.0':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
else:
    device = 'cpu'
print("Using device:", device)
