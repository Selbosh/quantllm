from modules.conjugate import NormalInverseGammaPrior, GammaExponentialPrior
import pandas as pd
from pathlib import Path

def main():
    data_dirpath = Path(__file__).parents[2] / 'data'