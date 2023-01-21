import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

cars196 = tfds.load('Cars196', as_supervised=True, shuffle_files=True)
cars_train = cars196['train']
cars_test = cars196['test']

values = [tensor[0].numpy() for tensor in cars_train]
df = pd.DataFrame(values, columns=['value'])
df.to_csv('test')

df = pd.read_csv('test')
Oron = 'homo'