import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import horovod.tensorflow.keras as hvd
import argparse
import gc

from PET import PET
import utils
from omnifold import Classifier
from common import *
from termcolor import cprint
import h5py

hvd.init()

def get_model_function():
    return PET, 'softmax'


def Produce_OmniLearn_info(fin, fout, flags):

  dataloader = utils.DelphesDataLoader(fin, flags.batch, hvd.rank(), hvd.size())
  model_function, activation = get_model_function()

  model = model_function(num_feat         = dataloader.num_feat,
                         num_jet          = dataloader.num_jet,
                         num_classes      = dataloader.num_classes,
                         local            = flags.local,
                         num_layers       = flags.num_layers,
                         drop_probability = flags.drop_probability,
                         simple           = flags.simple,
                         layer_scale      = flags.layer_scale,
                         talking_head     = flags.talking_head,
                         mode             = flags.mode,
                         class_activation = activation)

  X, y = dataloader.make_eval_data()
  if flags.nid > 0:
      add_string = '_{}'.format(flags.nid)
  else:
      add_string = ''

  model.load_weights(os.path.join(flags.indir, 'checkpoints', 
                                  utils.get_model_name(flags,fine_tune=flags.fine_tune,add_string=add_string)))

  y = hvd.allgather(tf.constant(y)).numpy()
  pred = hvd.allgather(tf.constant(model.predict(X, verbose=hvd.rank()==0)[0])).numpy()
  class_token = hvd.allgather(tf.constant(model.predict(X, verbose=hvd.rank() == 0)[2])).numpy()
  pred = pred.reshape(dataloader.nEvent, dataloader.nMaxJet, pred.shape[-1])
  class_token = class_token.reshape(dataloader.nEvent, dataloader.nMaxJet, class_token.shape[-1])
  print(np.shape(class_token))
  print(np.shape(pred))

  os.system('rm {}'.format(fout))
  with h5py.File(fout, mode = 'w') as f:
    data_dict = {}
    h5fr = h5py.File(fin, mode = 'r')
    dataset_structure = find_dataset_name(h5fr, list(h5fr))
    for key_ in dataset_structure:
      data_dict[key_] = np.array(list(h5fr[key_]))
    h5fr.close()

    for iFeat in range(class_token.shape[-1]):
      data_dict["INPUTS/Source/OmniEmbed{}".format(iFeat)] = class_token[:,:,iFeat] 
    data_dict["INPUTS/Source/OmniBTag"] = pred[:,1]
    
    for key_ in data_dict:
      f.create_dataset(key_, data = data_dict[key_])

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description = "Produce extra information given from OmniLearn network")
  parser.add_argument("--indir", type = str, help = "input directory")
  parser.add_argument("--outdir", type = str, help = "output directory")
  parser.add_argument("--dataset", type = str, default = "Delphes", help = "dataset name")
  parser.add_argument("--batch", type=int, default=5000, help="Batch size") 
  parser.add_argument("--mode", type=str, default="classifier", help="Loss type to train the model")
  parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
  parser.add_argument("--nid", type=int, default=0, help="Training ID for multiple trainings")
  parser.add_argument("--local", action='store_true', help="Use local embedding")
  parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
  parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
  parser.add_argument("--simple", action='store_true', help="Use simplified head model") 
  parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
  parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
  config = parser.parse_args()


  for type_ in ['Test']:
    CheckDir(os.path.join(config.outdir, type_))
    for file_ in os.listdir(os.path.join(config.indir, type_)):
      if 'light' in file_: continue # TODO: Need to fix light later
      cprint('running {}'.format(os.path.join(config.indir, type_, file_)), 'green')
      Produce_OmniLearn_info(fin = os.path.join(config.indir, type_, file_), fout = os.path.join(config.outdir, type_, file_), flags = config)
