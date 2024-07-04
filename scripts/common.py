import os, sys
import awkward as ak
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import numpy as np
python_version = int(sys.version.split('.')[0])
from typing import Optional
import logging
import pathlib
log = logging.getLogger(__name__)
import yaml

def CheckDir(path):
  if not os.path.exists(path):
    os.system('mkdir -p {}'.format(path))

def read_json(fname):
  jsonfile = open(fname)
  if python_version == 2:
    return_ =  json.load(jsonfile, encoding='utf-8')
  else:
    return_ =  json.load(jsonfile)
  jsonfile.close()
  return return_

def store_json(dict_, fname):
  with open(fname, 'w') as json_file:
    json.dump(dict_, json_file, indent = 4)

def read_yaml(fname):
  with open(fname, 'r') as file:
    return_ = yaml.safe_load(file)
  return return_

def prepare_shell(shell_file, command, condor, FarmDir):

############################################################
# Func: prepare sh file and add it to the condor schedule. #
############################################################

  cwd = os.getcwd()
  with open(os.path.join(FarmDir, shell_file), 'w') as shell:
    shell.write('#!/bin/bash\n')
    shell.write('WORKDIR=%s\n'%os.getcwd())
    shell.write('cd ${WORKDIR}\n')
    shell.write(command)

  condor.write('cfgFile=%s\n'%shell_file)
  condor.write('queue 1\n')



########################
##  Obtain Histogram  ##
########################

def Get_hist(bins, array, scale = 1.0):
    hist_ = (
        hist.Hist.new
        .Reg(bins[0], bins[1], bins[2], name = 'var')
        .Weight()
        .fill(var = array, weight = ak.from_numpy(np.ones(len(array))*scale))
      )
    return hist_


def Get_multi_hist(bins, array, name_list):

  hist_ = (
      hist.Hist.new
      .StrCat(name_list, name="ordered")
      .Reg(bins[0], bins[1], bins[2], name = 'var')
      .Weight()
  )
  for index, name in enumerate(name_list):
    hist_.fill(ordered = name, var = array[:,index])

  return hist_

def Get_hist2D(binsX, binsY, arrayX, arrayY, scale = 1.0):
  hist_ = (
    hist.Hist.new
    .Reg(binsX[0], binsX[1], binsX[2], name = 'varX')
    .Reg(binsY[0], binsY[1], binsY[2], name = 'varY')
    .Weight()
    .fill(varX = arrayX, varY = arrayY, weight = ak.from_numpy(np.ones(len(arrayX))*scale))
  )
  return hist_ 


def Evaluate_Generated_Event(dataset):

  Count = dict()
  for dataset_ in dataset:
    count_ = 0
    for file_ in dataset[dataset_]:
      fin = ROOT.TFile(file_, "READ")
      count_ += fin.Get("nGeneratedEvent").GetBinContent(1)
      fin.Close()
    Count[dataset_] = count_
  return Count

######################
##  Draw Histogram  ##
######################

def Draw_hist(hist, title, outputdir, fname, comments, density = False, dataset = None):
  CheckDir(outputdir)
  fig, ax = plt.subplots()
  hist.plot1d(ax=ax)
  y_label = 'Density' if density else 'nEntry'
  if title in comments:
    plt.text(0.7, 0.85, 'eff: %.3f'%comments[title], transform=ax.transAxes)
  if dataset is not None:
    plt.text(0.7, 0.8, dataset, transform=ax.transAxes)
  plt.title(title)
  plt.xlabel(title)
  plt.ylabel(y_label)
  plt.savefig(os.path.join(outputdir, fname + '.png'))
  plt.savefig(os.path.join(outputdir, fname + '.pdf'))
  plt.close()

def plot_pull(hist, title, outputdir, fname, comments, density = False, dataset = None):

  CheckDir(outputdir)
  fig = plt.figure(figsize=(10,8))
  main_ax, sub_ax = hist.plot_pull("normal", 
    eb_ecolor="steelblue",
    eb_mfc="steelblue",
    eb_mec="steelblue",
    eb_fmt="o",
    eb_ms=6,
    eb_capsize=1,
    eb_capthick=2,
    eb_alpha=0.8,
    fp_c="hotpink",
    fp_ls="-",
    fp_lw=2,
    fp_alpha=0.8,
    bar_fc="royalblue",
    pp_num=3,
    pp_fc="royalblue",
    pp_alpha=0.618,
    pp_ec=None,
    ub_alpha=0.2,
    fit_fmt=r"{name} = {value:.3g} $\pm$ {error:.3g}",
    )
  plt.savefig(os.path.join(outputdir, fname + '.png'))
  plt.savefig(os.path.join(outputdir, fname + '.pdf'))
  plt.close()
  

def Draw_hist2D(hist, title, xlabel, ylabel, outputdir, fname):

  CheckDir(outputdir)
  fig, ax = plt.subplots()
  hist.plot2d(ax = ax)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(os.path.join(outputdir, fname + '.png'))
  plt.savefig(os.path.join(outputdir, fname + '.pdf'))
  plt.close()

def Draw_multi_hist(hist, title, outputdir, fname, comments, dataset=None, density = False):
  CheckDir(outputdir)
  fig, ax = plt.subplots()
  hist[:,:].plot1d(ax=ax, density=density)
  ax.legend(title="ordered")
  y_label = 'Density' if density else 'nEntry'
  if title in comments:
    plt.text(0.7, 0.85, 'eff: %.3f'%comments[title], transform=ax.transAxes)
  if dataset is not None:
    plt.text(0.7, 0.8, dataset, transform=ax.transAxes)
  plt.title(title)
  plt.xlabel(title)
  plt.ylabel(y_label)
  plt.savefig(os.path.join(outputdir, fname + '.png'))
  plt.savefig(os.path.join(outputdir, fname + '.pdf'))
  plt.close()

def Draw_comparison(dataset_dict, histo_name, outputdir, fname, density = False):
  CheckDir(outputdir)
  fig, ax = plt.subplots()
  y_label = 'Density' if density else 'nEntry'
  for dataset_ in dataset_dict:
    try:
      dataset_dict[dataset_]["Histogram"][histo_name].plot1d(ax=ax, label = dataset_, density = density)
    except:
      dataset_dict[dataset_]["Histogram"][histo_name].plot1d(ax=ax, label = dataset_, density = False)
      y_label = 'nEntry'
  plt.title(histo_name)
  plt.xlabel(histo_name)
  plt.ylabel(y_label)
  ax.legend(title="Sample")
  plt.savefig(os.path.join(outputdir, fname + '.png'))
  plt.savefig(os.path.join(outputdir, fname + '.pdf'))
  plt.close()

def fig_save_and_close(
    fig: mpl.figure.Figure, path: str, close_figure: bool
) -> None:
    """Saves a figure at a given location if path is provided and optionally closes it.

    Args:
        fig (matplotlib.figure.Figure): figure to save
        path (Optional[pathlib.Path]): path where figure should be saved, or None to not
            save it
        close_figure (bool): whether to close figure after saving
    """
    if path is not None:
        log.info(f"saving figure as {path}")
        fig.savefig(path)
    if close_figure:
        plt.close(fig)

def find_dataset_name(f, path):

  path_return = []
  All_Array   = True
  for path_ in path:
    if type(list(f[path_])[0]) is str:
      All_Array = False
      for sub_path_ in list(f[path_]):
        path_return.append(os.path.join(path_, sub_path_))
    else:
      path_return.append(path_)
  if not All_Array:
    return find_dataset_name(f, path_return)
  else:
    return path_return
