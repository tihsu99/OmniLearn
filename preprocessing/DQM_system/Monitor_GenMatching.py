import os, sys
import argparse
import h5py
import glob
import numpy as np
import awkward as ak
import uproot
import vector
import hist
import matplotlib.pyplot as plt
from util import Get_hist, Draw_hist, find_dataset_name
vector.register_awkward()
from process_info import *
import re

def Lorentz_vector(obj, pad_length = None):
  v4 = vector.zip(
    {
        "pt":  obj[:, :, 0],
        "eta": obj[:, :, 1],
        "phi": obj[:, :, 2],
        "mass": obj[:, :, 3],
    }
  )
  
  if pad_length is not None:
    v4 = ak.pad_none(v4, target=pad_length, axis=1, clip=True)

  return v4

def RecoMass(Object, diagram):

  v4 = dict()
  for obj in ["els", "jets", "mus"]:
    v4[obj] = Lorentz_vector(Object[obj], pad_length=10)
  
  # ---------------------------
  # Reconstructed:
  #   -1: initial state, don't need reconstruction
  #    0: mediate state, need reconstruction
  #    1: final state, already reconstructed
  # ----------------------------

  Reco_v4 = vector.zip(
    {
      "pt": ak.ones_like(Object["genpart"][:,:,8]) * -1,
      "eta": ak.ones_like(Object["genpart"][:,:,8]) * -1,
      "phi": ak.ones_like(Object["genpart"][:,:,8]) * -1,
      "mass": ak.ones_like(Object["genpart"][:,:,8]) * -1
     }
  )

  nMaxJet = ak.max(ak.num(Object["jets"]))
  print(nMaxJet)

  Matched_index = ak.values_astype(Object["genpart"][:,:,9], int) 
  Matched_index = ak.where((Matched_index > -1), Matched_index, 0) # The padded one will not be used in any case
  #Matched_index_for_jet = ak.where(Matched_index > ak.broadcast_arrays(ak.unflatten((ak.num(Object["jets"]) -1), counts =1 , axis = -1))[0], 0, Matched_index)
  PDG_ID        = ak.values_astype(Object["genpart"][:,:,7], int)
  Reco_v4       = ak.where((Matched_index > -1) & (abs(PDG_ID) < 6), v4["jets"][Matched_index], Reco_v4) 
  #Reco_v4       = ak.where((Matched_index > -1) & (abs(PDG_ID) == 11), v4["els"][Matched_index_for_jet], Reco_v4) 
  #Reco_v4       = ak.where((Matched_index > -1) & (abs(PDG_ID) == 13), v4["mus"][Matched_index_for_jet], Reco_v4) 


  parton = ak.zip(
     {
      "index": Object["genpart"][:,:,4],
      "pdgId": Object["genpart"][:,:,7],
      "M1":    Object["genpart"][:,:,5]
     }
  )


  Event_dict = dict()
  Candidate_dict = dict()
  for product in diagram['diagram']:
    if product == "SYMMETRY":continue
    if "SYMMETRY" in diagram['diagram']:
        symmetry = diagram['diagram']["SYMMETRY"] 
        symmetry_map = {sym_: idx for idx, sym_ in enumerate(symmetry)}
    else:
        symmetry_map = None
 
    candidate_array = select_by_pdgId(parton, product)    
    product_name = 'EVENT/{}'.format(product)
    candidate_array, Event_dict = select_by_products(parton, candidate_array, diagram['diagram'][product], product_name, Event_dict)
    if symmetry_map is not None:
        candidate_array = ak.pad_none(candidate_array, len(symmetry), axis =1)
        candidate_array = candidate_array[..., [symmetry_map[product]]]
    Candidate_dict[product] = candidate_array

  Event_dict = dict()
  for product in diagram['diagram']:
    if product == 'SYMMETRY': continue
    product_name = 'EVENT/{}'.format(product)
    Event_dict[product_name] = Candidate_dict[product].index
    Event_dict = assignment(Candidate_dict[product], diagram['diagram'][product], product_name, Event_dict)

  for ielement in Event_dict:
    Event_dict[ielement] = ak.flatten(ak.fill_none(ak.pad_none(Event_dict[ielement], 1, axis = 1), -1, axis=1), axis = 1)
  
  parton["reco_v4"] = Reco_v4 
  
  Reconstructed_momentum = dict()
  for product in diagram['diagram']:
    if product == 'SYMMETRY': continue
    product_name = 'EVENT/{}'.format(product)
    reco_dict_ = assign_Reco_LorentzVector(Event_dict, diagram['diagram'][product], parton, product_name)
    for reco_ in reco_dict_:
      Reconstructed_momentum[reco_] = reco_dict_[reco_]

  return Reconstructed_momentum
#  decay_chain = decode_feynman_diagram(process_diagram['diagram'])
#  print(decay_chain)
  
#  builder = ak.ArrayBuilder()
#  for iEvent in range(len(Matched_index)):
#    print(iEvent)
#    builder.begin_list()
#    genpart_Event = Object["genpart"][iEvent]
#    Reco_v4_Event = Reco_v4[iEvent]
#    Merged_dict = {}
#    for iPart in range(len(genpart_Event)):
#      current_mother = int(genpart_Event[iPart, 5])
#      current_index  = int(genpart_Event[iPart, 4])
#      current_v4     = Reco_v4_Event[iPart]
#      if current_mother > 1:
#        if current_mother not in Merged_dict:
#          if current_v4.pt > 0:
#            Merged_dict[current_mother] = current_v4
#          else:
#            Merged_dict[current_mother] = None
#        else:
#          if ((current_v4.pt > 0) and (Merged_dict[current_mother] is not None)):
#             Merged_dict[current_mother]  = Merged_dict[current_mother].add(current_v4)
#          else:
#             Merged_dict[current_mother] = None
#    Merged_array = [] 
#    for iPart in range(len(genpart_Event)):
#      current_index = int(genpart_Event[iPart, 4])
#      if (current_index in Merged_dict) and (Merged_dict[current_index] is not None):
#        builder.append(Merged_dict[current_index])
#      else:
#        builder.append(Reco_v4_Event[iPart])
#    builder.end_list()
#  return builder.snapshot()     
  

def Monitor_GenMatching(config):

  plot_dir = os.path.join(config.outdir, config.process)  
  ################################
  # Collect all necessary arrays #
  ################################

  data_dict = {}
  dataset_structure = None 
  for h5name in glob.glob('{indir}/{process}_*.h5'.format(indir = config.indir, process = config.process)):
      if not re.match("{indir}/{process}_[0-9]+\.h5".format(indir = config.indir, process=config.process), h5name): continue
      print(h5name)
      h5fr = h5py.File(h5name, mode = 'r')
      if dataset_structure is None:
        dataset_structure = find_dataset_name(h5fr, list(h5fr))
        print(dataset_structure)
      for key_ in dataset_structure:
        data_dict[key_] = list(h5fr[key_])
      else:
        for key_ in dataset_structure:
          data_dict[key_] = np.concatenate([data_dict[key_], list(h5fr[key_])])
      h5fr.close()


  ##########################
  # Plot Basic Information #
  ##########################

  # Convert numpy to awkward array
  Object = dict()
  for key_ in data_dict:
    object_ = ak.from_regular(ak.from_numpy(data_dict[key_]))
    object_ = object_[object_[:,:,0] > 0]
    Object[key_] = object_
    
  del data_dict 

  DQM_Plot = dict()

  
  DQM_Plot['nLepton'] = Get_hist([6, -0.5, 5.5], ak.num(Object["els"]) + ak.num(Object["mus"]))  
  DQM_Plot['nJet']    = Get_hist([6, -0.5, 5.5], ak.num(Object["jets"]))
  DQM_Plot['nPart_final_product'] = Get_hist([9, -0.5, 8.5], ak.num(Object["genpart"][Object["genpart"][:, :, 8] == 23]))
  DQM_Plot['nPart_matched']       = Get_hist([6, -0.5, 5.5], ak.num(Object["genpart"][Object["genpart"][:, :, 9] > -1]))
  DQM_Plot['GentPart_Flavour']    = Get_hist([49 , -24.5, 24.5], Object["genpart"][:, :, 7])

  ###################
  ## Gen Matching  ##
  ###################

  Feynman_diagram_process = Feynman_diagram[config.process]
  Reco = RecoMass(Object, Feynman_diagram_process)
#  print(Reco)
  for Reco_name in Reco:
    Reco_v4 = Reco[Reco_name]
    DQM_Plot[Reco_name.replace('/', '_')]   = Get_hist([300, 0, 300], Reco_v4[(Reco_v4.rho > 0)].tau)
  Sorted_index = ak.sort(ak.values_astype(Object["genpart"][:, :, 9], int), axis = -1)
  Sorted_index = Sorted_index[Sorted_index > 0]
  run_lengths  = ak.run_lengths(Sorted_index)
  unique_check = ak.values_astype(ak.all(run_lengths == 1, axis = -1), int)
  DQM_Plot['GenPart_doubleAssignment'] = Get_hist([2, -0.5, 1.5], unique_check)


  for Histogram_name in DQM_Plot:
      Draw_hist(DQM_Plot[Histogram_name], Histogram_name, plot_dir, Histogram_name)

   
if __name__ == '__main__':

  usage = 'usage: %prog [options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--indir', type = str)
  parser.add_argument('--outdir', type = str, default = 'DQM_Plot')
  parser.add_argument('--process', type = str)
  parser.add_argument('--scan', action='store_true')
  config = parser.parse_args()
  if config.scan:
    for process_ in Feynman_diagram:
      config.process = process_
      os.system("python3 Monitor_GenMatching.py --indir {} --outdir {} --process {}".format(config.indir, config.outdir, process_))
  else:
    Monitor_GenMatching(config)
