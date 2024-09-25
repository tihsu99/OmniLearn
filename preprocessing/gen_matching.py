import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()

def define_Lorentz_vector(obj, obj_name):
  obj_out = ak.zip(obj[1])
  obj_out["v4"] = vector.zip(
    {
        "pt": obj_out["{}_pt".format(obj_name)],
        "eta": obj_out["{}_eta".format(obj_name)],
        "phi": obj_out["{}_phi".format(obj_name)],
        "mass": obj_out["{}_m".format(obj_name)],
    }
  )

  if "{}_PID".format(obj_name) in obj[1]:
    obj_out["PID"] = obj[1]["{}_PID".format(obj_name)]

  if "{}_Status".format(obj_name) in obj[1]:
    obj_out["Status"] = obj[1]["{}_Status".format(obj_name)]

  obj_out["idx"] = ak.local_index(obj_out)
  return obj_out

def match_object(candidate, target, dr_cut):
  if (len(target) < 1): 
    return -1

  deltaR = candidate.v4.deltaR(target.v4)
  min_deltaR = ak.min(deltaR)
  if (min_deltaR < dr_cut):
    return ak.argmin(deltaR)
  else:
    return -1  
  

def find_matching(objects, dr_cut):

  genParticles = define_Lorentz_vector(objects["genpart"], 'genpart')
  electrons    = define_Lorentz_vector(objects["els"], 'el')
  muons        = define_Lorentz_vector(objects["mus"], 'mu')
  jets         = define_Lorentz_vector(objects["jets"], 'jet')

  genParticles["matched_index"] = ak.ones_like(genParticles["idx"]) * -99

  builder = ak.ArrayBuilder()

  for iEvent in range(len(genParticles)):
   builder.begin_list()
   genPart_Event  = genParticles[iEvent]
   electron_Event = electrons[iEvent]
   muon_Event     = muons[iEvent] 
   jet_Event      = jets[iEvent]
  
   for iPart in range(len(genPart_Event)):
     genPart = genPart_Event[iPart]
     if not (genPart.Status == 23): # Not outgoing products
       builder.append(-1)
     elif ((abs(genPart.PID) == 12) or (abs(genPart.PID == 14)) or (abs(genPart.PID == 16))): # Neutrino case
       builder.append(-1)
     elif (abs(genPart.PID) == 11): # Electron case
       builder.append(match_object(genPart, electron_Event, dr_cut))
     elif (abs(genPart.PID) == 13): # Muon case
       builder.append(match_object(genPart, muon_Event, dr_cut))
     else: # Jet case
       builder.append(match_object(genPart, jet_Event, dr_cut))
   builder.end_list()
   #print('================')
   #print('PID', genPart_Event.PID)
   #print('Status', genPart_Event.Status)
   #print('gen Eta', genPart_Event.v4.eta)
   #print('elec Eta', electron_Event.v4.eta)
   #print('jet Eta', jet_Event.v4.eta)
   #print('matched_idx', builder.snapshot()[iEvent])
  return builder.snapshot()
