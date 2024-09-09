import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
# try to directly read delphes root files and export event level variables.
SCHEMA={
    "jets":[ # could be any object name
        4, # how many?
        {
            'pt':'Jet/Jet.PT',
            'eta':'Jet/Jet.Eta',
            'phi':'Jet/Jet.Phi',
            'mass':'Jet/Jet.Mass',
        }, # the 4-vector input
        {
            "BTag":"Jet/Jet.BTag",
            "NCharged":"Jet/Jet.NCharged",
            "Flavor":"Jet/Jet.Flavor"
        }, # the aux input
        lambda p,v:p.pt>10, # the cut
        {
            "jet_pt":lambda p,v:p.pt,
            "jet_eta":lambda p,v:p.eta,
            "jet_phi":lambda p,v:p.phi,
            "jet_m":lambda p,v:p.mass,
            "jet_btag":lambda p,v:v["BTag"],
            "jet_npart":lambda p,v:v["NCharged"],
            "jet_flavor":lambda p,v:v["Flavor"],
        }, # prepare the output variables
        ["jet_pt","jet_eta","jet_phi","jet_m","jet_btag","jet_npart","jet_flavor"], # the actual saved one and the order
    ],
    "phs":[ # could be any object name
        2, # how many?
        {
            'pt':'Photon/Photon.PT',
            'eta':'Photon/Photon.Eta',
            'phi':'Photon/Photon.Phi',
            'mass':"Photon/Photon.PT*0."
        }, # the 4-vector input
        {
        }, # the aux input
        lambda p,v:p.pt>10, # the cut
        {
            "ph_pt":lambda p,v:p.pt,
            "ph_eta":lambda p,v:p.eta,
            "ph_phi":lambda p,v:p.phi,
            "ph_m":lambda p,v:p.mass,
        }, # prepare the output variables
        ["ph_pt","ph_eta","ph_phi","ph_m"], # the actual saved one and the order
    ],
    "els":[ # could be any object name
        4, # how many?
        {
            'pt':'Electron/Electron.PT',
            'eta':'Electron/Electron.Eta',
            'phi':'Electron/Electron.Phi',
            'mass':"Electron/Electron.PT*0.511E-3"
        }, # the 4-vector input
        {
            "charge":"Electron/Electron.Charge",
        }, # the aux input
        lambda p,v:(p.pt>10) & (abs(p.eta)<2.47), # the cut
        {
            "el_pt":lambda p,v:p.pt,
            "el_eta":lambda p,v:p.eta,
            "el_phi":lambda p,v:p.phi,
            "el_m":lambda p,v:p.mass,
            "el_ch":lambda p,v:v["charge"],
        }, # prepare the output variables
        ["el_pt","el_eta","el_phi","el_m","el_ch"], # the actual saved one and the order
    ],
    "mus":[ # could be any object name
        4, # how many?
        {
            'pt':'Muon/Muon.PT',
            'eta':'Muon/Muon.Eta',
            'phi':'Muon/Muon.Phi',
            'mass':"Muon/Muon.PT*105.66E-3"
        }, # the 4-vector input
        {
            "charge":"Muon/Muon.Charge",
        }, # the aux input
        lambda p,v:(p.pt>10) & (abs(p.eta)<2.47), # the cut
        {
            "mu_pt":lambda p,v:p.pt,
            "mu_eta":lambda p,v:p.eta,
            "mu_phi":lambda p,v:p.phi,
            "mu_m":lambda p,v:p.mass,
            "mu_ch":lambda p,v:v["charge"],
        }, # prepare the output variables
        ["mu_pt","mu_eta","mu_phi","mu_m","mu_ch"], # the actual saved one and the order
    ],
    "tas":[ # could be any object name
        4, # how many?
        {
            'pt':'Jet/Jet.PT',
            'eta':'Jet/Jet.Eta',
            'phi':'Jet/Jet.Phi',
            'mass':"Jet/Jet.Mass"
        }, # the 4-vector input
        {
            "tauID":"Jet/Jet.TauTag",
            "charge":"Jet/Jet.Charge",
        }, # the aux input
        lambda p,v:v["tauID"]==1, # the cut
        {
            "ta_pt":lambda p,v:p.pt,
            "ta_eta":lambda p,v:p.eta,
            "ta_phi":lambda p,v:p.phi,
            "ta_m":lambda p,v:p.mass,
            "ta_ch":lambda p,v:v["charge"],
        }, # prepare the output variables
        ["ta_pt","ta_eta","ta_phi","ta_m","ta_ch"], # the actual saved one and the order
    ],
    "event":[ # hardcorded name. not change
        {
            "met_met":"MissingET/MissingET.MET",
            "met_phi":"MissingET/MissingET.Phi",
            "weight":"Event/Event.Weight",
        }, # the input variable
        lambda e,o:e["weight"]>0, # the cut.
        {
            "met_met":lambda e,o:e["met_met"],
            "met_phi":lambda e,o:e["met_phi"],
            "weight":lambda e,o:e["weight"],
        }, # prepare the output variables
        ["met_met","met_phi","weight",], # the actual saved one and the order
    ], 
}

def read_file(
        filepath,
        scheme=SCHEMA):

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x
    
    def read_fixed_length_objects(tree,max_len,v_dict,a_dict,cut,s_dict,o_list):
        table1=tree.arrays(v_dict.values())
        p4 = vector.zip({k: table1[v] for k,v in v_dict.items()})
        table2=tree.arrays(a_dict.keys(),aliases=a_dict) if len(a_dict)>0 else {}
        mask=cut(p4,table2)
        p4=p4[mask]
        table2=table2[mask] if len(table2)>0 else {}
        #
        ret_dict={k:v(p4,table2) for k,v in s_dict.items()}
        ret_np=np.stack([ak.to_numpy(_pad(ret_dict[n], maxlen=max_len)) for n in o_list], axis=-1)
        return ret_np,ret_dict

    def read_event(tree,objects,i_dict,cut,s_dict,o_list):
        table2=tree.arrays(i_dict.keys(),aliases=i_dict)
        mask=cut(table2,objects)
        table2=table2[mask]
        ret_dict={k:v(table2,objects) for k,v in s_dict.items()}
        ret_np=np.stack([ak.to_numpy(ret_dict[n]) for n in o_list], axis=-1)
        return ret_np,ret_dict
        

    tree = uproot.open(filepath)['Delphes'] # not load all the branches.
    
    # later proably dict could be ak record?
    objects={k:read_fixed_length_objects(tree,*scheme[k]) for k in scheme.keys() if k!="event"}
    event=read_event(tree,objects,*scheme["event"])
    
    x_objects={k:object[0] for k,object in objects.items()}
    x_event=event[0]
    
    y=None # remain for later usage
    
    return x_objects, x_event, y
