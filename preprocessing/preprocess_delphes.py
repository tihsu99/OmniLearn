import os
import numpy as np
import h5py
from sklearn.model_selection import KFold
from dataloader_delphes import read_file
import gc
from glob import glob
from optparse import OptionParser
import tqdm

def process_and_save(folder_path, n_splits,name, output_path,index):
    # List all files in the folder
    all_files = glob(folder_path)

    # Split the files into n_splits
    kf = KFold(n_splits=n_splits) if n_splits>1 else None
    files_list=kf.split(all_files) if kf else [[None,list(range(len(all_files)))]]

    for i, (_, split_indices) in enumerate(files_list):
        if index>=0 and i!=index:
            continue
            
        print("Running Fold {}".format(i))
        # Initialize lists to store concatenated data
        concat_objects = {} # object level things. each object save to one dataset
        concat_event = [] # event level things. save to "event"
        init=True
        # concat_y = []  # dummy truth label. determine later
        # Process each file in the current split
        for idx in split_indices:
            file_path = all_files[idx]
            x_objects, x_event, _ = read_file(file_path)
            if init:
                for k,v in x_objects.items():
                    concat_objects[k]=[v]
                init=False
            else:
                for k,v in x_objects.items():
                    concat_objects[k].append(v)

            concat_event.append(x_event)
            # concat_y.append(y)

        # Concatenate all X and y
        final_objects = {k:np.concatenate(obj, axis=0) for k,obj in concat_objects.items()}
        final_event = np.concatenate(concat_event, axis=0)
        
        # Save to h5py file
        fo='{}/{}'.format(output_path,name)
        if not os.path.exists(fo):
            os.makedirs(fo,exist_ok=True)
        with h5py.File('{}/{}/kfold_{}_{}.h5'.format(output_path,name,n_splits,i), 'w') as h5f:
            for k,v in final_objects.items():
                h5f.create_dataset(k, data=v)
            h5f.create_dataset('event', data=final_event)
        del final_objects, final_event
        gc.collect()

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--sample", type="string", default='top', help="Sample type")
    parser.add_option("--folder", type="string", default='{sample}.reco.root', help="Folder containing downloaded files")
    parser.add_option("--out", type="string", default='dump_h5', help="output folder")
    parser.add_option("--index", type="int", default=-1, help="for parallel jobs")

    (flags, args) = parser.parse_args()
    
    process_and_save(flags.folder.format(sample=flags.sample), n_splits=1,name=flags.sample,output_path=flags.out,index=flags.index)






