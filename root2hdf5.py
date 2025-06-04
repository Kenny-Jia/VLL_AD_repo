import numpy as np
import uproot
import h5py
# Configuration
MAX_PARTICLES = 4

# Feature lists based on README
EVENT_VARIABLES = [
    'PV_x', 'PV_y', 'PV_z',
    'electron_n_baseline', 'photon_n'
]

ELECTRON_FEATURES = [
    'electron_E', 'electron_pt', 'electron_eta', 'electron_phi',
    'electron_time',
    'electron_d0', 'electron_z0', 'electron_dpt',
    'electron_nPIX', 'electron_nMissingLayers',
    'electron_chi2', 'electron_numberDoF',  # Will need to handle ratio
    'electron_f1', 'electron_f3',
    'electron_z',
    'electron_LHValue',
    'electron_isIsolated_Loose_VarRad'
]

PHOTON_FEATURES = [
    'photon_E', 'photon_pt', 'photon_eta', 'photon_phi',
    'photon_time',
    'photon_maxEcell_E', 'photon_maxEcell_t',
    'photon_maxEcell_x', 'photon_maxEcell_y', 'photon_maxEcell_z',
    'photon_f1', 'photon_f3', 'photon_r1', 'photon_r2',
    'photon_etas1', 'photon_phis1',
    'photon_z',
    'photon_isIsolated_FixedCutLoose'  # Will need to calculate derived position
]

def get_particle_arrays(tree, particle_type, features):
    n_events = tree.num_entries
    arrays = {}
    
    root_arrays = tree.arrays(features, library="numpy")
    
    for feature in features:
        padded_array = np.zeros((n_events, MAX_PARTICLES), dtype=np.float32)
        for event_idx in range(n_events):
            event_particles = root_arrays[feature][event_idx]
            n_to_copy = min(len(event_particles), MAX_PARTICLES)
            padded_array[event_idx, :n_to_copy] = event_particles[:n_to_copy]
        arrays[feature] = padded_array
       
    return arrays

def convert_root_to_h5(input_file, output_file, tree_name='trees_SR_highd0'):
    root_file = uproot.open(input_file)
    tree = root_file[tree_name]
    
    event_arrays = tree.arrays(EVENT_VARIABLES, library="numpy")
    electron_arrays = get_particle_arrays(tree, 'electron', ELECTRON_FEATURES)
    photon_arrays = get_particle_arrays(tree, 'photon', PHOTON_FEATURES)
    
    with h5py.File(output_file, 'w') as f:
        events_group = f.create_group('events')
        
        for var in EVENT_VARIABLES:
            events_group.create_dataset(var, data=event_arrays[var])
        
        electron_group = events_group.create_group('electrons')
        for feature, array in electron_arrays.items():
            electron_group.create_dataset(feature, data=array)
        
        photon_group = events_group.create_group('photons')
        for feature, array in photon_arrays.items():
            photon_group.create_dataset(feature, data=array)
            
    root_file.close()

# Input configuration
input_prefix = ""
file_list = "../file_data.txt"  # text file containing the file names

# Process each file
with open(file_list, 'r') as f:
    for line in f:
        # Clean the line
        filename = line.strip()
        if not filename:  # skip empty lines
            continue
        # Get the filename part (after the last '/')
        shortname = filename.split('/')[-1]

        # Remove the 'user.ewoodwar.' prefix
        stripped_name = shortname.replace('user.ewoodwar.', '')

        # Remove the '.root' extension
        dataset_id = stripped_name[:-5]

        # Construct full paths
        input_path = filename
        output_file = f"hdf5_output/data_{dataset_id}.h5"
        
        print(f"Processing {filename}")
        print(f"  Dataset ID: {dataset_id}")
        print(f"  Output: {output_file}")
        
        try:
            convert_root_to_h5(input_path, output_file)
            print("  Conversion successful!")
        except Exception as e:
            print(f"  Error processing file: {e}")
        
        print("---")