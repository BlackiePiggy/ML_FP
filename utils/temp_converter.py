import pickle
import scipy.io as sio
import sys
import numpy as np

def convert_to_matlab_compatible(obj):
    if isinstance(obj, dict):
        return {k: convert_to_matlab_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_matlab_compatible(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return obj

with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

matlab_data = convert_to_matlab_compatible(data)
output_file = sys.argv[1].replace('.pkl', '.mat')
sio.savemat(output_file, matlab_data)
print(f'Converted to {output_file}')
