from scipy.io import wavfile
import os 
import h5py
import pdb
import sys
trainroot = sys.argv[1]
#trainroot = ['/data02/junjiang5/from_dlp/2020/11/cpc_resnet/data_short_7lid']

# store train 
h5f = h5py.File('train.h5', 'a')
#pdb.set_trace()

for subdir, dirs, files in os.walk(trainroot):
    for file in files:
        if file.endswith('.wav'):
            fullpath = os.path.join(subdir, file)
            fs, data = wavfile.read(fullpath)
            h5f.create_dataset(file[:-4], data=data)
            print(file[:-4])
h5f.close()


