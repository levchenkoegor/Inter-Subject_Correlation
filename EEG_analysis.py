### LOAD EEG DATA

import os

path = "Y:\\LevchenkoE\\ISC\\"
os.chdir(path)

# Collect all paths into one list of lists
paths_all_vids = [[path + '\\' + f for (path, folders, files) in os.walk(path)
                   if 'EEG\\preprocessed' in path
                   for f in files
                   if f.endswith('vid' + str(i) + '.npy')]
                  for i in range(1, 6)]

# Load all data into one dictionary
data_all_vids = dict()
for i in range(0, 5):
    vid_i_str = 'vid' + str(i + 1)
    data_all_vids[vid_i_str] = np.array([np.squeeze(np.load(path_vid_i)) for path_vid_i in paths_all_vids[i]])
    print(f'{vid_i_str} with {data_all_vids[vid_i_str].shape} shape is loaded.')


# Run ISC analysis on real data
isc_results = dict()
fs = 512
[W, ISC_overall] = train_cca(data_all_vids)

for condition, values in data_all_vids.items():
    print(condition)
    print(values.shape)
    print('Duration in seconds: ', values.shape[2]/fs)
    isc_results[str(condition)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(values, W, fs)))


# Run ISC analysis on synthetic data
import scipy.stats as st
D = 62

# vid1
rend_i_vid1 = np.zeros((D, 100))
data_iscs = np.load(r"Y:\LevchenkoE\ISC\Experiment1_videos_EEG\analysis\Surrogated data\Xr_vid1_ISCs.npy",
                    allow_pickle=True).item()
for rend in range(0, 100):
    rend_i_vid1[:, rend] = data_iscs['rend'+str(rend+1)][0]

ci_vid1 = st.t.interval(0.99, len(rend_i_vid1[0, :])-1, loc=np.mean(rend_i_vid1[0, :]), scale=st.sem(rend_i_vid1[0, :]))

# vid4
rend_i_vid4 = np.zeros((D, 100))
data_iscs = np.load(r"Y:\LevchenkoE\ISC\Experiment1_videos_EEG\analysis\Surrogated data\Xr_vid4_ISCs.npy",
                    allow_pickle=True).item()
for rend in range(0, 100):
    rend_i_vid4[:, rend] = data_iscs['rend'+str(rend+1)][0]

ci_vid4 = st.t.interval(0.99, len(rend_i_vid4[0, :])-1, loc=np.mean(rend_i_vid4[0, :]), scale=st.sem(rend_i_vid4[0, :]))

