## For MEG data
import os
import numpy as np

Gr1_names = os.listdir('Y:\LevchenkoE\ISC\Experiment3_videos_MEG\data\Group1')
Gr2_names = os.listdir('Y:\LevchenkoE\ISC\Experiment3_videos_MEG\data\Group2')

path = r'Y:\LevchenkoE\ISC\Experiment3_videos_MEG\data\preprocessed'
os.chdir(path)
X_gr1_vid2 = [data for data in os.listdir(path) if data.endswith('vid2.npy') and any(Gr1_name in data for Gr1_name in Gr1_names)]
X_gr2_vid2 = [data for data in os.listdir(path) if data.endswith('vid2.npy') and any(Gr1_name in data for Gr1_name in Gr2_names)]

X_gr1_data = [np.load(dataname)[:306, :72502] for dataname in X_gr1_vid2]
X_gr1_data = np.dstack(X_gr1_data)
X_gr1_data = np.moveaxis(X_gr1_data, -1, 0)
np.save('Z:\LevchenkoE\MEG_ISC_somedata\X_gr1_16subj_data.npy', X_gr1_data)

X_gr2_data = [np.load(dataname)[:306, :72502] for dataname in X_gr2_vid2]
X_gr2_data = np.dstack(X_gr2_data)
X_gr2_data = np.moveaxis(X_gr2_data, -1, 0)
np.save('Z:\LevchenkoE\MEG_ISC_somedata\X_gr2_16subj_data.npy', X_gr2_data)


X_gr1 = np.load(r"Y:\LevchenkoE\ISC\Experiment3_videos_MEG\data\np_mat\vid2_gr1.npy")
X_gr1 = X_gr1[:9,:,:43750]
X_gr2 = np.load(r"Y:\LevchenkoE\ISC\Experiment3_videos_MEG\raw_data\np_mat\vid2_gr2.npy")
X_gr2 = X_gr2[:,:,:43750]

#START HERE
X_gr1_data = np.load('Z:\LevchenkoE\MEG_ISC_somedata\X_gr1_16subj_data.npy')
X_gr2_data = np.load('Z:\LevchenkoE\MEG_ISC_somedata\X_gr2_16subj_data.npy')

X_all = dict(Gr1=X_gr1_data[:, :, :43750], Gr2=X_gr2_data[:, :, :43750])

isc_results = dict()
for name, group in X_all.items():
    print(name)
    print(group.shape)
    print('Seconds: ', group.shape[2]/250)
    [W, ISC_overall] = train_cca(dict(A=X_all[str(name)]))
    print('Train over')
    isc_results[str(name)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(group, W, 250)))
    print('Apply over')

np.save('Z:\LevchenkoE\MEG_ISC_somedata\isc_results_true.npy', isc_results)
print('Done')

# STATISTICS
import scipy.stats as st
from scipy.stats import mannwhitneyu

isc_results_shuffled = np.load("Z:\LevchenkoE\MEG_ISC_somedata\isc_results_shuffled.npy", allow_pickle=True).item()
isc_results = np.load('Z:\LevchenkoE\MEG_ISC_somedata\isc_results_true.npy', allow_pickle=True).item()

#comp1
nth_win_gr1_c1 = np.array([values['ISC_persecond'][0] for shuffle_i, values in isc_results_shuffled.items() if 'Gr1' in shuffle_i])
nth_win_gr2_c1 = np.array([values['ISC_persecond'][0] for shuffle_i, values in isc_results_shuffled.items() if 'Gr2' in shuffle_i])
nth_win_diff_gr1_gr2_c1 = nth_win_gr1_c1[:80, :] - nth_win_gr2_c1[:80, :]
nth_win_diff_gr2_gr1_c1 = nth_win_gr2_c1[:80, :] - nth_win_gr1_c1[:80, :]

conf_ints_gr1_c1 = np.array([st.t.interval(0.95, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_gr1_c1.T])
conf_ints_gr2_c1 = np.array([st.t.interval(0.95, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_gr2_c1.T])
conf_ints_diff_c1_gr1_gr2 = np.array([st.t.interval(0.99975, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_diff_gr1_gr2_c1.T])*5
conf_ints_diff_c1_gr2_gr1 = np.array([st.t.interval(0.99975, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_diff_gr2_gr1_c1.T])*5


#comp2
nth_win_gr1_c2 = np.array([values['ISC_persecond'][1] for shuffle_i, values in isc_results_shuffled.items() if 'Gr1' in shuffle_i])
nth_win_gr2_c2 = np.array([values['ISC_persecond'][1] for shuffle_i, values in isc_results_shuffled.items() if 'Gr2' in shuffle_i])
nth_win_diff_gr1_gr2_c2 = nth_win_gr1_c2[:80, :] - nth_win_gr2_c2[:80, :]
nth_win_diff_gr2_gr1_c2 = nth_win_gr2_c2[:80, :] - nth_win_gr1_c2[:80, :]

conf_ints_diff_c2_gr2_gr1 = np.array([st.t.interval(0.99975, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_diff_gr2_gr1_c2.T])*5
conf_ints_diff_c2_gr1_gr2 = np.array([st.t.interval(0.99975, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_diff_gr1_gr2_c2.T])*5

#comp3
nth_win_gr1_c3 = np.array([values['ISC_persecond'][2] for shuffle_i, values in isc_results_shuffled.items() if 'Gr1' in shuffle_i])
nth_win_gr2_c3 = np.array([values['ISC_persecond'][2] for shuffle_i, values in isc_results_shuffled.items() if 'Gr2' in shuffle_i])
nth_win_diff_gr1_gr2_c3 = nth_win_gr1_c3[:80, :] - nth_win_gr2_c3[:80, :]
nth_win_diff_gr2_gr1_c3 = nth_win_gr2_c3[:80, :] - nth_win_gr1_c3[:80, :]

conf_ints_diff_gr1_gr2_c3 = np.array([st.t.interval(0.99975, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_diff_gr1_gr2_c3.T])*5
conf_ints_diff_gr2_gr1_c3 = np.array([st.t.interval(0.99975, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_diff_gr2_gr1_c3.T])*5

stop = 175
N_win_sign_gr1 = 0
for i in range(0, len(isc_results['Gr1']['ISC_persecond'][0, i])):
    if isc_results['Gr1']['ISC_persecond'][0, i] > conf_ints_gr1[i, 1]:
        N_win_sign_gr1 = N_win_sign_gr1 + 1


#comp1
gr1_win_significance = [isc_results['Gr1']['ISC_persecond'][0][i] > conf_ints_gr1_c1[i, 1] or isc_results['Gr1']['ISC_persecond'][0][i] < conf_ints_gr1_c1[i, 0]
                        for i in range(0, len(conf_ints_gr1_c1))]

gr_win_difference_c1_099975 = [isc_results['Gr1']['ISC_persecond'][0][i]-isc_results['Gr2']['ISC_persecond'][0][i] > conf_ints_diff_c1_gr1_gr2[i, 1] or
                               isc_results['Gr1']['ISC_persecond'][0][i]-isc_results['Gr2']['ISC_persecond'][0][i] < conf_ints_diff_c1_gr1_gr2[i, 0]
                               for i in range(0, len(conf_ints_diff_c1_gr1_gr2))]

gr_win_difference_c1_095 = [isc_results['Gr1']['ISC_persecond'][0][i]-isc_results['Gr2']['ISC_persecond'][0][i] > conf_ints_diff[i, 1] for i in range(0, len(conf_ints_diff))]

#comp2
gr_win_difference_c2 = [isc_results['Gr1']['ISC_persecond'][1][i]-isc_results['Gr2']['ISC_persecond'][1][i] > conf_ints_diff[i, 1] for i in range(0, len(conf_ints_diff))]

gr_win_difference_c2_099975 = [isc_results['Gr1']['ISC_persecond'][1][i]-isc_results['Gr2']['ISC_persecond'][1][i] > conf_ints_diff_c2_gr1_gr2[i, 1] or
                               isc_results['Gr1']['ISC_persecond'][1][i]-isc_results['Gr2']['ISC_persecond'][1][i] < conf_ints_diff_c2_gr1_gr2[i, 0]
                               for i in range(0, len(conf_ints_diff_c2_gr1_gr2))]

#comp3
gr_win_difference_c3 = [isc_results['Gr1']['ISC_persecond'][2][i]-isc_results['Gr2']['ISC_persecond'][2][i] > conf_ints_diff[i, 1] for i in range(0, len(conf_ints_diff))]

gr_win_difference_c3_099975 = [isc_results['Gr1']['ISC_persecond'][2][i]-isc_results['Gr2']['ISC_persecond'][2][i] > conf_ints_diff_gr1_gr2_c3[i, 1] or
                               isc_results['Gr1']['ISC_persecond'][2][i]-isc_results['Gr2']['ISC_persecond'][2][i] < conf_ints_diff_gr1_gr2_c3[i, 0]
                               for i in range(0, len(conf_ints_diff_gr1_gr2_c3))]


N_win_sign_gr2 = 0
for i in range(0, stop):
    if isc_results['Gr2']['ISC_persecond'][0, i] > conf_ints_gr2[i, 1]:
        N_win_sign_gr2 = N_win_sign_gr2 + 1;

gr2_win_significance = [isc_results['Gr2']['ISC_persecond'][0][i] > conf_ints_gr2[i, 1] for i in range(0, len(conf_ints_gr2))]

# AREA UNDER THE CURVE
from numpy import trapz
import numpy as np

area_gr1_general = trapz(isc_results['Gr1']['ISC_persecond'][0,:])
area_gr1_general

area_gr2_general = trapz(isc_results['Gr2']['ISC_persecond'][0,:])
area_gr2_general


height_whole_gr1 = isc_results['Gr1']['ISC_persecond'][0,:]-conf_ints_gr1[:,1]
height_whole_gr2 = isc_results['Gr2']['ISC_persecond'][0,:]-conf_ints_gr2[:,1]
np.median(height_whole_gr1)
np.median(height_whole_gr2)
u_height_gr12_whole = mannwhitneyu(height_whole_gr1, height_whole_gr2)
u_height_gr12_whole


height_1part_gr1 = isc_results['Gr1']['ISC_persecond'][0,:175]-conf_ints_gr1[:175,1]
height_1part_gr2 = isc_results['Gr2']['ISC_persecond'][0,:175]-conf_ints_gr2[:175,1]
np.median(height_1part_gr1)
np.median(height_1part_gr2)
u_height_gr12_part1 = mannwhitneyu(height_1part_gr1, height_1part_gr2)
u_height_gr12_part1



# BETWEEN GROUPS COMPARISON FOR THE WHOLE MOVIE
from scipy.stats import ttest_ind

u_gr12 = mannwhitneyu(isc_results['Gr1']['ISC_bysubject'][0], isc_results['Gr2']['ISC_bysubject'][0])
u_gr12


t_gr12 = ttest_ind(isc_results['Gr1']['ISC_bysubject'][0], isc_results['Gr2']['ISC_bysubject'][0])
t_gr12

u_gr12_1part = mannwhitneyu(isc_results['Gr1']['ISC_bysubject'][0], isc_results['Gr2']['ISC_bysubject'][0])
u_gr12_1part
np.median(isc_results['Gr1']['ISC_bysubject'][0])
np.median(isc_results['Gr2']['ISC_bysubject'][0])


t_gr12_1part = ttest_ind(isc_results['Gr1']['ISC_bysubject'][0], isc_results['Gr2']['ISC_bysubject'][0])
t_gr12_1part

# BETWEEN GROUPS COMPARISON FOR THE 1ST PART OF THE MOVIE
from scipy.stats import ttest_ind

u_gr12 = mannwhitneyu(isc_results['Gr1']['ISC_bysubject'][0], isc_results['Gr2']['ISC_bysubject'][0])
u_gr12

t_gr12 = ttest_ind(isc_results['Gr1']['ISC_bysubject'][0], isc_results['Gr2']['ISC_bysubject'][0])
t_gr12

#BETWEEN GROUPS FOR BEHAVOIOURAL DATA
from scipy.stats import mannwhitneyu


gr1_influence = [6, 7, 5, 10,4,5,7,9,1,7,5,4,8,1]
gr2_influence = [1,4,8,2,2,2,4,5,8,2]

gr1_interest = [7,6,5,5,6,5,7,4,5,5,5,6,5,5]
gr2_interest = [5,5,3,8,4,3,5,5,8,6]

gr1_motivation = [1,4,1,10,2,1,7,3,2,7,1,1,7,1]
gr2_motivation = [3,4,1,5,3,4,3,1,2,1]

gr1_fabricated = [7,6,5,5,6,5,7,4,5,5,5,6,5,5]
gr2_fabricated = [5,5,3,8,4,3,5,5,8,6]


u_influence = mannwhitneyu(gr1_influence, gr2_influence)
u_interest = mannwhitneyu(gr1_interest, gr2_interest)
u_motivation = mannwhitneyu(gr1_motivation, gr2_motivation)
u_fabricated = mannwhitneyu(gr1_fabricated, gr2_fabricated)

