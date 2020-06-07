## STATISTICS FOR MEG ###
#Run data shuffling N times
import os
import numpy as np

isc_results_shuffled = dict()
for i in range(0, 100):
    data_shuffled = shuffle_in_time(X_all, 5, 250)

    for name, group in data_shuffled.items():
        print(name)
        print(group.shape)
        print('Seconds: ', group.shape[2] / 250)
        print(i)
        [W, ISC_overall] = train_cca(dict(A=data_shuffled[str(name)]))
        isc_results_shuffled[str(name)+'_shuffle'+str(i)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(group, W, 250)))

np.save("Z:\LevchenkoE\MEG_ISC_somedata\isc_results_shuffled", isc_results_shuffled)

### MEG ###
# BETWEEN GROUP COMPARISON #

import random

isc_results = np.load('Z:\LevchenkoE\MEG_ISC_somedata\isc_results_true.npy', allow_pickle=True).item()

X_gr1 = np.load(r"Z:\LevchenkoE\MEG_ISC_somedata\X_gr1_16subj_data.npy")
X_gr2 = np.load(r"Z:\LevchenkoE\MEG_ISC_somedata\X_gr2_16subj_data.npy")
X_all = dict(Gr1=X_gr1, Gr2=X_gr2)
X_allin_one = dict(Gr1_2=np.concatenate((X_gr1, X_gr2), axis=0))


isc_results_pseudo_groups = dict()
for i in range(0, 50):
    pseudo_gr1_inds = random.sample(range(32), 16)
    pseudo_gr2_inds = [ind for ind in list(range(0, 32)) if ind not in pseudo_gr1_inds]

    X_pseudo_groups = dict(Gr1=X_allin_one['Gr1_2'][pseudo_gr1_inds, :, :], Gr2=X_allin_one['Gr1_2'][pseudo_gr2_inds, :, :])
    for name, group in X_pseudo_groups.items():
        print(name)
        print(group.shape)
        print('Length in seconds: ', group.shape[2] / 250)
        print(i)
        [W, ISC_overall] = train_cca(dict(A=X_pseudo_groups[str(name)]))
        isc_results_pseudo_groups[str(name)+'_shuffle'+str(i)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(group, W, 250)))

np.save("Z:\LevchenkoE\MEG_ISC_somedata\isc_results_pseudo_groups", isc_results_pseudo_groups)

isc_results_pseudo_groups = np.load("Z:\LevchenkoE\MEG_ISC_somedata\isc_results_pseudo_groups.npy", allow_pickle=True).item()
del isc_results_pseudo_groups['Gr1_shuffle48']

pseudo_gr1_all_shuffles = np.array([values['ISC_persecond'][0] for shuffle_i, values in isc_results_pseudo_groups.items() if 'Gr1' in shuffle_i])
pseudo_gr2_all_shuffles = np.array([values['ISC_persecond'][0] for shuffle_i, values in isc_results_pseudo_groups.items() if 'Gr2' in shuffle_i])
pseudo_gr12_difference = pseudo_gr1_all_shuffles - pseudo_gr2_all_shuffles
pseudo_gr21_difference = pseudo_gr2_all_shuffles - pseudo_gr1_all_shuffles

import scipy.stats as st
conf_ints_difference = np.array([st.t.interval(0.95, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in pseudo_gr12_difference.T])

pseudo_grs_difference_c1_095 = [isc_results['Gr1']['ISC_persecond'][0][i]-isc_results['Gr2']['ISC_persecond'][0][i] > conf_ints_difference[i, 1] or
                                isc_results['Gr1']['ISC_persecond'][0][i]-isc_results['Gr2']['ISC_persecond'][0][i] < conf_ints_difference[i, 0]
                                for i in range(0, len(isc_results['Gr1']['ISC_persecond'][0]))]



###



# #
# #### MEG
# # Test
# X = dict(Sin=np.random.rand(10, 32, 512*60))
# X['Sin'][:,24,:] = np.sin(np.linspace(-np.pi, np.pi, 512*60))
# [W, ISC_overall] = train_cca(X)
#
# isc_results = dict()
# for cond_key, cond_values in X.items():
#     isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'],
#                                           apply_cca(cond_values, W, 250)))
#
# # NT DATA
# from scipy.io import loadmat
#
# # Preprocessed data, train overall
# # X = dict()
# # for i in range(1,6):
# #     X['cond'+str(i)] = loadmat(r"Y:\LevchenkoE\NT_ISC\data2_nt-preprocess\data" + str(i) + "_preproc.mat")['data'+str(i)]
# # print([cond.shape for cond in X.values()])
# #
# # [W, ISC_overall] = train_cca(X)
# #
# # isc_results = dict()
# # for cond_key, cond_values in data.items():
# #     isc_results[str(cond_key)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'],
# #                                           apply_cca(cond_values, W, 250)))
# #
# # # Preprocessed data, train on each movie independtly
# # X = dict()
# # isc_results = dict()
# # for i in range(1, 6):
# #     X['cond' + str(i)] = loadmat(r"Y:\LevchenkoE\NT_ISC\data2_nt-preprocess\data" + str(i) + "_preproc.mat")['data' + str(i)]
# #     [W, ISC_overall] = train_cca(X)
# #     isc_results['data'+str(i)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'],
# #                                           apply_cca(X['cond'+str(i)], W, 250)))
# #     plot_isc(isc_results)
# #

# # For MEG data
# X_gr1 = np.load(r"Y:\LevchenkoE\ISC\Experiment3_videos_MEG\raw_data\np_mat\vid2_gr1.npy")
# X_gr1 = X_gr1[:9,:,:]
# X_gr2 = np.load(r"Y:\LevchenkoE\ISC\Experiment3_videos_MEG\raw_data\np_mat\vid2_gr2.npy")
#
# X_all = dict(Gr1=X_gr1, Gr2=X_gr2)
#
# isc_results = dict()
# for name, group in X_all.items():
#     print(name)
#     print(group.shape)
#     print('Seconds: ', group.shape[2]/250)
#     [W, ISC_overall] = train_cca(dict(A=X_all[str(name)]))
#     isc_results[str(name)] = dict(zip(['ISC', 'ISC_persecond', 'ISC_bysubject', 'A'], apply_cca(group, W, 250)))
#
#

#
# isc_results_shuffle = np.load("Y:\LevchenkoE\ISC\Experiment3_videos_MEG\\analysis\isc_results_shuffled.npy", allow_pickle=True).item()
#
# nth_win_gr1 = np.array([values['ISC_persecond'][0] for shuffle_i, values in isc_results_shuffle.items() if 'Gr1' in shuffle_i])
# nth_win_gr2 = np.array([values['ISC_persecond'][0] for shuffle_i, values in isc_results_shuffle.items() if 'Gr2' in shuffle_i])
#
# import scipy.stats as st
# conf_ints_gr1 = np.array([st.t.interval(0.95, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_gr1.T])
# conf_ints_gr2 = np.array([st.t.interval(0.95, len(window)-1, loc=np.mean(window), scale=st.sem(window)) for window in nth_win_gr2.T])
#




# from scipy.io import loadmat
# import numpy as np
#
# data = dict()
# for i in range(1,6):
#     if i == 1:
#         data['cond'+str(i)] = loadmat(r"Y:\LevchenkoE\NT_ISC\data2\data" + str(i) + ".mat")['data'+str(i)][:10]
#     else:
#         data['cond' + str(i)] = loadmat(r"Y:\LevchenkoE\NT_ISC\data2\data" + str(i) + ".mat")['data' + str(i)]
# print([cond.shape for cond in data.values()])


# #
import mne
data_A = mne.io.read_raw_fif(fname=r'Y:\LevchenkoE\ISC\Experiment3_videos_MEG\data\Group1\akjolbaev_david\190416\MaxFilter_processing\akjolbaev_david_vid4_tsss_mc_trans.fif', preload=True)
montage = mne.channels.read_montage('EGI_256.csd')
data_A.set_montage(montage)
pos_2d = np.array( [montage.get_pos2d()
                   [np.where((data_A.info['chs'][channel]['loc'][0:3] == montage.pos).sum(axis=1) == 3)[0][0]]
                   for channel in range(0, 62)] )
#pos_32 = pos_2d[:32,:]
grad_indices = mne.pick_types(data_A.info, meg='grad')
reduced_info = mne.pick_info(data_A.info, grad_indices)
topoplot = mne.viz.plot_topomap(data=isc_results['Gr1']['A'][2, 0:204], pos=reduced_info)

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# from matplotlib.animation import FuncAnimation
# img = plt.imshow(r"C:\Users\jlamekina\Pictures\cat_png.png")
# # Some global variables to define the whole run
# total_number_of_frames = 100
# all_data = [np.random.rand(512, 512) for x in range(100)]
#
#
# def animate(frame):
#     """
#     Animation function. Takes the current frame number (to select the potion of
#     data to plot) and a line object to update.
#     """
#
#     # Not strictly neccessary, just so we know we are stealing these from
#     # the global scope
#     global all_data, image
#
#     # We want up-to and _including_ the frame'th element
#     image.set_array(all_data[frame])
#
#     return image
#
#
# # Now we can do the plotting!
# fig, ax = plt.subplots(1, figsize=(1, 1))
# # Remove a bunch of stuff to make sure we only 'see' the actual imshow
# # Stretch to fit the whole plane
# fig.subplots_adjust(0, 0, 1, 1)
# # Remove bounding line
# # ax.axis("off")
#
# # Initialise our plot. Make sure you set vmin and vmax!
# image = ax.imshow(all_data[0], vmin=0, vmax=1)
#
# animation = FuncAnimation(
#     # Your Matplotlib Figure object
#     fig,
#     # The function that does the updating of the Figure
#     animate,
#     # Frame information (here just frame number)
#     np.arange(total_number_of_frames),
#     # Extra arguments to the animate function
#     fargs=[],
#     # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
#     interval=1000 / 25
# )
# import cv2
# # Try to set the DPI to the actual number of pixels you're plotting
# animation.save("out_2dgrid.mp4", dpi=512)

import cv2
cap = cv2.VideoCapture(r"Y:\LevchenkoE\video3.avi")

while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    channels = cv2.split(frame)
    frame_merge = cv2.merge(channels)

    # horizintally concatenating the two frames.
    final_frame = cv2.vconcat((frame, frame_merge))
    final_frame = cv2.resize(final_frame, None, fx=0.5, fy=0.5)

    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', final_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def update_line(num, data, line):
    line.set_data(data[..., :num])
    plt.xlim()
    return line,

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('X')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l), interval=50, blit=True)

fig, ax = plt.subplots()
x = np.arange(0, 123, 1)
y = np.random.rand(123)
line, = ax.plot(x, y)

for i in x:
    plt.plot(x[:i+1], y[:i+1])


#ani = animation.FuncAnimation(fig, animate, interval=1, save_count=1, blit=True)

fig, ax = plt.subplots()
xdata, ydata = np.arange(0, 123), isc_results['Sin']['ISC_persecond'][0, :]

ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-0.1, 0.1)
    return ln,

def update(frame):
    ax.set_xlim(0, frame)
    ax.set_ylim(-0.1, max(xdata[frame]))
    ln.set_data(xdata[frame], ydata[:frame])
    return ln,

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), blit=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = np.arange(1, 123), isc_results['Sin']['ISC_persecond'][0, :]
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 123)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    ln.set_data(xdata[frame], ydata[:frame])
    return ln,

ani = FuncAnimation(fig, update, frames = xdata, init_func=init, blit=True)

plt.show()



### FOR PLOT CI
# plt.fill_between(np.arange(0, len(conf_ints_gr2[:, 1])), 0, conf_ints_gr2[:, 1], color='grey')
# N_win_sign = sum(cond['ISC_persecond'][0] > conf_ints_gr2[:, 1])
# print(N_win_sign)




