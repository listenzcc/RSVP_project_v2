# %%
import os
import mne
import tqdm
import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
print(__doc__)

# %%
path_table = pd.read_json('path_table.json')
path_table

# %%
iter_freqs = [
    ('a', 0.1, 4),
    ('b', 1, 4),
    ('c', 0.1, 30),
    ('d', 1, 30),
#     ('Delta0', 0, 4),
#     ('Delta', 1, 4),
#     ('Theta', 4, 7),
#     ('Alpha', 8, 12),
#     ('Beta', 13, 25),
#     ('Gamma', 30, 45)
]

tmin, tmax = -0.2, 1.2

# %%
n_jobs = 48
_svm = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced')
clf = make_pipeline(StandardScaler(), _svm)
all_decoder = make_pipeline(mne.decoding.Vectorizer(), clf)
time_decoder = mne.decoding.SlidingEstimator(clf, n_jobs=n_jobs, scoring='f1')
skf = StratifiedKFold(n_splits=10)
xdawn = mne.preprocessing.Xdawn(n_components=6, reg='diagonal_fixed')

def perform_MVPA(epochs, all_decoder=all_decoder, time_decoder=time_decoder, CV=skf, xdawn=False):
    # MVPA method for epochs
    # all_decoder: classification pipeline,
    #              when fit, receive 3-dim dataset and label,
    #              when predict, receive 3-dim dataset, return predicted label
    # time_decoder: a time-resolution version of all_decoder
    # CV: cross-validation generator
    # xdawn: xdawn denoise model, xdawn is invalid when xdawn is False

    # Get X, y
    X = epochs.get_data()
    events = epochs.events
    y = events[:, -1]
    print(X.shape, y.shape)

    # Prepare predicts
    y_predict = np.zeros(X.shape[0])
    y_time_predict = np.zeros((X.shape[0], X.shape[2]))
    print(y_predict.shape, y_time_predict.shape)

    # Cross validation
    # Split train and test
    for train, test in CV.split(X, y):
        # Make train and test data
        if xdawn:
            epochs.baseline = None
            # Fit xdawn
            xdawn.fit(epochs[train])
            # Transoform using xdawn
            X_train, y_train = xdawn.apply(epochs[train])['1'].get_data(), y[train]
            X_test, y_test = xdawn.apply(epochs[test])['1'].get_data(), y[test]
        else:
            # Seperate train and test
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]

        # All time train and test
        all_decoder.fit(X_train, y_train)
        y_predict[test] = all_decoder.predict(X_test)
        # Window time train and test
        time_decoder.fit(X_train, y_train)
        y_time_predict[test, :] = time_decoder.predict(X_test)

    # Return
    return dict(
        y_true = y,
        y_predict = y_predict,
        y_time_predict = y_time_predict,
        times = epochs.times,
    )

def report_MVPA(predicts):
    y_true = predicts['y_true']
    y_pred = predicts['y_predict']
    y_time_pred = predicts['y_time_predict']
    # All time report
    print(sklearn.metrics.classification_report(y_pred=y_pred, y_true=y_true))
    print(sklearn.metrics.precision_score(y_pred=y_pred, y_true=y_true, average='weighted'))
    # Window time report
    scores = np.zeros(y_time_pred.shape[1])
    for j, y_pred in enumerate(y_time_pred.transpose()):
        scores[j] = sklearn.metrics.precision_score(y_pred=y_pred, y_true=y_true, average='weighted')

# %%
regexp = 'MEG_S02'

for regexp in ['MEG_S01', 'MEG_S02', 'EEG_S01', 'EEG_S02', 'EEG_S03', 'EEG_S04', 'EEG_S06', 'EEG_S07', 'EEG_S09']:
    MVPA_results = defaultdict(list)

    # Read and concatenate raw data
    fnames = [os.path.join(e, 'ica_denoised-raw.fif') for e in path_table.loc[[e for e in path_table.index if regexp in e]]['processed_path']]
    rawraw = mne.concatenate_raws([mne.io.read_raw_fif(e) for e in fnames])

    # Filter events
    events = False
    if regexp.startswith('MEG'):
        picks = 'mag'
        events = mne.find_events(rawraw, stim_channel='UPPT001')
    if regexp.startswith('EEG'):
        picks = 'eeg'
        events = mne.events_from_annotations(rawraw)[0]
    assert(events is not False)

    sfreq = rawraw.info['sfreq']
    with tqdm.tqdm(events[events[:, -1]==1]) as pbar:
        for e in pbar:
            pbar.set_description('Index: {}'.format(e[0]))
            for ee in events:
                if all([ee[-1] == 2, abs(ee[0]-e[0])<sfreq]):
                    ee[-1] = 4

    for freqs in iter_freqs:
        print('-' * 80)
        print(freqs)
        name_freq, l_freq, h_freq = freqs

        # Get epochs
        # Raw epochs
        epochs = mne.Epochs(rawraw.copy(), events, picks=picks, tmin=tmin, tmax=tmax, decim=10)
        # Clear events
        epochs = epochs[['1', '2']]
        epochs.load_data()
        # Filter epochs
        epochs.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

        # perform MVPA
        # MVPA_predicts = perform_MVPA(epochs.copy())
        # Report MVPA
        # report_MVPA(MVPA_predicts)
        # MVPA_results[name_freq].append(('origin', MVPA_predicts))

        # perform xdawn MVPA
        xdawn_MVPA_predicts = perform_MVPA(epochs.copy(), xdawn=xdawn)
        # Report xdawn MVPA
        report_MVPA(xdawn_MVPA_predicts)
        MVPA_results[name_freq].append(('xdawn', xdawn_MVPA_predicts))

    df = pd.DataFrame(MVPA_results)

    df.to_json('MVPA_predicts_{}.json'.format(regexp))

