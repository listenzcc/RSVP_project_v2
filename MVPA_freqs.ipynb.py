# %%
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(__doc__)

# %%
import sklearn
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# %%
n_jobs = 64
_svm = svm.SVC(gamma='scale', kernel='rbf', class_weight='balanced')
clf = make_pipeline(StandardScaler(), _svm)
raw_decoder = make_pipeline(mne.decoding.Vectorizer(), clf)
time_decoder = mne.decoding.SlidingEstimator(clf, n_jobs=n_jobs, scoring='f1')
skf = StratifiedKFold(n_splits=10)
xdawn = mne.preprocessing.Xdawn(n_components=6, reg='diagonal_fixed')

# %%
fifs = pd.read_json('fifs.json')
fifs

# %%
iter_freqs = [
    ('Delta', 0, 4),
    ('Theta', 4, 7),
#     ('Alpha', 8, 12),
#     ('Beta', 13, 25),
#     ('Gamma', 30, 45)
]

# %%
tmin, tmax = -0.2, 1.2
picks = 'mag'
rawraw = mne.concatenate_raws([mne.io.read_raw_fif(fifs.loc['S02_R{:02d}'.format(j)]['icapath']) for j in range(4, 12)])
results = dict()
for freqs in iter_freqs:
    print('-' * 80)
    print(freqs)
    name_freq, l_freq, h_freq = freqs

    # Load raw
    raw = rawraw.copy()
    raw.load_data()

    # Filter
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Get epochs
    # Raw epochs
    epochs = mne.Epochs(raw, mne.find_events(raw, stim_channel='UPPT001'), picks=picks, tmin=tmin, tmax=tmax, decim=10)
    # Clear events
    epochs = epochs[['1', '2']]

    event_id = epochs.event_id
    # Get and plot events
    events = epochs.events
    # mne.viz.plot_events(events, sfreq=raw.info['sfreq'])
    print(event_id)
    # Plot epochs
    # for event in event_id:
    #     print(event)
    #     epochs[event].average().plot(spatial_colors=True)

    # MVPA raw
    # Get X, y
    X = epochs.get_data()
    y = events[:, -1]
    print(X.shape, y.shape)
    # Prepare predicts
    y_predict = np.zeros(X.shape[0])
    y_time_predict = np.zeros((X.shape[0], X.shape[2]))
    print(y_predict.shape, y_time_predict.shape)
    # Cross validation
    for train, test in skf.split(X, y):
        # Split train and test
        print('.')
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        # All time train and test
        print('..')
        raw_decoder.fit(X_train, y_train)
        y_predict[test] = raw_decoder.predict(X_test)
        # Window time train and test
        print('...')
        time_decoder.fit(X_train, y_train)
        y_time_predict[test, :] = time_decoder.predict(X_test)
    # Reports
    # All time report
    print(sklearn.metrics.classification_report(y_pred=y_predict, y_true=y))
    print(sklearn.metrics.precision_score(y_pred=y_predict, y_true=y, average='weighted'))
    # Window time report
    scores = np.zeros(y_time_predict.shape[1])
    for j, y_pred in enumerate(y_time_predict.transpose()):
        scores[j] = sklearn.metrics.precision_score(y_pred=y_pred, y_true=y, average='weighted')
    # plt.plot(scores)

    # MVPA xdawn
    epochs.baseline = None
    # Get X, y
    X = epochs.get_data()
    y = events[:, -1]
    print(X.shape, y.shape)
    # Prepare predicts
    xdawn_y_predict = np.zeros(X.shape[0])
    xdawn_y_time_predict = np.zeros((X.shape[0], X.shape[2]))
    print(xdawn_y_predict.shape, xdawn_y_time_predict.shape)
    # Cross validation
    for train, test in skf.split(X, y):
        # Split train and test
        # Transoform using xdawn
        X_train, y_train = xdawn.fit_transform(epochs[train]), y[train]
        X_test, y_test = xdawn.transform(epochs[test]), y[test]
        # All time train and test
        raw_decoder.fit(X_train, y_train)
        xdawn_y_predict[test] = raw_decoder.predict(X_test)
        # Window time train and test
        time_decoder.fit(X_train, y_train)
        xdawn_y_time_predict[test, :] = time_decoder.predict(X_test)
    # Reports
    # All time report
    print(sklearn.metrics.classification_report(y_pred=xdawn_y_predict, y_true=y))
    print(sklearn.metrics.precision_score(y_pred=xdawn_y_predict, y_true=y, average='weighted'))
    # Window time report
    xdawn_scores = np.zeros(xdawn_y_time_predict.shape[1])
    for j, y_pred in enumerate(xdawn_y_time_predict.transpose()):
        xdawn_scores[j] = sklearn.metrics.precision_score(y_pred=y_pred, y_true=y, average='weighted')
    # plt.plot(xdawn_scores)

    # Save results
    results[name_freq] = dict(
        y_true = y,
        y_pred = y_predict,
        y_time_pred = y_time_predict,
        xdawn_y_pred = xdawn_y_predict,
        xdawn_y_time_pred = xdawn_y_time_predict,
    )

# %%
import pandas as pd
df = pd.DataFrame(results)
df = df.T
df.to_json('MVPAresults.json')
df
