
# coding: utf-8

# In[1]:

import copy
import pandas as pd
import numpy as np
import librosa
import seaborn as sb
import matplotlib.pyplot as plt
import itertools
import re
import random
import gc
from os import listdir, rename
from os.path import isfile, join
from numpy import median, diff
from operator import itemgetter, attrgetter, methodcaller
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# # Get beat features

# In[2]:

steps_per_bar = 48


# In[3]:

sample_rate_down = 1
hop_length_down = 8
sr = 11025 * 16 / sample_rate_down
hop_length = 512 / (sample_rate_down * hop_length_down)
samples_per_beat = steps_per_bar / 4

def load_misc_from_music(y):
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return (beat_times[0], get_beats(beat_times, beat_frames))

def get_beats(beat_times, beat_frames):
    changes = []
    changes_time = []
    for i in range(len(beat_frames) - 1):
        changes.append(beat_frames[i + 1] - beat_frames[i])
        changes_time.append(beat_times[i + 1] - beat_times[i])

    sorted_changes = sorted(changes)
    median = sorted_changes[int(len(changes) / 2)]
    median = max(set(sorted_changes), key=sorted_changes.count)

    changes_counted = [False] * len(changes)
    time_changes_sum = 0
    time_changes_count = 0
    for i in range(len(changes)):
        # can use other factors (eg if song has a slow part take double beats into accout)
        # in [0.5, 1, 2]:
        for change_factor in [1]:
            if abs((changes[i] * change_factor) - median) <= hop_length_down:
                changes_counted[i] = True
                time_changes_sum += (changes_time[i] * change_factor)
                time_changes_count += change_factor
            
    average = time_changes_sum / time_changes_count
    
    time_differences = []
    earliest_proper_beat = 1
    for i in range(1, len(beat_times) - 1):
        if changes_counted[i] & changes_counted[i - 1]:
            earliest_proper_beat = i
            break
            
    last_proper_beat = len(beat_times) -2
    for i in range(1, len(beat_times) - 1):
        if changes_counted[len(beat_times) - i - 1] & changes_counted[len(beat_times) - i - 2]:
            last_proper_beat = len(beat_times) - i - 1
            break
    
    time_differences = []
    buffer = 5
    for i in range(20):
        start_beat = earliest_proper_beat + buffer * i
        if changes_counted[start_beat] & changes_counted[start_beat - 1]:
            for j in range(20):
                end_beat = last_proper_beat - buffer * j
                if changes_counted[end_beat] & changes_counted[end_beat - 1]:
                    time_differences.append(beat_times[end_beat] - beat_times[start_beat])
        
    # get num beats, round, and make new average
    new_averages = [time_difference / round(time_difference / average) for time_difference in time_differences]
    new_averages.sort()
    num_averages = len(new_averages)
    new_average = new_averages[int(num_averages/2)]
    bpm = 60./new_average
    while bpm >= 200:
        bpm /= 2
    while bpm < 100:
        bpm *= 2
    return bpm

def calculate_indices(offset, bpm, y):
    # take samples_per_beat samples for each beat (need 3rds, 8ths)
    seconds = len(y) / sr
    num_samples = int(seconds * samples_per_beat * bpm / 60)
    beat_length = 60. / bpm
    sample_length = beat_length / samples_per_beat

    if offset < 0:
        offset += 4 * beat_length

    sample_times = [offset + (sample_length * i) for i in range(num_samples)]
    # only take samples where music still playing
    return [round(time * sr) for time in sample_times if round(time * sr) < len(y)]

def calculate_features(indices, y):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    beat_frames = librosa.samples_to_frames(indices)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    beat_mfcc_delta = librosa.feature.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)

    custom_hop = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=custom_hop)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env, hop_length=custom_hop)

    i = 0
    onset_happened_in_frame = [0] * (len(indices) + 1)
    for onset in onsets:
        onset_scaled = onset * custom_hop
        while abs(onset_scaled - indices[i]) > abs(onset_scaled - indices[i + 1]):
            i += 1
        onset_happened_in_frame[i] = max(onset_env[onset], onset_env[onset + 1], onset_env[onset + 2], onset_env[onset + 3], onset_env[onset + 4])

    indices = [0]
    indices.extend(indices)
    max_offset_bounds = [(int(indices[i] / custom_hop), int(indices[i + 1] / custom_hop)) for i in range(len(indices) - 1)]
    max_offset_strengths = [max(onset_env[bounds[0]:bounds[1]]) for bounds in max_offset_bounds]
    max_offset_strengths.append(0)

    return np.vstack([beat_chroma, beat_mfcc_delta, [onset_happened_in_frame, max_offset_strengths]])


# # Get beat importance

# In[4]:

samples_back_included = 8
num_features = 40
def get_features_for_index(beat_features, index):
    if index < 0:
        return [0] * num_features
    return beat_features[index]
    
importance_rankings = [48, 24, 12, 16, 6, 8, 3, 4, 2, 1]
def get_beat_importance(index):
    for i in range(len(importance_rankings)):
        if index % importance_rankings[i] == 0:
            return i

def get_features_for_song(key, beat_features_rotated):
    X = []
    y = []
    beat_features = np.flipud(np.rot90(np.array(beat_features_rotated)))
    for i in range(len(beat_features)):
        features = [feature for j in range(samples_back_included) for feature in get_features_for_index(beat_features, i - j)]
        features.append(i % 48)
        features.append(get_beat_importance(i))
        features.append(i / 48)
        features.append(len(beat_features) - i / 48)
        X.append(features)
    return np.array(X)


# # Get song output

# In[5]:

note_types = ['0', '1', 'M', '2', '4', '3']
def get_features_for_row(row):
    return [int(char == target) for target in note_types for char in row]

empty_row = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def get_previous_notes(index, features):
    previous_notes = [features[i] for i in range(index, index + song_padding) if not np.array_equal(features[i], empty_row)]
    return [empty_row] * (8 - len(previous_notes)) + previous_notes[-8:]
    
song_padding = 96
song_end_padding = 96
important_indices = [1, 2, 3, 4, 8, 16, 20, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
important_indices_classes = [-96, -84, -72, -60, -48, -36, -24, -12, 0, 1, 2, 3, 4, 8, 16, 20, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
def get_features(index, features, note_classes):
    indices = [index + song_padding - i for i in important_indices]
    indices_classes = [index + song_padding - i for i in important_indices_classes]
    past_classes = np.array([note_classes[i] for i in indices_classes]).flatten()
    past_features = np.array([features[i] for i in indices]).flatten()
    previous_notes = np.array(get_previous_notes(index, features)).flatten()
    return np.concatenate((past_classes, past_features, previous_notes), axis = 0)

def get_model_class_for_notes(row):
    note_counts = [row.count(note_type) for note_type in note_types]
    (blank, steps, mines, hold_starts, roll_starts, hold_ends) = note_counts
    
    model_classes = []
    if steps + hold_starts + roll_starts == 1:
        model_classes.append(1)

    if steps + hold_starts + roll_starts == 2:
        model_classes.append(2)
        
    if steps + hold_starts + roll_starts > 2:
        model_classes.append(3)
        
    if hold_starts > 0:
        model_classes.append(4)
        
    if roll_starts > 0:
        model_classes.append(5)
        
    if mines > 0:
        model_classes.append(6)
        
    return model_classes

def get_model_output_for_class(model_class, row):
    if model_class == 1 or model_class == 2 or model_class == 3:
        return [int(char == '1' or char == '2' or char == '4') for char in row]
    if model_class == 4:
        return [int(char == '2') for char in row]
    if model_class == 5:
        return [int(char == '4') for char in row]
    if model_class == 6:
        return [int(char == 'M') for char in row]

def get_hold_length(notes, note_row, note_column):
    i = 0
    while i < len(notes) - note_row:
        if notes[note_row + i][0][note_column] == '3':
            return i
        i += 1
    return False

surrounding_beat_indices = [i for i in range(-24, 25)]#[-48, -36, -24, -12, 12, 24, 36, 48]
def get_average_for_class(note_classes, i, class_num):
    surrounding_beats = [note_classes[i] for i in surrounding_beat_indices if i > -1 and i < len(note_classes)]
    return sum([beat[class_num] for beat in surrounding_beats]) / float(len(surrounding_beats))

def normalize_row(note_classes, i):
    return [note_classes[i][class_num] / get_average_for_class(note_classes, i, class_num) for class_num in range(7)]
    
def normalize_classes(note_classes):
    return [normalize_row(note_classes, i) for i in range(len(note_classes))]

def replace_char(prediction, i, new_char):
    return prediction[:i] + new_char + prediction[i+1:]

hold_lengths = [3, 6, 9, 12, 18, 24, 36, 48]
pattern = ['1000', '0100', '0001', '0010', '0100', '1000', '0001', '0010', '1000', '0100', '0001', '0010', '0100', '1000', '0001', '0010']
cutoff_per_class = [0, 0.85, 0.95, 1, 0.98, 1, 0.98]
def get_output(note_classes):
    hold_lengths_current = [0, 0, 0, 0]
    roll_lengths_current = [0, 0, 0, 0]
    hold_lengths_max = [12, 12, 12, 12]
    roll_lengths_max = [12, 12, 12, 12]
    predicted_notes = []
    # TODO: figure out better normilazation
    normalized_note_classes = note_classes #normalize_classes(note_classes)
    sortedLists = [sorted(normalized_note_classes, key=itemgetter(i)) for i in range(7)]
    num_samples = len(note_classes)
    cutoffs = [sortedLists[i][min(int(num_samples * cutoff_per_class[i]), len(sortedLists[i]) - 1)][i] for i in range(7)]
    
    note_classes = np.concatenate((([[1, 0, 0, 0, 0, 0, 0]] * song_padding), note_classes, ([[1, 0, 0, 0, 0, 0, 0]] * song_end_padding)), axis = 0)
    dummy_rows = [row for eigth in pattern for row in [eigth] + ['0000'] * 5]
    features = [get_features_for_row(row) for row in dummy_rows]
    for i in range(num_samples):
        note_class = note_classes[i]
        normalized_note_class = normalized_note_classes[i]
        prediction = '0000'
        X_row = get_features(len(features) - song_padding, features, note_classes)
        # order by reverse importance of decision
        # TODO up limit if something has been bumped out of existance (eg put more jumps if they get covered by holds)
        targets = ['0', '1', '1', '1', '2', '4', 'M']
        ammounts = [0, 1, 2, 3, 1, 1, 1]
        for i in [1, 6, 2, 5, 4, 3]:
            if normalized_note_class[i] > cutoffs[i]:
                prediction_values = models[i].predict(np.array([X_row]))[0]
                prediction_values = [value + random.uniform(-0.01, 0.01) for value in prediction_values]
                number_to_include = ammounts[i]
                for j in range(4):
                    if hold_lengths_current[j] > 0 or roll_lengths_current[j] > 0:
                        number_to_include -= 1
                cutoff = sorted(prediction_values)[-(max(0, number_to_include) + 1)]
                prediction = ''.join([targets[i] if value > cutoff else '0' for value in prediction_values])
        
        for i in range(4):
            if hold_lengths_current[i] > 0:
                hold_lengths_current[i] += 1
                if hold_lengths_current[i] == hold_lengths_max[i]:
                    prediction = replace_char(prediction, i, '3')
                    hold_lengths_current[i] = 0
                else:
                    prediction = replace_char(prediction, i, '0')
            if roll_lengths_current[i] > 0:
                roll_lengths_current[i] += 1
                if roll_length_currents[i] == roll_lengths_max[i]:
                    prediction = replace_char(prediction, i, '3')
                    roll_lengths_current[i] = 0
                else:
                    prediction = replace_char(prediction, i, '0')
            if prediction[i] == '2':
                hold_lengths_index = np.argmax(hold_model.predict(np.array([X_row]))[0])
                if hold_lengths_index == 4 or hold_lengths_index == 5 or hold_lengths_index == 6:
                    hold_lengths_index = 3
                hold_lengths_current[i] = 1
                hold_lengths_max[i] = hold_lengths[hold_lengths_index]
            if prediction[i] == '4':
                hold_lengths_index = np.argmax(hold_model.predict(np.array([X_row]))[0])
                if hold_lengths_index == 4 or hold_lengths_index == 5 or hold_lengths_index == 6:
                    hold_lengths_index = 3
                roll_lengths_current[i] = 1
                roll_lengths_max[i] = hold_lengths[hold_lengths_index]

        predicted_notes.append(prediction)
        features.append(get_features_for_row(prediction))
    return predicted_notes


# # Write file

# In[6]:

def write_song_metadata(output_stepfile, song, music_file, offset, bpm):
    keys = ['TITLE', 'MUSIC', 'OFFSET', 'SAMPLESTART', 'SAMPLELENGTH', 'SELECTABLE', 'BPMS']
    header_info = {
        'TITLE': song,
        'MUSIC': music_file,
        'OFFSET': -offset,
        'SAMPLESTART': offset + 32 * (60. / bpm),
        'SAMPLELENGTH': 32 * (60. / bpm),
        'SELECTABLE': 'YES',
        'BPMS': '0.000={:.3f}'.format(bpm)
    }
    
    for key in keys:
        print ("#{0}:{1};".format(key, str(header_info[key])), file=output_stepfile)
        
def write_song_steps(output_stepfile, predicted_notes):
    print("\n//---------------dance-single - J. Zukewich----------------", file=output_stepfile)
    print ("#NOTES:", file=output_stepfile)
    for detail in ['dance-single', 'J. Zukewich', 'Expert', '9', '0.242,0.312,0.204,0.000,0.000']:
        print ('\t{0}:'.format(detail), file=output_stepfile)
    
    for i in range(len(predicted_notes)):
        row = predicted_notes[i]
        print (row, file=output_stepfile)
        if i % steps_per_bar == steps_per_bar - 1:
            print (",", file=output_stepfile)

    print ("0000;", file=output_stepfile)


# # Step songs

# In[12]:

def step_song_from_music_file(key):
    pack, song = key.split('~')
    folder = 'StepMania/Songs/{0}/{1}/'.format(pack, song)
    music_file = [file for file in listdir(folder) if file.endswith('.ogg') or file.endswith('.mp3')][0]
    stepfile_name = 'StepMania/Songs/{0}/{1}/{1}.sm'.format(pack, song)
    
    print ('Loading Song {0}'.format(song))
    y, _ = librosa.load(folder + music_file, sr=sr)
        
    print ('Calculating BPM')
    offset, bpm = load_misc_from_music(y)
    #pd.DataFrame([offset, bpm]).to_csv('prod_data/{0}_misc.csv'.format(key), index=False)

    print ('Calculating Features')
    indices = calculate_indices(offset, bpm, y)
    beat_features = calculate_features(indices, y)
    #pd.DataFrame(beat_features).to_csv('prod_data/{0}_beat_features.csv'.format(key), index=False)
    y = None
    indices = None
    
    print ('Getting Song Predicted Classes')
    X = get_features_for_song(key, beat_features)
    note_classes = beat_feature_model.predict(X)
    #pd.DataFrame(note_classes).to_csv('prod_data/{0}_note_classes_generated.csv'.format(key), index=False)
    beat_features = None
    X = None
    
    
    print ('Stepping Song')
    predicted_notes = get_output(note_classes)
    #pd.DataFrame(predicted_notes).to_csv('prod_data/{0}_predicted_notes.csv'.format(key), index=False)
    note_classes = None
    
    print ('Writing Song to File')
    stepfile=open(stepfile_name, 'w')
    write_song_metadata(stepfile, song, music_file, offset, bpm)
    write_song_steps(stepfile, predicted_notes)
    stepfile.close()
    
    print ('Done')


# # Load models

# In[8]:

beat_feature_model = load_model('models/beat_importance_model_classes_full_testing.h5')
# beat_importance_model_classes_full_testing_with_classes


# In[9]:

models = [None] + [load_model('models/write_notes_model_full{0}.h5'.format(i)) for i in range(1, 7)]


# In[10]:

hold_model = load_model('models/hold_length_model.h5')
roll_model = load_model('models/roll_length_model.h5')


# In[ ]:

step_song_from_music_file('a_test~WeBelongTogether')
#step_song_from_music_file('WeBelongTogether')
#step_song_from_music_file('CallMeBaby')


# In[ ]:

step_song_from_music_file('a_test~CallMeBaby')


# In[ ]:

step_song_from_music_file('a_test~Fire')


# In[ ]:

step_song_from_music_file('a_test~View')


# In[ ]:



