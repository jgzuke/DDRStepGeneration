
# coding: utf-8

# In[1]:
from __future__ import print_function
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
from os import listdir
from os.path import isfile, join
from numpy import median, diff


# # SongFile:
# ### Fields
# - beat_frames:  
# - beat_times: 
# - bpm: 
# - bpm_string: 
# - beat_length: 
# - indices: 
# - data: 
# 
# - pack
# - name
# - extension
# - music_file
# - stepfile
# ### Output
# - data/{0}_beat_features.csv
# - data/{0}_misc.csv

# In[12]:

# started 12:19

sample_rate_down = 1
hop_length_down = 8
sr = 11025 * 16 / sample_rate_down
hop_length = 512 / (sample_rate_down * hop_length_down)
samples_per_beat = 48 / 4
steps_per_bar = 48
class SongFile:
    # misc includes
    # - offset
    # - bpm
    def load_misc_from_music(self, y):
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        self.offset = beat_times[0]
        self.bpm = get_beats(beat_times, beat_frames)

    def load_misc_from_stepfile(self):
        with open(self.stepfile, "r") as txt:
            step_file = txt.read()
            step_file = step_file.replace('\n', 'n')
            bpm_search = re.search('#BPMS:([0-9.=,]*);', step_file)
            bpm_string = bpm_search.group(1)
            self.bpm = float(bpm_string.split('=')[1]) if len(bpm_string.split(',')) == 1 else 0
            
            offset_search = re.search('#OFFSET:([0-9.-]*);', step_file)
            self.offset = -float(offset_search.group(1))

    def calculate_indices(self, y):
        # take samples_per_beat samples for each beat (need 3rds, 8ths)
        seconds = len(y) / sr
        num_samples = int(seconds * samples_per_beat * self.bpm / 60)
        beat_length = 60. / self.bpm
        sample_length = beat_length / samples_per_beat
        
        if self.offset < 0:
            self.offset += 4 * beat_length
        
        sample_times = [self.offset + (sample_length * i) for i in range(num_samples)]
        # only take samples where music still playing
        self.indices = [round(time * sr) for time in sample_times if round(time * sr) < len(y)]
        
    def calculate_features(self, y):
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        beat_frames = librosa.samples_to_frames(self.indices)

        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

        # And the first-order differences (delta features)
        mfcc_delta = librosa.feature.delta(mfcc)

        # Stack and synchronize between beat events
        # This time, we'll use the mean value (default) instead of median
        beat_mfcc_delta = librosa.feature.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # Aggregate chroma features between beat events
        # We'll use the median value of each feature between beat frames
        beat_chroma = librosa.feature.sync(chromagram, beat_frames, aggregate=np.median)

        custom_hop = 256
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=custom_hop)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env, hop_length=custom_hop)

        i = 0
        onset_happened_in_frame = [0] * (len(self.indices) + 1)
        for onset in onsets:
            onset_scaled = onset * custom_hop
            while abs(onset_scaled - self.indices[i]) > abs(onset_scaled - self.indices[i + 1]):
                i += 1
            onset_happened_in_frame[i] = max(onset_env[onset], onset_env[onset + 1], onset_env[onset + 2], onset_env[onset + 3], onset_env[onset + 4])

        indices = [0]
        indices.extend(self.indices)
        max_offset_bounds = [(int(indices[i] / custom_hop), int(indices[i + 1] / custom_hop)) for i in range(len(indices) - 1)]
        max_offset_strengths = [max(onset_env[bounds[0]:bounds[1]]) for bounds in max_offset_bounds]
        max_offset_strengths.append(0)

        # Finally, stack all beat-synchronous features together
        self.beat_features = np.vstack([beat_chroma, beat_mfcc_delta, [onset_happened_in_frame, max_offset_strengths]])

    def __init__(self, pack, name, load_type):
        self.name = name
        self.folder = 'StepMania/Songs/{0}/{1}/'.format(pack, name)
        self.music_file = self.folder + next(file for file in listdir(self.folder) if file.endswith('.ogg') or file.endswith('.mp3'))
        self.stepfile = self.folder + next(file for file in listdir(self.folder) if file.endswith('.ssc') or file.endswith('.sm'))
        key = '{0}~{1}'.format(pack, name)
        y = None
        print ('Loading song {0}'.format(key))
        
        if load_type == 'from_music' or load_type == 'from_stepfile':
            if load_type == 'from_music':
                print ('Loading music')
                y, _ = librosa.load(self.music_file, sr=sr)
                print ('Calculating misc from music')
                self.load_misc_from_music(y)
            else:
                print ('Loading misc from stepfile')
                self.load_misc_from_stepfile()
                if self.bpm == 0:
                    raise Exception('Inconsistent bpm')
                print ('Loading music')
                y, _ = librosa.load(self.music_file, sr=sr)
                

            print ('Calculating indices')
            self.calculate_indices(y)
            print ('Calculating features')
            self.calculate_features(y)
            print ('Saving song\n')
            pd.DataFrame([self.offset, self.bpm]).to_csv('data/{0}_misc.csv'.format(key), index=False)
            pd.DataFrame(self.beat_features).to_csv('data/{0}_beat_features.csv'.format(key), index=False)
            
        if load_type == 'from_store':
            if not '{0}_beat_features.csv'.format(key) in listdir('data'):
                print ('Song hasnt been loaded yet')
            else:
                [self.offset], [self.bpm] = pd.read_csv('data/{0}_misc.csv'.format(key)).values
                self.beat_features = pd.read_csv('data/{0}_beat_features.csv'.format(key)).values


# # Some useful functions to load induvidual or lists of songs
# - load_song(pack: String, pack: String, force_new: Bool)
# - load_songs(songs: Array(Pair(String~pack, String~title)), force_new: Bool)
# - load_all_songs(force_new: Bool)

# In[3]:

def load_songs(songs, load_type):
    return {'{0}~{1}'.format(song[0], song[1]): SongFile(song[0], song[1], load_type) for song in songs}

def load_all_songs(load_type):
    songs = [('In The Groove', song) for song in listdir('StepMania/Songs/In The Groove') if song != '.DS_Store']
    songs.extend([('a_test', song) for song in ['A', 'B', 'C']])
    return load_songs(songs, load_type)


# # Functions to get bpm from song
# - get_beats(beat_times: Array(Float), beat_frames: Array(Int))

# In[4]:

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
    #print (new_averages)
    new_averages.sort()
    num_averages = len(new_averages)
    #new_average = sum(new_averages[5:num_averages - 5]) / (num_averages - 10)
    new_average = new_averages[int(num_averages/2)]
    bpm = 60./new_average
    while bpm >= 200:
        bpm /= 2
    while bpm < 100:
        bpm *= 2
    return bpm

#songs = [('In The Groove', song) for song in listdir('StepMania/Songs/In The Groove') if song != '.DS_Store'][:15]
#song_data_temp = load_songs(songs, True)
#test_get_beats(song_data_temp)


# In[15]:
def generate_all():
    packs = ['In The Groove', 'In The Groove 2', 'In The Groove 3', 'In The Groove Rebirth', 'In The Groove Rebirth +', 'In The Groove Rebirth 2 (BETA)', 'Piece of Cake', 'Piece of Cake 2', 'Piece of Cake 3', 'Piece of Cake 4', 'Piece of Cake 5']
    for pack in packs:
        songs = [(pack, song) for song in listdir('StepMania/Songs/{0}'.format(pack)) if song != '.DS_Store']
        for song in songs:
            gc.collect()
            try:
                if '{0}~{1}_beat_features.csv'.format(song[0], song[1]) in listdir('data'):
                    print ('Song Already Loaded')
                else:
                    SongFile(song[0], song[1], 'from_stepfile')
            except:
                print ('Error loading song\n') 
            gc.collect()


# In[ ]:



