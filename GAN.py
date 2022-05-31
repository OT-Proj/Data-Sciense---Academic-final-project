import Consts
import torch
from joblib import dump, load
import pandas as pd
from random import randrange
import pretty_midi
from datetime import datetime
import random
from MyDataset import MyDataset
import pickle
import numpy as np

class GAN:
    def __init__(self, generator_dat_path, discriminator_dat_path, scaler_dat_path, x_path, y_path):
        self.generator = torch.load(generator_dat_path)
        #self.discriminator = torch.load(discriminator_dat_path)
        self.generator.eval()
        #self.discriminator.eval()
        self.scaler = load(scaler_dat_path)


        x_real = pickle.load(open(x_path, "rb"))
        y_real = pickle.load(open(y_path, "rb"))

        num_songs = x_real.shape[0]
        train_percentage = 0.9

        x_real = self.scaler.fit_transform(x_real)
        x_domain = x_real[int(train_percentage * num_songs):]
        y_domain = y_real[int(train_percentage * num_songs):]
        x_domain = torch.from_numpy(x_domain)
        y_domain = torch.from_numpy(y_domain)

        self.domain_dataset = MyDataset(x_domain,y_domain)
        self.uniform_range = 1

    def generate(self):
        seed = self.domain_dataset.getRandomX()
        noise = np.random.uniform(low=-1*self.uniform_range, high=self.uniform_range, size=seed.shape)
        noise_tensor = torch.from_numpy(noise).float()
        seed = seed + noise_tensor
        generated = self.generator(seed).detach().numpy()
        generated = self.scaler.inverse_transform(generated)
        gan_result = pd.DataFrame(generated.reshape(-1, Consts.num_features), columns=Consts.columns)
        start_time_filtered_df = gan_result[gan_result["start"] >= 0]
        duration_filtered_df = start_time_filtered_df[start_time_filtered_df["duration"] > 0]

        now = datetime.now()
        filename = "tmp/" + now.strftime("%m%d%Y%H%M%S") + str(random.uniform(0, 1))[2:] + ".mid"

        self.pandasToMIDI(duration_filtered_df, round_digits=3, quantize=True, soft_fix_notes=True,
                    transpose_by=randrange(0, 11), prevent_overlap=True,
                    filename="static/" + filename, instrument=1)
        return filename

    def pandasToMIDI(self, df_input, instrument=0, filename="/content/drive/MyDrive/GAN out/tmp file garbage.mid",
                     round_digits=5, quantize=False, soft_fix_notes=False, transpose_by=0, prevent_overlap=False):
        midi_file = pretty_midi.PrettyMIDI()
        finalized_notes = [] # returns output notes
        midi_channel = pretty_midi.Instrument(program=instrument)
        min_time = round(min(abs(df_input["start"])), round_digits)
        last_endtime = {}
        if (min_time < 0):
            min_time *= -1
        else:
            min_time = 0
        num_notes = len(df_input)
        written = 0
        for row in df_input.iterrows():
            # ensure duration is valid
            duration = round(row[1]["duration"], round_digits)
            if (duration < 0):
                duration *= -1
            if (duration > 10):
                duration = 10
            start = -min_time + abs(round(row[1]["start"], round_digits))

            # quantize start times
            if (quantize):
                div = int(start / Consts.quantize_measure)
                mod = start / Consts.quantize_measure - int(start / Consts.quantize_measure)
                start = start - mod * Consts.quantize_measure

            # soft-move notes to scale
            if (round(row[1]["note"]) < 1 or round(row[1]["note"]) > 127):
                print("Warning: Integrity issue: %d" % round(row[1]["note"]))
            cur_note = round(row[1]["note"])
            remainder = row[1]["note"] - cur_note
            if (soft_fix_notes):
                # if the current note is not in-scale, but rounded up/down is - convert it. Otherwise - leave note as is.
                if (remainder >= 0.5 and not cur_note % 12 in Consts.scaleA_notes and (cur_note + 1) % 12 in Consts.scaleA_notes):
                    cur_note = cur_note + 1
                if (remainder < 0.5 and not cur_note % 12 in Consts.scaleA_notes and (cur_note - 1) % 12 in Consts.scaleA_notes):
                    cur_note = cur_note - 1

            cur_note = cur_note + transpose_by

            if (row[1]["is_bass"] > 0.09):
                cur_note -= 12 # bass is one octave lower

            if (row[1]["is_pipe"] > 0.09):
                cur_note += 12 # bass is one octave lower


            if (cur_note >= 0 and cur_note <= 127):
                finalized_notes.append(cur_note)
                write_note = True
                if cur_note in last_endtime.keys():
                    if last_endtime[cur_note] > start: # we allow overlap only if the note is about to end
                        write_note = False
                last_endtime[cur_note] = start + duration

                if write_note or not prevent_overlap:
                    note = pretty_midi.Note(velocity=randrange(65, 75), pitch=cur_note, start=start, end=start + duration)
                    midi_channel.notes.append(note)
                    written += 1

        midi_file.instruments.append(midi_channel)
        midi_file.write(filename)
        print("File created: %s" % filename)
        print("Written %d out of %d" %(written, num_notes))
        return finalized_notes