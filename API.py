import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pretty_midi
import collections
from torch.utils import data
import os
import random
import math
import copy
import sys
import music21 as m21
from random import randrange


class JetBit:
    def __init__(self):
        self.transpose_table = {"A-": 1, "A": 0, "B-": -1, "B": -2, "C": -3, "C#": -4, "D-": -4, "D": -5, "E-": -6,
                                "E": 5, "F": 4, "F#": 3, "G-": 3, "G": 2, "G#": 1}

    def extractMessages(self, midi_file: str, features_to_drop=[], base_bpm=240, normalize_time=False,
                        normalize_scale=False):
        pm = pretty_midi.PrettyMIDI(midi_file)
        estimated_bpm = pm.estimate_tempo()
        bpm_ratio = estimated_bpm / base_bpm
        if (not normalize_time):
            bpm_ratio = 1
        notes = collections.defaultdict(list)
        for instrument in pm.instruments:
            if (not instrument.is_drum and instrument.program < 97):  # filter out drum instruments
                # Sort the notes by start time inside the instrument
                sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
                prev_start = sorted_notes[0].start * bpm_ratio
                prev_pitch = 0

                for note in sorted_notes:
                    if (note.pitch < 120):
                        start = note.start
                        end = note.end
                        notes['note'].append(note.pitch)
                        notes['start'].append(start * bpm_ratio)
                        notes['step'].append(
                            start - prev_start)  # step = midi event timing, delta time relative to previous event
                        notes['duration'].append((end - start) * bpm_ratio)
                        notes['end'].append(start * bpm_ratio + (end - start) * bpm_ratio)
                        notes['program'].append(instrument.program)
                        notes['local_diff'].append(note.pitch - prev_pitch)
                        notes['is_piano'].append(int(instrument.program >= 1 and instrument.program <= 8))
                        notes['is_organ'].append(int(instrument.program >= 17 and instrument.program <= 24))
                        notes['is_guitar'].append(int(instrument.program >= 25 and instrument.program <= 32))
                        notes['is_bass'].append(int(instrument.program >= 33 and instrument.program <= 40))
                        notes['is_strings'].append(int(instrument.program >= 41 and instrument.program <= 48))
                        notes['is_ensemble'].append(int(instrument.program >= 49 and instrument.program <= 56))
                        notes['is_brass'].append(int(instrument.program >= 57 and instrument.program <= 64))
                        notes['is_reed'].append(int(instrument.program >= 65 and instrument.program <= 72))
                        notes['is_pipe'].append(int(instrument.program >= 73 and instrument.program <= 80))
                        notes['is_synth_lead'].append(int(instrument.program >= 81 and instrument.program <= 88))
                        notes['is_synth_pad'].append(int(instrument.program >= 89 and instrument.program <= 96))
                        prev_start = start
                        prev_pitch = note.pitch

        # sort all notes before resulting DF
        result = pd.DataFrame({name: np.array(value) for name, value in notes.items()})
        result = result.sort_values(by=['start'], ignore_index=True).reset_index(drop=True)

        # normalize scale - transpose to each scale A
        if (normalize_scale):
            score = m21.converter.parse(midi_file)
            key = score.analyze('key')
            result["note"] = result["note"] + self.transpose_table[str(key.tonic)]

        # calculate diff
        diff = []
        for i in range(len(result["note"]) - 1, 0, -1):
            diff.insert(0, result["note"][i] - result["note"][i - 1])
        diff.insert(0, result["note"][0])  # first note diff is it's distance from 0.
        result["diff"] = diff

        # count previous observations for each note
        observed = np.zeros(128)
        obs_column = []
        for i in range(len(result["note"])):
            cur_note = result["note"][i]
            obs_column.append(observed[cur_note])
            observed[cur_note] += 1
        result["observed"] = obs_column

        # assign tokens
        token_column = list(range(len(result["note"])))
        result["token"] = token_column

        # show most common note (mode)
        number_list = list(observed)

        return result.drop(features_to_drop, axis=1)

    def train(self, midi_dir, batch_size=16, lr_discriminator=0.0001,
              lr_generator=0.0001, num_epochs=30):

        comp_length = 192

        if (not midi_dir[-2:-1] == "\\"):
            midi_dir = midi_dir + "\\"

        midi_files = os.listdir(midi_dir)

        columns = ["note", "start", "end", "step", "duration", "program", "diff", "observed", "token", 'local_diff',
                   'is_piano', 'is_organ', 'is_guitar', 'is_bass', 'is_strings', 'is_ensemble', 'is_brass', 'is_reed',
                   'is_pipe', 'is_synth_lead', 'is_synth_pad']  # do not touch this line
        features_to_drop = ["end", "step", "program", "token", "observed"]
        self.columns = columns = [item for item in columns if item not in features_to_drop]
        self.num_features = num_features = len(columns)
        seed_size = 64 * num_features
        songs_processed = 0

        print("Reading from disk...")
        x = np.array([])
        y = np.array([])
        i = 0
        for song in midi_files:
            print(song)
            raw_data = self.extractMessages(midi_dir + song, features_to_drop=features_to_drop, base_bpm=210,
                                            normalize_time=True, normalize_scale=True)
            if (len(raw_data["note"]) < comp_length):
                print("Error: not enough notes in this file. File will be skipped.")
            note_slice = raw_data[0:comp_length]
            note_slice.reset_index(drop=True, inplace=True)
            x = np.append(x, note_slice.to_numpy().reshape(1, comp_length, num_features))
            y = np.append(y, [1])

            i += 1

        print("Loading done. Beginning train procedure...")

        x_real = x.reshape(len(midi_files), -1)
        y_real = y.reshape(len(midi_files), -1)
        self.scaler = MinMaxScaler()
        x_real = self.scaler.fit_transform(x_real)

        num_songs = x_real.shape[0]
        train_percentage = 0.9
        x_image = x_real[:int(train_percentage * num_songs)]
        y_image = y_real[:int(train_percentage * num_songs)]

        x_domain = x_real[int(train_percentage * num_songs):]
        y_domain = y_real[int(train_percentage * num_songs):]

        x_image = torch.from_numpy(x_image)
        x_domain = torch.from_numpy(x_domain)
        y_image = torch.from_numpy(y_image)
        y_domain = torch.from_numpy(y_domain)

        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_size = comp_length * num_features
                self.layer_multiplier = 128

                self.bn1 = nn.BatchNorm1d(num_features)

                # convolution 1
                self.conv1_kernel_size = 32
                self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_features * 2,
                                       kernel_size=self.conv1_kernel_size, padding="same", padding_mode="zeros")
                self.mp1 = nn.MaxPool1d(num_features)

                # fc1
                self.fc1_in_size = comp_length * num_features * 2
                self.fc1 = nn.Sequential(
                    nn.Linear(self.fc1_in_size, self.fc1_in_size),
                    nn.LeakyReLU(0.2),
                )

                # convolution 2
                self.conv2_kernel_size = 32
                self.conv2 = nn.Conv1d(in_channels=num_features * 2, out_channels=num_features * 4,
                                       kernel_size=self.conv2_kernel_size, padding="same", padding_mode="zeros")
                self.mp2 = nn.MaxPool1d(num_features)

                # fc2
                self.fc2_in_size = (comp_length) * num_features * 4
                self.fc2 = nn.Sequential(
                    nn.Linear(self.fc2_in_size, self.fc2_in_size),
                    nn.LeakyReLU(0.2)
                )

                # convolution 3
                self.conv3_kernel_size = 8
                self.conv3 = nn.Conv1d(in_channels=num_features * 4, out_channels=num_features * num_features,
                                       kernel_size=self.conv3_kernel_size, padding="same", padding_mode="zeros")
                self.mp3 = nn.MaxPool1d(num_features)

                # fc3
                self.fc3_in_size = comp_length * num_features
                self.fc3 = nn.Sequential(
                    nn.Linear(self.fc3_in_size, self.fc3_in_size),
                    nn.Dropout(0.2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.fc3_in_size, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = x.reshape(-1, num_features, comp_length)
                x = self.conv1(x)
                x = x.reshape(-1, self.fc1_in_size)
                x = self.fc1(x)
                x = x.reshape(-1, num_features * 2, comp_length)
                x = self.conv2(x)
                x = x.reshape(-1, self.fc2_in_size)
                x = self.fc2(x)
                x = x.reshape(-1, num_features * 4, comp_length)
                x = self.mp3(self.conv3(x))
                x = x.reshape(-1, self.fc3_in_size)
                x = self.fc3(x)
                return x

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_size = 128

                # fc1 - transform the seed into an initial song shape
                self.fc1 = nn.Sequential(
                    nn.Linear(seed_size, 2 * seed_size, bias=False),
                    nn.LeakyReLU(0.2),
                    nn.Linear(2 * seed_size, 3 * seed_size, bias=False)
                )

                self.bn1 = nn.BatchNorm1d(num_features)

                # convolution 1
                self.conv1_kernel_size = 32
                self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=num_features * 2,
                                       kernel_size=self.conv1_kernel_size, padding="same", padding_mode="zeros")

                # fc 2
                self.fc2_in_size = comp_length * num_features
                self.fc2 = nn.Sequential(
                    nn.Linear(self.fc2_in_size, self.fc2_in_size, bias=False),
                    nn.LeakyReLU(0.2)
                )

                # convolution 2
                self.conv2_comp_length = comp_length
                self.conv2_kernel_size = 24
                self.conv2 = nn.Conv1d(in_channels=num_features * 2, out_channels=num_features * 3,
                                       kernel_size=self.conv2_kernel_size, padding="same", padding_mode="zeros")

                # fc 3
                self.fc3_in_size = (
                                       self.conv2_comp_length) * num_features * 3  # each convolution loses (kernel_size) features (because of the layer edges)
                self.fc3 = nn.Sequential(
                    nn.LeakyReLU(0.2)
                )

                # convolution 3
                self.conv3_comp_length = self.conv2_comp_length
                self.conv3_kernel_size = 20
                self.conv3 = nn.Conv1d(in_channels=num_features * 3, out_channels=num_features * 4,
                                       kernel_size=self.conv3_kernel_size, padding="same", padding_mode="zeros")

                # fc 4
                self.fc4_in_size = self.conv3_comp_length * num_features * 4
                self.fc4 = nn.Sequential(
                    nn.LeakyReLU(0.2)
                )

                # convolution 4
                self.conv4_comp_length = self.conv3_comp_length
                self.conv4_kernel_size = 8
                self.conv4 = nn.Conv1d(in_channels=num_features * 4, out_channels=num_features * num_features,
                                       kernel_size=self.conv4_kernel_size, padding="same", padding_mode="zeros")
                self.mp4 = nn.MaxPool1d(num_features)

                # fc 5
                self.fc5_in_size = comp_length * num_features
                self.fc5 = nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.fc5_in_size, comp_length * num_features, bias=False),
                )

            def forward(self, x):
                x = self.fc1(x)
                x = x.reshape(-1, num_features, comp_length)
                x = self.bn1(x)
                x = self.conv1(x)
                x = x.reshape(-1, self.fc2_in_size)
                x = self.fc2(x)
                x = x.reshape(-1, num_features * 2, self.conv2_comp_length)
                x = self.conv2(x)
                x = x.reshape(-1, self.fc3_in_size)
                x = self.fc3(x)
                x = x.reshape(-1, num_features * 3, self.conv3_comp_length)
                x = self.conv3(x)
                x = x.reshape(-1, self.fc4_in_size)
                x = self.fc4(x)
                x = x.reshape(-1, num_features * 4, self.conv4_comp_length)
                x = self.mp4(self.conv4(x))
                x = x.reshape(-1, self.fc5_in_size)
                x = self.fc5(x)
                return x

        discriminator = Discriminator()
        generator = Generator()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: %s" % device)
        discriminator = discriminator.to(device)
        generator = generator.to(device)
        x_domain, x_image, y_domain, y_image = x_domain.to(device), x_image.to(device), y_domain.to(device), y_image.to(
            device)

        class MyDataset(data.Dataset):
            def __init__(self, X, Y):
                self.X = X
                self.Y = Y

            def __len__(self):
                return len(self.Y)

            def __getitem__(self, index):
                X = self.X[index].float().reshape(-1)
                Y = self.Y[index].float();
                return X, Y

            def getRandomX(self):
                index = random.randint(0, len(self.X) - 1)
                return self.X[index][0:seed_size].float().reshape(-1)

        train_dataset = MyDataset(x_image, y_image)
        train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.domain_dataset = domain_dataset = MyDataset(x_domain, y_domain)

        # train parameters:
        loss_function = nn.BCELoss()
        loss_randomness = nn.MSELoss()

        # noise parameters:
        mu, sigma = 0, 0.0005  # mean and standard deviation for the noise
        uniform_range =  0.01
        self.uniform_range = 10 * uniform_range

        # optimizers
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_discriminator)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr_generator)

        def runsTest(l_in, l_median):

            l = l_in[::num_features]  # extracts only notes

            runs, n1, n2 = 0, 0, 0

            # Checking for start of new run
            for i in range(len(l)):
                # no. of runs (transformations from PtoN or NtoP)
                if (l[i] >= l_median and l[i - 1] < l_median) or \
                        (l[i] < l_median and l[i - 1] >= l_median):
                    runs += 1

                    # no. of positive values
                if (l[i]) >= l_median:
                    n1 += 1

                    # no. of negative values
                else:
                    n2 += 1

            runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
            stan_dev = math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                                 (((n1 + n2) ** 2) * (n1 + n2 - 1)))

            if (stan_dev == 0):
                stan_dev = 0.000001
            z = (runs - runs_exp) / stan_dev

            result = z

            return result

        def bartels_test(r_in):
            r = r_in[::num_features]  # extract notes
            sum1 = 0
            for i in range(len(r) - 1):
                sum1 += (r[i] - r[i + 1]) ** 2
            r_mean = np.array(r).mean()
            sum2 = 0
            for i in range(len(r)):
                sum2 += (r[i] - r_mean) ** 2
            result = sum1 / sum2

            r = r_in[1::num_features]  # extract start times
            r.sort()  # sort by start time to reflect differences of each two consecutive events
            sum1 = 0
            for i in range(len(r) - 1):
                sum1 += (r[i] - r[i + 1]) ** 2
            r_mean = np.array(r).mean()
            sum2 = 0
            for i in range(len(r)):
                sum2 += (r[i] - r_mean) ** 2
            result = result + sum1 / sum2

            return result / 2

        print_every = 1
        d_loss, g_loss, r_loss, z_ratio, z_real, z_generated, b_ratio, b_real, b_generated = (
            [], [], [], [], [], [], [], [], [])
        GAN_at_epoch = {}
        GAN_at_minZ = (None, None, sys.float_info.max, 0)  # (generator, discriminator, abs(1-z_ratio), epoch)
        GAN_at_minBart = (None, None, sys.float_info.max, 0)  # (generator, discriminator, abs(1-z_ratio), epoch)
        GAN_at_lastConvergence = (
            None, None, sys.float_info.max, 0)  # (generator, discriminator, abs(1-z_ratio), epoch)
        for epoch in range(num_epochs):
            loss_sum_d, loss_sum_g, loss_sum_r, num_batches = 0, 0, 0, 0
            z_batch_real, z_batch_generated = 0, 0
            b_batch_real, b_batch_generated = 0, 0
            for real_samples, real_labels in train_loader:
                num_batches += 1

                # seed generation
                # latent_space_samples = torch.randn((len(real_samples), seed_size)).to(device=device)
                latent_space_samples = torch.cat(tuple([domain_dataset.getRandomX() for _ in range(len(real_samples))]))
                latent_space_samples = latent_space_samples.reshape(len(real_samples), -1)
                noise = np.random.uniform(low=-1 * uniform_range, high=uniform_range, size=latent_space_samples.shape)
                noise_tensor = torch.from_numpy(noise).float().to(device)
                latent_space_samples = latent_space_samples + noise_tensor
                # generator: convert seed to "fake" data
                generated_samples = generator(latent_space_samples)
                # generated midi files are "fake" - all should be labled 0
                generated_samples_labels = torch.zeros((len(real_samples), 1)).to(device=device)

                # add noise to the real data to prevent weight memory on the discriminator
                # normal distribution helps maintaining the overall structure of the data
                noise = np.random.normal(mu, sigma, size=real_samples.shape)
                noise_tensor = torch.from_numpy(noise).float().to(device)
                distorted_real_samples = real_samples + noise_tensor

                # to train the discriminator we add the real samples to the generated ones
                all_samples = torch.cat((distorted_real_samples, generated_samples))
                all_samples_labels = torch.cat((real_labels, generated_samples_labels))

                # Training the discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)  # discriminator forward
                loss_discriminator = loss_function(output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.cat(tuple([domain_dataset.getRandomX() for _ in range(len(real_samples))]))
                latent_space_samples = latent_space_samples.reshape(len(real_samples), -1)
                noise = np.random.uniform(low=-1 * uniform_range, high=uniform_range, size=latent_space_samples.shape)
                noise_tensor = torch.from_numpy(noise).float().to(device)
                latent_space_samples = latent_space_samples + noise_tensor

                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples)  # generator forward
                output_discriminator_generated = discriminator(generated_samples)  # discriminator gives feedback
                loss_generator = loss_function(output_discriminator_generated, real_labels)
                loss_generator.backward()
                optimizer_generator.step()

                # Calculate Z-statistic and Bartel's test for both real data and generated data
                z_real_mean_current, z_generated_mean_current = 0, 0
                b_real_mean_current, b_generated_mean_current = 0, 0
                real_samples_cpu = real_samples.detach().cpu()
                for i in range(len(real_samples)):
                    z_real_mean_current += runsTest(real_samples_cpu[i].numpy(), real_samples_cpu[i].median().numpy())
                    b_real_mean_current += bartels_test(real_samples_cpu[i].numpy())

                generated_samples_cpu = generated_samples.detach().cpu()
                for i in range(len(generated_samples)):
                    z_generated_mean_current += runsTest(generated_samples_cpu[i].numpy(),
                                                         generated_samples_cpu[i].median().numpy())
                    b_generated_mean_current += bartels_test(generated_samples_cpu[i].numpy())

                # Get avarages for Z-statistic and Bartel's test
                z_real_mean_current, z_generated_mean_current = z_real_mean_current / len(
                    real_samples), z_generated_mean_current / len(output_discriminator_generated)
                z_batch_real += z_real_mean_current
                z_batch_generated += z_generated_mean_current

                b_real_mean_current, b_generated_mean_current = b_real_mean_current / len(
                    real_samples), b_generated_mean_current / len(output_discriminator_generated)
                b_batch_real += b_real_mean_current
                b_batch_generated += b_generated_mean_current

                # Sum discriminator/generator/randomness loss
                loss_sum_d += loss_discriminator.detach().cpu().numpy()
                loss_sum_g += loss_generator.detach().cpu().numpy()

            loss_sum_d /= num_batches
            loss_sum_g /= num_batches
            d_loss.append(loss_sum_d)
            g_loss.append(loss_sum_g)

            z_batch_real /= num_batches
            z_batch_generated /= num_batches
            z_real.append(z_batch_real)
            z_generated.append(z_batch_generated)
            z_ratio.append(z_batch_generated / z_batch_real)

            b_batch_real /= num_batches
            b_batch_generated /= num_batches
            b_real.append(b_batch_real)
            b_generated.append(b_batch_generated)
            b_ratio.append(b_batch_generated / b_batch_real)

            # save minimum z_statistic ratio
            if (abs(1 - z_batch_generated / z_batch_real) - epoch * 0.0001 < GAN_at_minZ[
                2]):  # prefer later epochs than small differences
                GAN_at_minZ = (copy.deepcopy(generator).to("cpu"), copy.deepcopy(discriminator).to("cpu"),
                               abs(1 - z_batch_generated / z_batch_real), epoch)
            # save minimum barlet test ratio
            if (abs(1 - b_batch_generated / b_batch_real) - epoch * 0.0001 < GAN_at_minBart[2]):
                GAN_at_minBart = (copy.deepcopy(generator).to("cpu"), copy.deepcopy(discriminator).to("cpu"),
                                  abs(1 - b_batch_generated / b_batch_real), epoch)
            if (loss_sum_d < 0.9 and loss_sum_g < 0.9 and abs(1 - z_batch_generated / z_batch_real) < 0.2):
                GAN_at_lastConvergence = (
                    copy.deepcopy(generator).to("cpu"), copy.deepcopy(discriminator).to("cpu"), loss_sum_g, epoch)
            # Show loss
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch: {epoch} Loss Discriminator: {loss_sum_d}")
                print(f"Epoch: {epoch} Loss Generator: {loss_sum_g}")
                print(f"Epoch: {epoch} Z: %f B: %f" % (
                    abs(1 - z_batch_generated / z_batch_real), abs(1 - b_batch_generated / b_batch_real)))
                print("-----")

        selected_method = GAN_at_minZ
        selected_epoch_state = selected_method[3]
        self.generator, self.discriminator = (selected_method[0], selected_method[1])

        print("Train done. Please use the Produce method to generate new files.")

    def produce(self, midi_dir, num_files = 1):
        print("Generating %d files..." %num_files)
        if (not midi_dir[-2:-1] == "\\"):
            midi_dir = midi_dir + "\\"

        for i in range(0, num_files):
            seed = self.domain_dataset.getRandomX()
            noise = np.random.uniform(low=-1 * self.uniform_range, high=self.uniform_range, size=seed.shape)
            noise_tensor = torch.from_numpy(noise).float()
            seed = seed + noise_tensor
            generated = self.generator(seed).detach().numpy()
            generated = self.scaler.inverse_transform(generated)
            gan_result = pd.DataFrame(generated.reshape(-1, self.num_features), columns=self.columns)
            start_time_filtered_df = gan_result[gan_result["start"] >= 0]
            duration_filtered_df = start_time_filtered_df[start_time_filtered_df["duration"] > 0]
            filename = "JetBit_generated_" + str(i) + ".mid"
            self.pandasToMIDI(duration_filtered_df, round_digits=3, quantize=True, soft_fix_notes=True,
                              transpose_by=randrange(0, 11), prevent_overlap=True,
                              filename=midi_dir + filename, instrument=1)
            print("File generated: %s" % filename)

        return filename

    def pandasToMIDI(self, df_input, instrument=0, filename="/content/drive/MyDrive/GAN out/tmp file garbage.mid",
                     round_digits=5, quantize=False, soft_fix_notes=False, transpose_by=0, prevent_overlap=False):
        scaleA_notes = [0, 2, 4, 5, 7, 9, 11]
        quantize_measure = 0.150  # 1/16 for 120bpm
        midi_file = pretty_midi.PrettyMIDI()
        finalized_notes = []  # returns output notes
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
                div = int(start / quantize_measure)
                mod = start / quantize_measure - int(start / quantize_measure)
                start = start - mod * quantize_measure

            # soft-move notes to scale
            if (round(row[1]["note"]) < 1 or round(row[1]["note"]) > 127):
                print("Warning: Integrity issue: %d" % round(row[1]["note"]))
            cur_note = round(row[1]["note"])
            remainder = row[1]["note"] - cur_note
            if (soft_fix_notes):
                # if the current note is not in-scale, but rounded up/down is - convert it. Otherwise - leave note as is.
                if (remainder >= 0.5 and not cur_note % 12 in scaleA_notes and (
                        cur_note + 1) % 12 in scaleA_notes):
                    cur_note = cur_note + 1
                if (remainder < 0.5 and not cur_note % 12 in scaleA_notes and (
                        cur_note - 1) % 12 in scaleA_notes):
                    cur_note = cur_note - 1

            cur_note = cur_note + transpose_by

            if (row[1]["is_bass"] > 0.09):
                cur_note -= 12  # bass is one octave lower

            if (row[1]["is_pipe"] > 0.09):
                cur_note += 12  # bass is one octave lower

            if (cur_note >= 0 and cur_note <= 127):
                finalized_notes.append(cur_note)
                write_note = True
                if cur_note in last_endtime.keys():
                    if last_endtime[cur_note] > start:  # we allow overlap only if the note is about to end
                        write_note = False
                last_endtime[cur_note] = start + duration

                if write_note or not prevent_overlap:
                    note = pretty_midi.Note(velocity=randrange(65, 75), pitch=cur_note, start=start,
                                            end=start + duration)
                    midi_channel.notes.append(note)
                    written += 1

        midi_file.instruments.append(midi_channel)
        midi_file.write(filename)
        return finalized_notes


if __name__ == '__main__':
    jb = JetBit()
    jb.train(midi_dir="C:\\Users\\ofek\\Desktop\\mid", num_epochs=10, batch_size=16)
    jb.produce(midi_dir="C:\\Users\\ofek\\Desktop", num_files=5)
    print("done")

