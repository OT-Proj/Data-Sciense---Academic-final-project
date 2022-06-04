columns = ['note','start','duration','diff','local_diff','is_piano','is_organ','is_guitar','is_bass','is_strings','is_ensemble','is_brass','is_reed','is_pipe','is_synth_lead','is_synth_pad']
comp_length = 192
num_features = len(columns)
seed_size = 64 * num_features
quantize_measure = 0.150  # 1/16 for 120bpm
scaleA_notes = [0, 2, 4, 5, 7, 9, 11]

messages = [
    "Here is a randomly generated MIDI file, just for you!",
    "Another grammy award winning MIDI file from JetBit!",
    "Where words fail, music speaks.",
    "A song is like a face. Both can be generated with GAN!",
    "Some people have lives; some people have music. Some people are programmers.",
    "A painter paints pictures on canvas. But musicians paint their pictures on silence."
]