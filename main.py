from flask import Flask, render_template, request
from torch._C._te import Cond

import Consts
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import random

app = Flask(__name__)
model = GAN("model/GAN_generator_55 (4).dat", "model/GAN_discriminator_5.dat", "model/GAN_scaler_5.dat",
            "model/x_real.pickle", "model/y_real.pickle")


@app.route('/midiPlayer')
def midiPlayer():
    filename = model.generate()
    msg = random.choice(Consts.messages)
    return render_template('midiPlayer.html', midi_file=filename, message=msg)

@app.route('/')
def about():
    return render_template('about.html')

app.run(host='localhost', port=5000, debug=True)
