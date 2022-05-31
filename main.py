from flask import Flask, render_template, request
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
model = GAN("model/GAN_generator_55 (4).dat", "model/GAN_discriminator_5.dat", "model/GAN_scaler_5.dat",
            "model/x_real.pickle", "model/y_real.pickle")


@app.route('/')
def midiPlayer():
    filename = model.generate()
    return render_template('midiPlayer.html', midi_file=filename)

app.run(host='localhost', port=5000, debug=True)
