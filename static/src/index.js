import { FancyMidiPlayer } from "static/midi";

const createMusicalPiece = (id, name, path) => ({ id, name, path });

const pieces = [
  createMusicalPiece(
    0,
    "Chopin - Etude Revolutionary",
    "/static/tmp/0529202221341132630077604992247.mid"
  )
];

const instrumentUrl =
  "https://raw.githubusercontent.com/gleitz/midi-js-soundfonts/gh-pages/FatBoy/acoustic_grand_piano-mp3.js";

const setAppBusy = (isBusy) => {
  const playButton = document.querySelector("#play-piece");
  const stopButton = document.querySelector("#stop-piece");
  const musicalPiecesSelect = document.querySelector("#musical-pieces");

  if (isBusy) {
    playButton.setAttribute("disabled", true);
    stopButton.setAttribute("disabled", true);
    musicalPiecesSelect.setAttribute("disabled", true);
  } else {
    playButton.removeAttribute("disabled");
    stopButton.removeAttribute("disabled");
    musicalPiecesSelect.removeAttribute("disabled");
  }
};

const cp = new FancyMidiPlayer(document);
setAppBusy(true);
cp.setInstrument(instrumentUrl).then(() => {
  const playButton = document.querySelector("#play-piece");
  const stopButton = document.querySelector("#stop-piece");
  playButton.onclick = cp.playMidi.bind(cp);
  stopButton.onclick = cp.stopMidi.bind(cp);
  changePiece(0);
});

const changePiece = (pieceId) => {
  setAppBusy(true);
  cp.stopMidi();
  cp.setMidi(pieces[pieceId].path).then(() => setAppBusy(false));
};

const musicalPiecesSelect = document.querySelector("#musical-pieces");
musicalPiecesSelect.onchange = (evt) => changePiece(evt.target.value);

pieces
  .map((piece) => {
    const option = document.createElement("option");
    option.id = piece.id;
    option.value = piece.id;
    option.innerHTML = piece.name;
    option.selected = piece.id === 0;
    return option;
  })
  .forEach((pieceOption) => musicalPiecesSelect.append(pieceOption));
