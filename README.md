# makemore

This is a simple character-level language model built in Python and PyTorch, heavily inspired by [Andrej Karpathy's makemore](https://github.com/karpathy/makemore). It's a reimplementation meant for learning purposes — the goal is to better understand how neural language models work under the hood, from bigrams to multi-layer MLPs.

This repo follows Karpathy’s project flow closely but is built and documented independently as part of a personal learning journey.

Makemore is intended to take in a textfile as input, used in training, then generates more of that thing.

## Files

- `names.txt` - download link in ipynb files, contains 32k names used as the dataset
- `makemore_bigram.ipynb` - the simplest model with an inuitive approach - counting occurences of characters after another, then sampling from that distribution until a termining character is met
- `makemore_mlp.ipynb` - neural network approach based on [this](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

## Sample Output

After training, the model can generate new names like: (So Far)
jaydella.
laya.
sleenoz.
aitya.
ailenys.
mailyn.
tey.
karrah.
samayana.
devanna.
atjoslit.
conton.
khammariangelda.
unacearia.
yillille.
zen.
popper.
oui.
dmir.
dedero.
