# makemore

This showcases multiple language models built in Python and PyTorch.

This repo reimplimets Andrej Karpathys first few nn's, then diverges with rnn and transformers archs.

Makemore is intended to take in a textfile as input, used in training, then generates more of that thing.

## Files

- `names.txt` - download link in ipynb files, contains 32k names used as the dataset
- `bigram.ipynb` - the simplest model with an inuitive approach - counting occurences of characters after another, then sampling from that distribution until a termining character is met
- `mlp.ipynb` - neural network approach based on [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- `wavenet.ipynb` - wavenet approach [A¨aron van den Oor et al. 2016](https://arxiv.org/pdf/1609.03499)
- `backprop.ipynb` - backprop from scratch.
- `rnn.ipynb` - rnn with lstm approach.
- `transform.ipynb` - transformer arch.

## Sample Output

After training, the model can generate new names like: 
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
