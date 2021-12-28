import os
from torchaudio.datasets import SPEECHCOMMANDS


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, path="./", subset: str = None):
        super().__init__(path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fh:
                return [
                    os.path.join(self._path, line.strip()) for line in fh
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        return waveform, sample_rate, label

    
    
LABELS = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',
    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',
    'up',
    'visual',
    'wow',
    'yes',
    'zero'
]

_label_to_idx = {label: i for i, label in enumerate(LABELS)}
_idx_to_label = {i: label for label, i in _label_to_idx.items()}
    
def label_to_idx(label):
    return _label_to_idx[label]

def idx_to_label(idx):
    return _idx_to_label[idx]
    
    
if __name__ == "__main__":
    train_set = SubsetSC(subset="training")
    test_set = SubsetSC(subset="testing")

    waveform, sample_rate, label = train_set[15000]
    print(waveform.shape, sample_rate, label)
