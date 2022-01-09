# BC-ResNet for Keyword Spotting
Unofficial implementation of [Broadcasted Residual Learning for Efficient Keyword Spotting](https://arxiv.org/abs/2106.04140)

# TODO:
- add specaug to train

### Usage
Train
```
; train scaled 2 times model for 50 epochs and save best checkpoint to model-sc-2.pt
python main.py train --scale 2 --epoch 50 --checkpoint-file model-sc-2.pt

; Device: cuda
; Use subspectral norm: True
; --- start epoch 0 ---
; Train Epoch: 0  Loss: 3.6272
; Train Epoch: 0  Loss: 1.6613
; ...
; Train Epoch: 49 Loss: 0.3026
; Validation accuracy: 0.9626289950906722
; Top validation accuracy: 0.9628293758140467
; Test accuracy: 0.9604725124943208
```

Test
```
; test saved model on test dataset
python main.py test --scale 2 --model-file model-sc-2.pt

; Test accuracy: 0.9604725124943208
```

Apply
```
; apply saved model to wav file
python main.py apply --scale 2 --model-file model-sc-2.pt --wav-file SpeechCommands/speech_commands_v0.02/seven/5744b6a7_nohash_0.wav

seven   0.99977
six     0.00011
stop    0.00008
happy   0.00002
up      0.00000
```

You can found pretrained `model-sc-2.pt` model in `example_model` folder.

### Options and help
Use
```
python main.py --help
python main.py train --help
python main.py test --help
python main.py apply --help
```

This implementation use all 35 labels from Google Speech Commands Dataset. Original paper use 10 commands and additional re-balanced "Unknown word" and "Silence" labels (section 4.1 in paper).
