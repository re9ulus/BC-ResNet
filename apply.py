import torch
import get_data
import numpy as np
import torchaudio


def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    return tensor.argmax(dim=-1)


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        pred = model(data)
        pred = get_likely_index(pred)

        correct += number_of_correct(pred, target)

    score = correct / len(data_loader.dataset)
    return score


def apply_to_wav(model, waveform: torch.Tensor, sample_rate: float, device: str):
    model.eval()
    mel_spec = get_data.prepare_wav(waveform, sample_rate)
    mel_spec = torch.unsqueeze(mel_spec, dim=0).to(device)
    res = model(mel_spec)

    probs = torch.nn.Softmax(dim=-1)(res).cpu().detach().numpy()
    predictions = []
    for idx in np.argsort(-probs):
        label = get_data.idx_to_label(idx)
        predictions.append((label, probs[idx]))
    return predictions


def apply_to_file(model, wav_file: str, device: str):
    waveform, sample_rate = torchaudio.load(wav_file)
    return apply_to_wav(model, waveform, sample_rate, device)
