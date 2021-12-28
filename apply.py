def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    return tensor.argmax(dim=-1)


def test(model, test_loader, device):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        pred = model(data)
        pred = get_likely_index(pred)

        correct += number_of_correct(pred, target)

    print(f"Accuracy: {correct / len(test_loader.dataset)}")
