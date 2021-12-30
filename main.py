import torch
import torch.utils.data
import baseline_model
import bc_resnet_model
import get_data
import train
import apply


def main(model, train_loader, test_loader, optimizer, scheduler, device, n_epoch=10):
    for epoch in range(n_epoch):
        print(f"--- start epoch {epoch} ---")
        train.train_epoch(model, optimizer, train_loader, device, epoch, log_interval=10)
        scheduler.step()
        apply.test(model, test_loader, device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print(f"Device: {device}")

    # model_name = "baseline"
    model_name = "nope"
    if model_name == "baseline":
        print("Model: baseline")
        model_module = baseline_model
        model = baseline_model.M5().to(device)
    else:
        print("Model: bc-resnet")
        model_module = bc_resnet_model
        model = bc_resnet_model.BcResNetModel().to(device)

    train_set = get_data.SubsetSC(subset="training")
    test_set = get_data.SubsetSC(subset="testing")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=model_module.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=model_module.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    main(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        device
    )
