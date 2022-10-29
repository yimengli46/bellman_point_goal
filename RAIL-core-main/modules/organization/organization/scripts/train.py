import argparse
import os
import numpy as np
import torch
import learning
from torch.utils.tensorboard import SummaryWriter
from organization.models.organizationCNN import OrganizationNet


def _get_preprocess_env_data_fn(args):
    def preprocess_env_data(datum):
        datum["image"] = (
            np.transpose(datum["image"], (2, 0, 1)).astype(np.float32) / 255
        )

        return datum

    return preprocess_env_data


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = OrganizationNet(args).to(device)
    print(f"Training Device: {device}")

    # Create the datasets and loaders
    preprocess_function = _get_preprocess_env_data_fn(args)
    train_dataset = learning.data.CSVPickleDataset(
        args.training_data_file, preprocess_function
    )
    test_dataset = learning.data.CSVPickleDataset(
        args.test_data_file, preprocess_function
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader_iter = iter(test_loader)

    # Set up logging
    train_writer = SummaryWriter(log_dir=os.path.join(args.logdir, "train"))
    test_writer = SummaryWriter(log_dir=os.path.join(args.logdir, "test"))

    # Define the optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tot_index = 0
    for epoch in range(args.num_epochs):
        for index, batch in enumerate(train_loader):
            out = model(batch, device)
            loss = model.loss(
                out, batch, device=device, writer=train_writer, index=tot_index
            )

            if index % 10 == 0:
                with torch.no_grad():
                    try:
                        tbatch = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_loader)
                        tbatch = next(test_loader_iter)

                    tim = tbatch["image"]
                    tout = model(tbatch, device)
                    tloss = model.loss(
                        tout, tbatch, device=device, writer=test_writer, index=tot_index
                    )
                    next(iter(test_loader))

                    print(f"Test Loss({epoch}.{index}, {tot_index}): {tloss.item()}")
                    model.plot_images(
                        test_writer,
                        "image",
                        tot_index,
                        image=tim[0].detach(),
                        out=tout[0].detach().cpu(),
                        data=tbatch,
                    )

            # Perform update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_index += 1

    # Now save the trained model to file
    torch.save(model.state_dict(), os.path.join(args.logdir, f"{model.name}.pt"))


def get_parser():
    # Add new arguments
    parser = argparse.ArgumentParser(
        description="Train Blockworld Env net with PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training_data_file",
        default=["tmp.tfrecords"],
        nargs="+",
        help="TFRecord containing the training data",
        type=str,
    )
    parser.add_argument(
        "--test_data_file",
        default=[],
        nargs="+",
        help="TFRecord containing the test data",
        type=str,
    )

    # Logging
    parser.add_argument(
        "--logdir",
        help="Directory in which to store log files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--mini_summary_frequency",
        default=100,
        help="Frequency (in steps) mini summary printed to the terminal",
        type=int,
    )
    parser.add_argument(
        "--summary_frequency",
        default=1000,
        help="Frequency (in steps) summary is logged to file",
        type=int,
    )

    # Training
    parser.add_argument(
        "--num_epochs", default=20, help="Number of epochs to run training", type=int
    )
    parser.add_argument(
        "--learning_rate", default=0.005, help="Initial learning rate", type=float
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        help="Number of data per training iteration batch",
        type=int,
    )

    return parser


if __name__ == "__main__":
    # Parse the command line args and set device
    args = get_parser().parse_args()
    main(args)
