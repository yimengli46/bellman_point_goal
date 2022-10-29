import argparse
import os
import torch
from torch import autograd
import learning
from torch.utils.tensorboard import SummaryWriter
from envprep.models.prepare_env_gcn import PrepareEnvGCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _get_preprocess_env_data_fn(args):
    def preprocess_env_data(datum):
        data = datum.copy()
        data["expected_cost"] = torch.tensor(data["expected_cost"], dtype=torch.float)
        data_gcn = Data(
            x=data["graph_nodes"],
            edge_index=data["graph_edge_index"],
            edge_attr=data["graph_edge_feats"],
            exc=data["expected_cost"],
            image=data["graph_image"],
        )

        return data_gcn

    return preprocess_env_data


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = PrepareEnvGCN(args).to(device)
    print(f"Training Device: {device}")

    # Create the datasets and loaders
    preprocess_function = _get_preprocess_env_data_fn(args)
    train_dataset = learning.data.CSVPickleDataset(
        args.training_data_file, preprocess_function
    )
    # print(list(train_dataset))
    test_dataset = learning.data.CSVPickleDataset(
        args.test_data_file, preprocess_function
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    test_loader_iter = iter(test_loader)

    # Set up logging
    train_writer = SummaryWriter(log_dir=os.path.join(args.logdir, "train"))
    test_writer = SummaryWriter(log_dir=os.path.join(args.logdir, "test"))

    # Define the optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tot_index = 0
    autograd.set_detect_anomaly(True)
    for epoch in range(args.num_epochs):
        for index, batch in enumerate(train_loader):
            out = model.forward(batch, device)
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

                    tim = tbatch.image
                    tout = model.forward(batch, device)
                    tloss = model.loss(
                        tout, tbatch, device=device, writer=test_writer, index=tot_index
                    )
                    next(iter(test_loader))

                    print(f"Test Loss({epoch}.{index}, {tot_index}): {tloss.item()}")
                    model.plot_images(
                        test_writer,
                        "image",
                        tot_index,
                        image=tim[0],
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
        "--num_epochs", default=10, help="Number of epochs to run training", type=int
    )
    parser.add_argument(
        "--learning_rate", default=0.1, help="Initial learning rate", type=float
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
