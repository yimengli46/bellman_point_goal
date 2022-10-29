import os
import torch
import torch.nn as nn
import lsp_cond
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from learning.data import CSVPickleDataset
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class LSPConditionalGNN(nn.Module):
    name = 'LSPConditionalGNN'

    def __init__(self, args=None):
        super(LSPConditionalGNN, self).__init__()
        torch.manual_seed(8616)
        self._args = args
        self.conv1 = GCNConv(258, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, data, device):
        edge_index = data['edge_data'].to(device)
        x = data['latent_features']
        history = data['history'].view(-1, 1).to(device)
        is_subgoal = data['is_subgoal'].view(-1, 1).to(device)
        h = torch.cat((x, history), 1)
        h = torch.cat((h, is_subgoal), 1)
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h)
        h = self.classifier(h)
        return h

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0]
        delta_cost_pred = nn_out[:, 1]
        exploration_cost_pred = nn_out[:, 2]

        # Convert the data
        is_feasible_label = data.y.to(device)
        delta_cost_label = data.dsc.to(device)
        exploration_cost_label = data.ec.to(device)
        pweight = data.pweight.to(device)
        nweight = data.nweight.to(device)
        history = data.history.to(device)
        rpw = self._args.relative_positive_weight

        # Compute the contribution from the is_feasible_label
        is_feasible_xentropy = rpw * is_feasible_label * -F.logsigmoid(is_feasible_logits) * \
            pweight / 10 + (1 - is_feasible_label) * -F.logsigmoid(-is_feasible_logits) * nweight / 10
        is_feasible_xentropy = torch.mean(history * is_feasible_xentropy)

        # Delta Success Cost
        delta_cost_pred_error = torch.square(
            delta_cost_pred - delta_cost_label) / (100 ** 2) * is_feasible_label
        delta_cost_pred_error = torch.mean(history * delta_cost_pred_error)

        # Exploration Cost
        exploration_cost_pred_error = torch.square(
            exploration_cost_pred - exploration_cost_label) / (200 ** 2) * (1 - is_feasible_label)
        exploration_cost_pred_error = torch.mean(history * exploration_cost_pred_error)

        # Sum the contributions
        loss = is_feasible_xentropy + delta_cost_pred_error + exploration_cost_pred_error

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/is_feasible_xentropy",
                              is_feasible_xentropy.item(),
                              index)
            writer.add_scalar("Loss/delta_success_cost_loss",
                              delta_cost_pred_error.item(),
                              index)
            writer.add_scalar("Loss/exploration_cost_loss",
                              exploration_cost_pred_error.item(),
                              index)
            writer.add_scalar("Loss/total_loss",
                              loss.item(),
                              index)

        return loss

    @classmethod
    def get_net_eval_fn(_, network_file, device):
        model = LSPConditionalGNN()
        model.load_state_dict(torch.load(network_file, 
                                         map_location=device)) 
        model.eval()
        model.to(device)

        def frontier_net(datum, vertex_points, subgoals):
            graph = lsp_cond.utils.preprocess_gcn_data(datum)
            prob_feasible_dict = {}
            dsc_dict = {}
            ec_dict = {}
            vp = vertex_points.tolist()
            with torch.no_grad():
                out = model.forward(graph, device)
                out = out[:, :3]
                out[:, 0] = torch.sigmoid(out[:, 0])
                out = out.detach().cpu().numpy()
                for subgoal in subgoals:
                    possible_node = lsp_cond.utils. \
                        get_subgoal_node(vertex_points, subgoal).tolist()
                    # Extract subgoal properties for a subgoal
                    subgoal_props = out[vp.index(possible_node)]
                    prob_feasible_dict[subgoal] = subgoal_props[0]
                    dsc_dict[subgoal] = subgoal_props[1]
                    ec_dict[subgoal] = subgoal_props[2]
                return prob_feasible_dict, dsc_dict, ec_dict, out[:, 0]
             
        return frontier_net


def train(args, flag, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    prep_fn = lsp_cond.utils.preprocess_gcn_training_data(flag)

    # Create the datasets and loaders
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    print("Number of training graphs:", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_iter = iter(train_loader)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "train"))

    test_dataset = CSVPickleDataset(test_path, prep_fn)
    print("Number of testing graphs:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "test"))

    # Initialize the network and the optimizer
    model = LSPConditionalGNN(args)
    model.to(device)
    latent_features_net = lsp_cond.learning.models.auto_encoder.AutoEncoder. \
        get_net_eval_fn(args.autoencoder_network_file, 
                        device=device, do_preprocess_cnn_datum=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    index = 0
    while index < args.num_steps:
        # Get the batches
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
        train_batch = train_batch[0]
        # Get latent features by running the encoder of AutoEncoder
        train_latent_features = latent_features_net(
            lsp_cond.utils.preprocess_encoder_batch(train_batch)
        )
        out = model.forward({
            'edge_data': train_batch.edge_index,
            'history': train_batch.history,
            'is_subgoal': train_batch.is_subgoal,
            'latent_features': train_latent_features
        }, device)

        train_loss = model.loss(out,
                                data=train_batch,
                                device=device,
                                writer=train_writer,
                                index=index)

        if index % args.test_log_frequency == 0:
            print(train_loss)
            print(f"[{index}/{args.num_steps}] "
                  f"Train Loss: {train_loss}")
            # print(f"Prediction", torch.sigmoid(out).T)

        # Train the system
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if index % args.test_log_frequency == 0:
            try:
                test_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_batch = next(test_iter)
            test_batch = test_batch[0]
            test_latent_features = latent_features_net(
                lsp_cond.utils.preprocess_encoder_batch(test_batch)
            )
            with torch.no_grad():
                out = model.forward({
                    'edge_data': test_batch.edge_index,
                    'history': test_batch.history,
                    'is_subgoal': test_batch.is_subgoal,
                    'latent_features': test_latent_features
                }, device)
                test_loss = model.loss(out,
                                       data=test_batch,
                                       device=device,
                                       writer=test_writer,
                                       index=index)
                print(f"[{index}/{args.num_steps}] "
                      f"Test Loss: {test_loss.cpu().numpy()}")
        index += 1

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, "model.pt"))
