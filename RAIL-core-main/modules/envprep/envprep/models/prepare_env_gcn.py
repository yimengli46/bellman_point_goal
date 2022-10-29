import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool
import learning


class PrepareEnvGCN(nn.Module):
    name = "PrepareEnvGCN"

    def __init__(self, args=None):
        super(PrepareEnvGCN, self).__init__()
        self.args = args
        self.conv1 = TransformerConv(16, 16, edge_dim=1)
        self.conv2 = TransformerConv(16, 8, edge_dim=1)
        self.conv3 = TransformerConv(8, 8, edge_dim=1)
        self.conv4 = TransformerConv(8, 4, edge_dim=1)
        self.classifier = nn.Linear(4 * 2, 1)

    def forward(self, data, device):
        data = data.to(device)
        x, edge_index, eddge_attr, batch_index = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = F.leaky_relu(self.conv1(x, edge_index, eddge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, eddge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, eddge_attr))
        x = self.conv4(x, edge_index, eddge_attr)
        x = torch.cat(
            [global_mean_pool(x, batch_index), global_add_pool(x, batch_index)], dim=1
        )
        x = self.classifier(x)
        return x

    def loss(self, nn_out, data, device="cpu", writer=None, index=None):
        y = data.exc.to(device)
        op = nn_out[:, 0]
        loss = nn.MSELoss()
        loss_tot = loss(op, y)
        # Logging
        if writer is not None:
            writer.add_scalar("Loss/total_loss", loss_tot.item(), index)

        return loss_tot

    @learning.logging.tensorboard_plot_decorator
    def plot_images(self, fig, image, out, data):
        pred_cost = (out[0]).cpu().numpy()
        true_cost = data.exc[0]
        axs = fig.subplots(1, 1)
        axs.imshow(image)
        axs.set_title(f"true cost: {true_cost} | predicted cost: {pred_cost}")

    @classmethod
    def get_net_eval_fn(_, network_file, device):
        model = PrepareEnvGCN()
        model.load_state_dict(torch.load(network_file))
        model.eval()
        model.to(device)

        def prepare_net(graph):
            batch_idx = torch.tensor(
                [0 for i in range(graph["num_nodes"])], dtype=torch.int64
            )
            gcn_data = Data(
                x=graph["graph_nodes"],
                edge_index=graph["graph_edge_index"],
                edge_attr=graph["graph_edge_feats"],
                batch=batch_idx,
            )
            with torch.no_grad():
                out = model(gcn_data, device)
                out = out[:, 0].detach().cpu().numpy()
                return out[0]

        return prepare_net
