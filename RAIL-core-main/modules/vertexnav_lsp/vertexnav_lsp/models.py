import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.logging import tensorboard_plot_decorator
from vertexnav.models import EncoderNBlocks, DecoderNBlocks


def masked_reduce_mean(data, mask):
    mask = (mask > 0.5).float()
    mask_sum = torch.sum(mask)
    data_sum = torch.sum(data * mask)
    return data_sum / (mask_sum + 1)


class VertexLSPOmni(nn.Module):
    name = "VertexLSPOmni"
    ROLL_VARIABLES = [
        'image', 'is_vertex', 'is_right_gap',
        'is_corner', 'is_left_gap',
        'is_point_vertex',
        'is_frontier',
        'is_feasible',
        'delta_success_cost',
        'exploration_cost',
        'positive_weighting',
        'negative_weighting',
        'goal_loc_x',
        'goal_loc_y',
    ]

    def __init__(self, args=None, do_triple_input=True):
        super(VertexLSPOmni, self).__init__()
        self._args = args

        # Initialize the blocks
        self.enc_1 = EncoderNBlocks(3, 64, num_layers=2)
        self.enc_2 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_3 = EncoderNBlocks(64 + 2, 128, num_layers=2)
        self.enc_4 = EncoderNBlocks(128, 128, num_layers=2)
        self.enc_5 = EncoderNBlocks(128, 256, num_layers=2)

        self.dec_1 = DecoderNBlocks(256, 128, num_layers=2)
        self.dec_2 = DecoderNBlocks(128, 64, num_layers=2)
        self.dec_3 = DecoderNBlocks(64, 64, num_layers=2)
        self.conv_out = nn.Conv2d(64, 3 + 5, kernel_size=1)
        self.goal_bn = nn.BatchNorm2d(2)
        self.do_triple_input = do_triple_input

    def forward(self, data, device):
        image = data['image'].to(device)
        g = self.goal_bn(
            torch.stack((data['goal_loc_x'], data['goal_loc_y']),
                        1).expand([-1, -1, 32, -1]).float().to(device))

        if self.do_triple_input:
            image = torch.tile(image, (1, 1, 1, 3))
            g = torch.tile(g, (1, 1, 1, 3))

        # Encoding layers
        x = image
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = torch.cat((x, g), 1)  # Add the goal info tensor
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)

        # Decoding layers
        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_3(x)

        # Output layer
        x = self.conv_out(x)
        if self.do_triple_input:
            x = x[:, :, :, 128:256]

        return x

    def _loss_lsp(self, nn_out, data, device, writer, index):
        # Separate outputs.
        is_feasible_logits = nn_out[:, 0, :, :]
        delta_cost_pred = nn_out[:, 1, :, :]
        exploration_cost_pred = nn_out[:, 2, :, :]

        is_feasible_label = data['is_feasible'].to(device)
        is_frontier = data['is_frontier'].to(device)
        delta_cost_label = data['delta_success_cost'].to(device)
        exploration_cost_label = data['exploration_cost'].to(device)
        pweight = data['positive_weighting'].to(device)
        nweight = data['negative_weighting'].to(device)
        rpw = self._args.relative_positive_weight_lsp

        # Compute the contribution from the is_feasible_label
        is_feasible_xentropy = rpw * is_feasible_label * -F.logsigmoid(is_feasible_logits) * pweight / 10 + \
            (1 - is_feasible_label) * -F.logsigmoid(-is_feasible_logits) * nweight / 10
        is_feasible_xentropy = masked_reduce_mean(is_feasible_xentropy, is_frontier)

        # Delta Success Cost
        delta_cost_pred_error = masked_reduce_mean(
            torch.square(delta_cost_pred - delta_cost_label) / (100 ** 2),
            is_frontier * is_feasible_label)

        # Exploration Cost
        exploration_cost_pred_error = masked_reduce_mean(
            torch.square(exploration_cost_pred - exploration_cost_label) / (200 ** 2),
            is_frontier * (1 - is_feasible_label))

        # Sum the contributions
        loss = is_feasible_xentropy + delta_cost_pred_error + exploration_cost_pred_error

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/LSP/is_feasible_xentropy",
                              is_feasible_xentropy.item(),
                              index)
            writer.add_scalar("Loss/LSP/delta_success_cost_loss",
                              delta_cost_pred_error.item(),
                              index)
            writer.add_scalar("Loss/LSP/exploration_cost_loss",
                              exploration_cost_pred_error.item(),
                              index)
            writer.add_scalar("Loss/LSP/lsp_loss",
                              loss.item(),
                              index)

        return loss

    def _loss_vertnav(self, nn_out, data, device, writer, index):

        rpw = self._args.relative_positive_weight_vert

        # Compute the contribution from is_vertex
        is_vertex_logits = nn_out[:, 0, :, :]
        is_vertex_label = data['is_vertex'].to(device)
        is_vertex_xentropy = rpw * is_vertex_label * -F.logsigmoid(is_vertex_logits) + \
            (1 - is_vertex_label) * -F.logsigmoid(-is_vertex_logits)
        is_vertex_xentropy = torch.mean(is_vertex_xentropy)

        # Separate outputs.
        is_label_logits = nn_out[:, 1:, :, :]
        is_right_gap = data['is_right_gap'].to(device)
        is_corner = data['is_corner'].to(device)
        is_left_gap = data['is_left_gap'].to(device)
        is_point_vertex = data['is_point_vertex'].to(device)
        is_label_label = torch.stack(
            [is_right_gap, is_corner, is_left_gap, is_point_vertex],
            axis=1)
        is_label_logsoftmax = torch.nn.LogSoftmax(dim=1)(is_label_logits)
        is_label_xentropy = -torch.sum(
            is_label_logsoftmax * is_label_label,
            dim=1)
        is_label_xentropy = masked_reduce_mean(
            is_label_xentropy, is_vertex_label)

        loss = self._args.vertex_pred_weight * is_vertex_xentropy + is_label_xentropy

        # Logging
        if writer is not None:
            writer.add_scalar("Loss/VertNav/is_vertex_xentropy",
                              is_vertex_xentropy.item(),
                              index)
            writer.add_scalar("Loss/VertNav/vertnav_loss",
                              loss.item(),
                              index)

        return loss

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):

        loss = (self._loss_lsp(nn_out[:, :3], data, device, writer, index) +
                self._loss_vertnav(nn_out[:, 3:], data, device, writer, index))

        if writer is not None:
            writer.add_scalar("Loss/total_loss",
                              loss.item(),
                              index)

        return loss

    @classmethod
    def get_net_eval_fn(_, network_file, device, do_return_model=False):
        model = VertexLSPOmni()
        model.load_state_dict(torch.load(network_file,
                                         map_location=torch.device('cpu')),
                              strict=False)
        model.eval()
        model.to(device)

        # empty_goal = np.zeros([args.num_range, args.num_bearing])

        def vertex_lsp_net(image, goal_loc_x, goal_loc_y):
            with torch.no_grad():
                image = np.transpose(image, (2, 0, 1))
                out = model({
                    'image': torch.tensor(np.expand_dims(image, axis=0)).float(),
                    'goal_loc_x': torch.tensor(np.expand_dims(goal_loc_x, axis=0)).float(),
                    'goal_loc_y': torch.tensor(np.expand_dims(goal_loc_y, axis=0)).float()
                }, device=device).detach().cpu()

                return {
                    'subgoal_prob_feasible': torch.sigmoid(out[0, 0]).numpy(),
                    'subgoal_delta_success_cost': out[0, 1].numpy(),
                    'subgoal_exploration_cost': out[0, 2].numpy(),
                    'is_vertex': torch.sigmoid(out[0, 3]).numpy(),
                    'vertex_label': np.transpose(torch.softmax(out[0, 4:8], dim=0).numpy(),
                                                 axes=(1, 2, 0)),
                }

        if do_return_model:
            return vertex_lsp_net, model
        else:
            return vertex_lsp_net

    @tensorboard_plot_decorator
    def plot_images(self, fig, image, out, data):
        image = np.transpose(image, (1, 2, 0))
        is_frontier_pred = torch.sigmoid(out[0]).cpu().numpy()
        is_frontier = data['is_frontier'][0]
        is_feasible = data['is_feasible'][0]
        is_feasible[is_frontier < 0.5] = float('NaN')
        is_frontier_pred_masked = is_frontier_pred.copy()
        is_frontier_pred_masked[is_frontier < 0.5] = float('NaN')

        is_vertex_pred = torch.sigmoid(out[3]).cpu().numpy()
        is_vertex = data['is_vertex'][0]

        axs = fig.subplots(4, 2)
        axs[0, 0].imshow(image, interpolation='none')
        axs[1, 0].imshow(is_feasible, interpolation='none', vmin=0.0, vmax=1.0)
        axs[2, 0].imshow(is_frontier_pred, interpolation='none', vmin=0.0, vmax=1.0)
        axs[3, 0].imshow(is_frontier_pred_masked, interpolation='none', vmin=0.0, vmax=1.0)

        axs[0, 1].imshow(image, interpolation='none')
        axs[1, 1].imshow(is_vertex, interpolation='none', vmin=0.0, vmax=1.0)
        axs[2, 1].imshow(is_vertex_pred, interpolation='none', vmin=0.0, vmax=1.0)
        axs[3, 1].imshow(0 * is_vertex_pred, interpolation='none', vmin=0.0, vmax=1.0)
