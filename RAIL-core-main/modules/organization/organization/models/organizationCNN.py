import numpy as np
import torch
import torch.nn as nn
import learning
from lsp.learning.models.shared import EncoderNBlocks


class OrganizationNet(nn.Module):
    name = "OrganizationNet"

    def __init__(self, args=None, num_outputs=1):
        super(OrganizationNet, self).__init__()
        self._args = args

        # Initialize the blocks
        self.enc_1 = EncoderNBlocks(3, 64, num_layers=2)
        self.enc_2 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_3 = EncoderNBlocks(64, 128, num_layers=2)
        self.enc_4 = EncoderNBlocks(128, 128, num_layers=2)
        self.enc_5 = EncoderNBlocks(128, 256, num_layers=2)
        self.enc_6 = EncoderNBlocks(256, 128, num_layers=2)
        self.conv_1x1 = nn.Conv2d(128, 16, kernel_size=1)

        self.fc_outs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, num_outputs),
        )

    # Defining the forward pass
    def forward(self, data, device):
        image = data["image"].to(device)
        x = image
        # Encoding layers
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        x = self.conv_1x1(x)
        x = self.fc_outs(x)

        return x

    def loss(self, nn_out, data, device="cpu", writer=None, index=None):
        y = data["expected_cost"].to(device)
        loss = nn.MSELoss()
        ip = nn_out.double()[:, 0]
        loss_tot = loss(ip, y)

        # Logging
        if writer is not None:
            writer.add_scalar("Loss", loss_tot.item(), index)

        return loss_tot

    @learning.logging.tensorboard_plot_decorator
    def plot_images(self, fig, image, out, data):
        image = np.transpose(image, (1, 2, 0))
        pred_cost = (out[0]).cpu().numpy()
        true_cost = data["expected_cost"][0]

        axs = fig.subplots(1, 1)
        axs.imshow(image, interpolation="none")
        axs.set_title(f"true cost: {true_cost} | predicted cost: {pred_cost}")

    @classmethod
    def get_net_eval_fn(_, network_file, device, do_return_model=False):
        model = OrganizationNet()
        model.load_state_dict(torch.load(network_file))
        model.eval()
        model.to(device)

        def blockworld_net(image):
            with torch.no_grad():
                image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255
                out = model(
                    {
                        "image": torch.tensor(np.expand_dims(image, axis=0)).float(),
                    },
                    device=device,
                )
            out = out[:, 0].detach().cpu().numpy()
            return out[0]

        if do_return_model:
            return blockworld_net, model
        else:
            return blockworld_net
