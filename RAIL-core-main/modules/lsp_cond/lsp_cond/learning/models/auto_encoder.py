import torch
import os
import lsp_cond
import numpy as np
import torch.nn as nn
import learning
from learning.data import CSVPickleDataset
from torch.utils.data import DataLoader
from lsp.learning.models.shared import EncoderNBlocks
from vertexnav.models import DecoderNBlocks
from torch.utils.tensorboard import SummaryWriter


BATCH_SCALE = 1000


class AutoEncoder(nn.Module):
    name = "AutoEncoder"

    def __init__(self, args=None):
        super(AutoEncoder, self).__init__()
        torch.manual_seed(8616)

        self._args = args
        self.enc_1 = EncoderNBlocks(3, 32, num_layers=2)
        self.enc_2 = EncoderNBlocks(32, 32, num_layers=2)
        self.enc_3 = EncoderNBlocks(32 + 4, 64, num_layers=2)
        self.enc_4 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_5 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_6 = EncoderNBlocks(64, 64, num_layers=2)
        self.enc_1_conv_1x1 = nn.Conv2d(64, 16, kernel_size=1)

        self.dec_1_conv_1x1 = nn.Conv2d(16, 64, kernel_size=1)
        self.dec_1 = DecoderNBlocks(64, 64, num_layers=2)
        self.dec_2 = DecoderNBlocks(64, 64, num_layers=2)
        self.dec_3 = DecoderNBlocks(64, 64, num_layers=2)
        self.dec_4 = DecoderNBlocks(64, 32 + 4, num_layers=2)
        self.dec_5 = DecoderNBlocks(32, 32, num_layers=2)
        self.dec_6 = DecoderNBlocks(32, 3, num_layers=2)
        self.loc_conv_1x1 = nn.Conv2d(4, 4, kernel_size=1)

    # The following method is used only during evaluating
    def encoder(self, data, device):
        x = data['image'].to(device)
        x = self.enc_1(x)
        x = self.enc_2(x)
        # Compute goal info tensor
        if 'goal_loc_x' in data.keys():
            g = torch.stack((data['goal_loc_x'], data['goal_loc_y']),
                            1).expand([-1, -1, 32, -1]).float().to(device) / BATCH_SCALE
        else:
            raise ValueError("Missing goal location data.")
        if 'subgoal_loc_x' in data.keys():
            s = torch.stack((data['subgoal_loc_x'], data['subgoal_loc_y']),
                            1).expand([-1, -1, 32, -1]).float().to(device) / BATCH_SCALE
        else:
            raise ValueError("Missing subgoal location data.")
        
        x = torch.cat((x, s, g), 1)  # Add the goal info tensor
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        latent_features = self.enc_1_conv_1x1(x)
        return latent_features

    def decoder(self, x, device):
        x = self.dec_1_conv_1x1(x)
        x = self.dec_1(x)
        x = self.dec_2(x)
        x = self.dec_3(x)
        x = self.dec_4(x)
        # Apply non-ReLU layer & Compress expanded dimension
        loc_vec = torch.mean(self.loc_conv_1x1(x[:, 32:]), 2)
        glx = loc_vec[:, 0]
        gly = loc_vec[:, 1]
        slx = loc_vec[:, 2]
        sly = loc_vec[:, 3]
        # Continue decoding for the image
        x = x[:, :32]
        x = self.dec_5(x)
        x = self.dec_6(x)
        output = {
            'image': x,
            'goal_loc_x': glx,
            'goal_loc_y': gly,
            'subgoal_loc_x': slx,
            'subgoal_loc_y': sly,
        }
        return output

    # The following method is used only during training
    def forward(self, data, device):
        x = self.encoder(data, device)
        return self.decoder(x, device)

    def loss(self, nn_out, data, device='cpu', writer=None, index=None):
        mse_loss = nn.MSELoss()
        pred_img = nn_out['image']
        trgt_img = data['image'].to(device)
        loss_img = mse_loss(pred_img, trgt_img)

        pred_glx = torch.reshape(nn_out['goal_loc_x'], (-1, 1, 128))
        trgt_glx = data['goal_loc_x'].to(device) / BATCH_SCALE  # Diving to scale
        loss_glx = mse_loss(pred_glx, trgt_glx)

        pred_gly = torch.reshape(nn_out['goal_loc_y'], (-1, 1, 128))
        trgt_gly = data['goal_loc_y'].to(device) / BATCH_SCALE
        loss_gly = mse_loss(pred_gly, trgt_gly)

        pred_slx = torch.reshape(nn_out['subgoal_loc_x'], (-1, 1, 128))
        trgt_slx = data['subgoal_loc_x'].to(device) / BATCH_SCALE
        loss_slx = mse_loss(pred_slx, trgt_slx)

        pred_sly = torch.reshape(nn_out['subgoal_loc_y'], (-1, 1, 128))
        trgt_sly = data['subgoal_loc_y'].to(device) / BATCH_SCALE
        loss_sly = mse_loss(pred_sly, trgt_sly)
        total_loss = (loss_img + loss_glx + loss_gly + loss_slx + loss_sly) / 5

        # Logging
        if writer is not None:
            writer.add_scalar("Loss_AE/image_loss",
                              loss_img.item(),
                              index)
            writer.add_scalar("Loss_AE/goal_loc_x_loss",
                              loss_glx.item(),
                              index)
            writer.add_scalar("Loss_AE/goal_loc_y_loss",
                              loss_gly.item(),
                              index)
            writer.add_scalar("Loss_AE/subgoal_loc_x_loss",
                              loss_slx.item(),
                              index)
            writer.add_scalar("Loss_AE/subgoal_loc_y_loss",
                              loss_sly.item(),
                              index)
            writer.add_scalar("Loss_AE/total_loss",
                              total_loss.item(),
                              index)
        return total_loss
        
    @classmethod
    def get_net_eval_fn(_, network_file, device, do_preprocess_cnn_datum=True):
        model = AutoEncoder()
        model.load_state_dict(torch.load(network_file, 
                                         map_location=device)) 
        model.eval()
        model.to(device)
        
        def latent_features_net(datum):    
            with torch.no_grad():
                if do_preprocess_cnn_datum:
                    data = lsp_cond.utils.preprocess_cnn_data(datum)
                else:
                    data = datum
                latent_features = model.encoder(data, device)
            return latent_features.flatten(start_dim=1)
                     
        return latent_features_net

    @learning.logging.tensorboard_plot_decorator
    def plot_images(self, fig, image, out, data):
        pred_img = np.transpose(out['image'][0].detach().cpu().numpy(), (1, 2, 0))
        trgt_img = np.transpose(data['image'][0].cpu().numpy(), (1, 2, 0))

        pred_glx = torch.reshape(out['goal_loc_x'][0], (128,)).detach().cpu().numpy()
        trgt_glx = torch.reshape(data['goal_loc_x'][0], (128,)).cpu().numpy() / BATCH_SCALE

        pred_gly = torch.reshape(out['goal_loc_y'][0], (128,)).detach().cpu().numpy()
        trgt_gly = torch.reshape(data['goal_loc_y'][0], (128,)).cpu().numpy() / BATCH_SCALE

        pred_slx = torch.reshape(out['subgoal_loc_x'][0], (128,)).detach().cpu().numpy()
        trgt_slx = torch.reshape(data['subgoal_loc_x'][0], (128,)).cpu().numpy() / BATCH_SCALE

        pred_sly = torch.reshape(out['subgoal_loc_y'][0], (128,)).detach().cpu().numpy()
        trgt_sly = torch.reshape(data['subgoal_loc_y'][0], (128,)).cpu().numpy() / BATCH_SCALE

        axs = fig.subplots(2, 2)
        axs[0][0].imshow(trgt_img, interpolation='none')
        axs[1][0].plot(trgt_glx)
        axs[1][0].plot(trgt_gly)
        axs[1][0].plot(trgt_slx)
        axs[1][0].plot(trgt_sly)

        axs[0][1].imshow(pred_img, interpolation='none')
        axs[1][1].plot(pred_glx)
        axs[1][1].plot(pred_gly)
        axs[1][1].plot(pred_slx)
        axs[1][1].plot(pred_sly)
        axs[0][0].set_title("Input image")
        axs[0][1].set_title("Recreated image")
        axs[1][0].set_title("Input location vectors")
        axs[1][1].set_title("Recreated location vectors")


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    prep_fn = lsp_cond.utils.preprocess_autoencoder_data

    # Create the datasets and loaders
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    print("Number of training images:", len(train_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0)

    train_iter = iter(train_loader)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "train_autoencoder"))

    test_dataset = CSVPickleDataset(test_path, prep_fn)
    print("Number of testing images:", len(test_dataset))
    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=0)
    
    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "test_autoencoder"))

    # Initialize the network and the optimizer
    model = AutoEncoder(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    index = 0
    while index < args.num_steps:
        # Get the batches
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
        out = model.forward(train_batch, device)  # out is a dictionary
        train_loss = model.loss(out,
                                data=train_batch,
                                device=device,
                                writer=train_writer,
                                index=index)
        if index % args.test_log_frequency == 0:
            print(train_loss)
            print(f"[{index}/{args.num_steps}] "
                  f"Train Loss: {train_loss}")

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
            with torch.no_grad():
                out = model.forward(test_batch, device)
                test_loss = model.loss(out,
                                       data=test_batch,
                                       device=device,
                                       writer=test_writer,
                                       index=index)
                # Plotting
                model.plot_images(test_writer,
                                  'image',
                                  index,
                                  image=None,
                                  out=out,
                                  data=test_batch)
                print(f"[{index}/{args.num_steps}] "
                      f"Test Loss: {test_loss.cpu().numpy()}")
        index += 1

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, "AutoEncoder.pt"))
