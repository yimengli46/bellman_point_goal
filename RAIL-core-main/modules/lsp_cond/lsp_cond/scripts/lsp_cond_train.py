import torch
import lsp_cond


if __name__ == "__main__":
    args = lsp_cond.utils.parse_args()
    # Always freeze your random seeds
    torch.manual_seed(8616)
    train_path, test_path = lsp_cond.utils.get_data_path_names(args)
    # Train the neural network
    if args.autoencoder_network_file:
        print("Training GCN ... ...")       
        # flag == ['marginal' or 'random']
        lsp_cond.learning.models.gcn.train(
            args=args, flag='random', 
            train_path=train_path, test_path=test_path)
    else:
        print("Training AutoEncoder ... ...")
        lsp_cond.learning.models.auto_encoder.train(
            args=args, train_path=train_path, test_path=test_path)
