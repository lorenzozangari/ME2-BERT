import argparse

def set_params():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for model training")

    parser.add_argument('--data_path', type=str, default="./data/e2mocase_full.csv", help='Path to the input data file.')
    parser.add_argument('--device', type=int, default=7, help='GPU device index to use (-1 for CPU).')
    parser.add_argument('--seed', type=int, default=72, help='Random seed for reproducibility.')
    
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased', help='Pre-trained language model to use for fine-tuning.')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length for tokenization.')
    parser.add_argument('--no_gate', action='store_true', help='Flag to disable the gate component.')
        
    parser.add_argument('--padding', type=str, default="max_length", help='Padding strategy (e.g., "max_length").')
    parser.add_argument('--no_truncation', action='store_true', help='Flag to disable truncation (False if truncation is required).')
    parser.add_argument('--mf_classes', type=int, default=5, help='Number of moral foundation classes (-1 for automatic detection).')
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation.")
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of training epochs.')

    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for regularization.')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for the optimizer.')
    parser.add_argument('--lambda_con', type=float, default=1, help='Weighting factor for the contrastive loss term.')
    parser.add_argument('--alpha', type=float, default=10, help='Decay rate parameter for the learning rate schedule.')
    parser.add_argument('--beta', type=float, default=0.25, help='Sharpness parameter for the learning rate decay.')
    parser.add_argument('--gamma', type=float, default=10, help='Scaling factor for domain adaptation and/or contrastive loss.')
        
    parser.add_argument('--lambda_trans', type=float, default=1.0, help='Scaling factor for the autoencoder loss.')
    parser.add_argument('--num_no_adv', type=int, default=5, help='Number of initial epochs without adversarial learning.')        
    parser.add_argument('--num_epoch_save', type=int, default=5, help='Frequency (in epochs) for saving the best model.')    
    parser.add_argument('--save_data', action='store_true', help='Save the input data if not already saved, or load it if it exists.')
    parser.add_argument('--output_dir', type=str, default='./artifacts', help='Directory where performance scores will be saved.')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint", help='Directory for storing checkpoint models.')
    parser.add_argument('--contrastive', action='store_true', help='Enable contrastive learning.')
    parser.add_argument('--transformation', action='store_true', help='Enable the use of an autoencoder for transformation.')    

    args = parser.parse_args()
    
    return args
