aliases = {
    'model1': '/data3/sreevatsa/models/db_resnet50.pt',
    'model2': '/home2/sreevatsa/models/final/Class_balanced finetune for all_layers_epoch11.pt',
    'model3': '/home2/sreevatsa/models/final/Random_sampling-finetune for all layers (Backbone unfreezed)_epoch22.pt',
    'model4': '/home2/sreevatsa/models/final/Random_sampling-finetune for last layers (Backbone freezed)_v2_epoch9.pt'
}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--ImageFile", type=str, default=None, help="Input file to get text detections")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=1024, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=0, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, choices=aliases.keys(), metavar='choice',help="Path to your checkpoint")
    # parser.add_argument("--resume", type=str, default=None, choices=aliases.keys(), metavar='choice',help="Path to your checkpoint")
    # parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--para_only", type=bool, default=False, help="Visualize only paragraphs")
    parser.add_argument("--col_only", type=bool, default=False, help="Visualize only columns")
    parser.add_argument("--metric", type=str, default='euclidean', help="Euclidean/Cheybshev distance")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")

    
    args = parser.parse_args()

    return args

args = parse_args()