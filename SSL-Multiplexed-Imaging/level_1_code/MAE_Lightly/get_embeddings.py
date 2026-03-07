from pathlib import Path
import numpy as np

from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

import torch
import torchvision
from lightly.data import LightlyDataset

import chunked_h5_dataset
from main import MAE

import argparse



def get_embeddings(args, weightp):
    
    mean_std, args.n_channels = chunked_h5_dataset.get_mean_std()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.Normalize(mean=mean_std['mean'],
                                         std=mean_std['std']
                                         ),
        ])
    
    # create a lightly dataset for training with augmentations
    base = chunked_h5_dataset.h5_chunk_wrapper(Path(args.data_path))
    dataset = LightlyDataset.from_torch_dataset(base, transform=test_transform)
    print('Loaded dataset with length:', dataset.__len__())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE.load_from_checkpoint(weightp)
    model = model.to(device)
    model.eval()
        
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 3:
                views, fnames, _ = batch
            else:
                views, fnames = batch
                
            img = views[0] if isinstance(views, (list, tuple)) else views
            img = img.to(device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb.cpu())
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    
    np.save(args.save_dir / f'embeddings_{weightp.stem}.npy', embeddings)
    np.save(args.save_dir / f'names_{weightp.stem}.npy', filenames)
    
    print('ALL DONE!')

 
def main(args):
    ckpt_path = Path(args.checkpoint_path)

    # exact checkpoints you want
    wanted_names = [
        "epochepoch=0099-v1.ckpt",
        "epochepoch=0199-v1.ckpt",
        "epochepoch=0299-v1.ckpt",
        "epochepoch=0399.ckpt",
        "epochepoch=0499.ckpt",
    ]

    if ckpt_path.is_file():
        weight_list = [ckpt_path]
    else:
        weight_list = [ckpt_path / name for name in wanted_names]

        # check missing files
        missing = [p for p in weight_list if not p.exists()]
        if missing:
            print("These checkpoints were not found:")
            for p in missing:
                print(" ", p)
            raise FileNotFoundError("Some requested checkpoints are missing.")

    print("Selected checkpoints:")
    for p in weight_list:
        print(" ", p.name)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for weightp in weight_list:
        get_embeddings(args, weightp)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for your script')
    
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers for data loading (default: 7)')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=6e-2, help='Learning rate (default: 6e-2)')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Momentum (default: 0.9)')
    
    parser.add_argument('--input_size', type=int, default=224, help='Input size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    
    parser.add_argument('--dataset', type=str, default='', help='Name of the dataset')
    parser.add_argument('--data_path', type=str, default="")
    
    parser.add_argument('--checkpoint_path', type=str, default="", help='Path to the checkpoint')
    parser.add_argument('--save_dir', type=Path, default="embeddings_mae", help='Path to the checkpoint')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    main(args)
