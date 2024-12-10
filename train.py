import torch
import logging
from utils.params import set_params
from utils.util import set_seed
from datetime import datetime
import time
from input_data.load_data import load
from models.trainer import ME2BertTrainer
from pathlib import Path
import os

def get_device_info():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current Device: {current_device}")
        print(f"Device Name: {device_name}")
    else:
        print("CUDA is not available. Using CPU.")


if __name__ == '__main__':
    clean_data = True    
    args = set_params()
    args.weighted_loss = True 
    args.checkpoint_dir = os.path.join(args.checkpoint_dir)
    args.output_dir = os.path.join(args.output_dir)
    args.temperature = .0
    print(f'Setting up output and checkpoint dir at {args.output_dir} and {args.checkpoint_dir}')
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >=0 else "cpu"
    
    
    seed = args.seed
    
    set_seed(seed)
    
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logfilename = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d%H%M%S"))
    logging.basicConfig(filename=logfilename + '.log',
                format="%(message)s",
                level=logging.DEBUG)
    
    if not args.contrastive:
        logging.info('Contrastive learning set to False!')
        print('Contrastive learning set to false')
    if not args.transformation:
        logging.info('No transformation!')
        print('No transformation!')

    start_time = time.time()
    logging.info('Start processing data...')
    print('Start processing data...')
    if clean_data:
        logging.info('Cleanining text')
    fold = load(args, clean_data=clean_data)
    for arg in sorted(vars(args)):
        logging.info(f'{arg}: {getattr(args, arg)}')

    logging.info(f'Finished processing data. Time: {time.time()-start_time}')
    print(f'Finished processing data. Time: {time.time()-start_time}')
    
    start_time = time.time()
    logging.info('Start training...')
    print('Start training...')
    get_device_info()
    trainer = ME2BertTrainer(fold, args)
    
    print(args)
    trainer.train()
    
    logging.info(
        f"Finished training data. Time: {time.time()-start_time}") 
    