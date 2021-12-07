import torch
import torch.nn.functional as F
import os
from omegaconf import OmegaConf
import sys

from dataloader.stir_dataset import STIRDataset
from dataloader.data_loaders import get_dataloader

from model import composition_models
from utils.simple_tokenizer import SimpleTokenizer
from utils.util import set_seed, mkdir, load_config_file, write_json
from utils.logger import setup_logger

from torch.optim import Adam, AdamW # both are same but AdamW has a default weight decay
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
from tqdm import tqdm

DATA_CONFIG_PATH = 'configs/dataset_config.yaml'
TRAINER_CONFIG_PATH = 'configs/train_config.yaml'

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tirg')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    parser.add_argument("--do_eval", action='store_true', help="Whether to run inference.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=False, help="Path of model checkpoint")
    parser.add_argument("--gallery_size", default=5000, type=int, required=False, help="Size of image gallery to be searched. should be <= 5000")

    args = parser.parse_args()
    return args

def evaluate(config, eval_dataset, model):
    '''
    Evaluates the model
    '''
    # taking gallery size samples randomly from eval dataset
    indices = torch.randperm(len(eval_dataset))[:config.gallery_size]
    eval_data_subset = torch.utils.data.Subset(eval_dataset, indices)

    logger.info("  Num eval examples = %d", len(eval_data_subset))

    model.eval()
    eval_dataloader = get_dataloader(config, eval_data_subset, is_train=False) # is_train => shuffling

    losses = []
    all_query_features = []
    all_target_features = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_dataloader), desc="evaluating"):
            query_img_input, query_text_input, target_img_input = batch['query_img_input'], batch['query_text'], batch['target_img_input']

            query_img_input = query_img_input.to(torch.device(config.device))
            # query_text_input = query_text_input.to(torch.device(config.device)) # query_text_input is now plain text
            target_img_input = target_img_input.to(torch.device(config.device))
            
            # calculating loss
            if config.loss == 'soft_triplet':
                loss = model.compute_loss(
                    query_img_input, query_text_input, target_img_input, soft_triplet_loss=True)
            elif config.loss == 'batch_based_classification':
                loss = model.compute_loss(
                    query_img_input, query_text_input, target_img_input, soft_triplet_loss=False)
            else:
                print('Invalid loss function', config.loss)
                sys.exit()

            composition_features = model.compose_img_text(query_img_input, query_text_input)
            target_image_features = model.extract_img_feature(target_img_input)

            all_query_features.append(composition_features.detach().cpu())
            all_target_features.append(target_image_features.detach().cpu())

            if config.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            losses.append(loss.item())

        all_query_features = torch.vstack(all_query_features)
        all_target_features = torch.vstack(all_target_features)

        # putting into cpu
        all_query_features = all_query_features.cpu()
        all_target_features = all_target_features.cpu()


        # normalizing
        # normalized features
        all_query_features = all_query_features / all_query_features.norm(dim=-1, keepdim=True)
        all_target_features = all_target_features / all_target_features.norm(dim=-1, keepdim=True)

        similarity = all_query_features @ all_target_features.t()

        sorted_targets = torch.argsort(similarity, dim=1, descending=True)

    recall_values = {}
    for k in [1, 5, 10, 50, 100]:
        r=0.0
        for i in range(all_query_features.shape[0]):
            if i in sorted_targets[i][:k]:
                r += 1
        r = r/all_query_features.shape[0]
        recall_values[f"recall_at_{k}"] = r
    
    return recall_values, sum(losses)/len(losses)

def train(config, train_dataset, eval_dataset, model, optimizer):
    '''
    Trains the model.
    '''
    
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)    
    train_dataloader = get_dataloader(config, train_dataset, is_train=True)

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps \
                * config.num_train_epochs
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(torch.device(config.device))
    model.train()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)

    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step, global_loss, global_acc = 0, 0.0, 0.0
    best_recall_at_5 = 0.0
    best_epoch = 0
    
    val_losses = []

    model.zero_grad()

    for epoch in range(int(config.num_train_epochs)):
        
        model.train()
        for step, batch in enumerate(train_dataloader):
            query_img_input, query_text_input, target_img_input = batch['query_img_input'], batch['query_text'], batch['target_img_input']

            query_img_input = query_img_input.to(torch.device(config.device))
            # query_text_input = query_text_input.to(torch.device(config.device)) # it is a text now
            target_img_input = target_img_input.to(torch.device(config.device))
            
            # calculating loss
            if config.loss == 'soft_triplet':
                loss = model.compute_loss(
                    query_img_input, query_text_input, target_img_input, soft_triplet_loss=True)
            elif config.loss == 'batch_based_classification':
                loss = model.compute_loss(
                    query_img_input, query_text_input, target_img_input, soft_triplet_loss=False)
            else:
                print('Invalid loss function', config.loss)
                sys.exit()

            if config.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            global_loss += loss.item()

            # getting batch accuracy
            ######################################
            composition_features = model.compose_img_text(query_img_input, query_text_input)
            target_image_features = model.extract_img_feature(target_img_input)
            composition_features_norm =  composition_features / composition_features.norm(dim=-1, keepdim=True)
            target_image_features_norm = target_image_features / target_image_features.norm(dim=-1, keepdim=True)

            similarity = composition_features_norm @ target_image_features_norm.t()
            # print("similarity.shape = ", similarity.shape) # 
            sorted_targets = torch.argsort(similarity, dim=1, descending=True)
            predicted_labels = sorted_targets[:, 0]
            predicted_labels = predicted_labels.cpu() # changing it to cpu
            correct_predictions = predicted_labels == torch.arange(predicted_labels.shape[0])
            batch_acc = torch.sum(correct_predictions)/len(correct_predictions)
            global_acc = batch_acc.item()
            ########################################

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step() 
                    
                model.zero_grad()

                if global_step % config.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, train loss: {:.4f} ({:.4f}), acc: {:.4f}".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step, global_acc / global_step)
                    )

        # seeing performamce on validation set
        recall_values, val_loss = evaluate(config, eval_dataset, model)
        val_losses.append(val_loss)
        recall_at_5 = recall_values['recall_at_5']

        # lr scheduling
        if scheduler:
            scheduler.step(recall_at_5)

        logger.info("Epoch: {}, global_step: {}, val loss: {:.4f}".format(epoch, global_step, 
                        loss.item() / len(eval_dataset))
                    )
        logger.info("recall@5: {:.4f}, recall@10: {:.4f}, recall@50: {:.4f}, recall@100: {:.4f}".format(recall_at_5, recall_values['recall_at_10'], recall_values['recall_at_50'], recall_values['recall_at_100']))

        # if current validation accuracy is the best seen during training, save model
        if recall_at_5 > best_recall_at_5:
            best_recall_at_5 = recall_at_5
            # saving model
            save_checkpoint(config, epoch, global_step, model, optimizer)             
            best_epoch = epoch
        
    logger.info(f"Best seen model at epoch: {best_epoch}")
    return global_step, global_loss / global_step


def save_checkpoint(config, epoch, global_step, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(config.saved_checkpoints, f'checkpoint_best_{config.model}.pt')

    # saving with epoch in the filename
    # checkpoint_path = os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_{global_step}.pt')
    save_num = 0
    while (save_num < 10):
        try:

            if config.n_gpu > 1:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            logger.info("Save checkpoint to {}".format(checkpoint_path))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return

###
# OG methods

def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""
    print('Creating model and optimizer for', opt.model)
    if opt.model == 'imgonly':
        model = composition_models.SimpleModelImageOnly(
                texts, embed_dim=opt.embed_dim)
    elif opt.model == 'textonly':
        model = composition_models.SimpleModelTextOnly(
                texts, embed_dim=opt.embed_dim)
    elif opt.model == 'concat':
        model = composition_models.Concat(texts, embed_dim=opt.embed_dim)
    elif opt.model == 'tirg':
        model = composition_models.TIRG(texts, embed_dim=opt.embed_dim)
    elif opt.model == 'tirg_lastconv':
        model = composition_models.TIRGLastConv(
                texts, embed_dim=opt.embed_dim)
    else:
        print('Invalid model', opt.model)
        print('available: imgonly, textonly, concat, tirg or tirg_lastconv')
        sys.exit()
    
    if torch.cuda.is_available():
        model = model.cuda()

    # create optimizer
    params = []
    # low learning rate for pretrained layers on real image datasets
    params.append({
            'params': [p for p in model.img_model.fc.parameters()],
            'lr': opt.learning_rate
    })
    params.append({
            'params': [p for p in model.img_model.parameters()],
            'lr': 0.1 * opt.learning_rate
    })
    params.append({'params': [p for p in model.parameters()]})
    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)
    
    optimizer = Adam(params, lr=opt.learning_rate)

    return model, optimizer
##########################################33


def main():

    opt = parse_opt()
    print('Arguments:')
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)

    config = OmegaConf.merge(train_config, data_config)

    config = OmegaConf.merge(OmegaConf.create(vars(opt)), config)  

    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    logger = setup_logger("dlcv_project", config.logs, 0, filename = f"training_logs_{config.model}.txt")

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count() # config.n_gpu 
    set_seed(seed=11, n_gpu=config.n_gpu)

    # getting text tokenizer
    tokenizer = SimpleTokenizer()
    
    logger.info(f"Training/evaluation parameters {train_config}")
    
    # getting dataset for training/validation
    train_dataset = STIRDataset(data_config, tokenizer, split='train')
    val_dataset = STIRDataset(data_config, tokenizer, split='val')

    model, optimizer = create_model_and_optimizer(opt, train_dataset.get_all_texts())

    # evaluate saved model
    if config.do_eval:
        checkpoint_path = config.checkpoint_path
        assert checkpoint_path is not None
        assert os.path.isfile(checkpoint_path)

        logger.info(f"Loading saved checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(torch.device(config.device))
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f"Loaded saved checkpoint saved at training epoch {epoch}")

        recall_values, val_loss = evaluate(config, val_dataset, model)
        logger.info("Validation loss: {:.4f}, val R@1: {:.4f}, val R@5: {:.4f}".format(val_loss, recall_values['recall_at_1'], 
                        recall_values['recall_at_5'])
                    )
        write_json(recall_values, 'recall_values.json')
        
    else:
        # Now training
        if config.checkpoint_path:
            logger.info(f"Loading checkpoint from {config.checkpoint_path}")
            checkpoint_path = config.checkpoint_path
            assert checkpoint_path is not None
            assert os.path.isfile(checkpoint_path)

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(torch.device(config.device))
        global_step, avg_loss = train(config, train_dataset, val_dataset, model, optimizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

if __name__ == "__main__":
    main()