import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from scripts.data_utils import TransliterationDataset, pad_collate
from scripts.models import Attention_seq2seq, Simple_seq2seq
from scripts.train_utils import masked_accuracy, masked_loss
from scripts.transliteration_tokenizers import create_source_target_tokenizers

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

PAD_ID = 0

def overfit_small_batch_testing(vocab_dict,sample_file,model_dict,title_suffix = None,batch_size = 4,iterations = 1000):
    """Overfits the model on a single batch to check whether model have enough capacity

    Args:
        vocab_dict (dict): Parameters required to create tokenizers
        sample_file (file): Sample file to extract a single batch
        model_dict (dict): Parameters required to create model
        title_suffix (str, optional): Title of experiment. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 4.
        iterations (int, optional): Number of max iterations to perform. Defaults to 1000.
    """
    
    #Creates tokenizers
    source_tokenizer, target_tokenizer = create_source_target_tokenizers(**vocab_dict)
    pad_id = target_tokenizer.padding['pad_id']
    
    #Creates datasets and dataloaders
    sample_dataset = TransliterationDataset(sample_file,source_tokenizer, target_tokenizer)
    batch = next(iter(DataLoader(sample_dataset, batch_size = batch_size, collate_fn=pad_collate)))
    
    #Selecting the appropriate model
    if model_dict['type'] == "simple_seq2seq":
        model = Simple_seq2seq(model_dict['embed_size'], model_dict['hidden_size'],
                               src_tokenizer = source_tokenizer, tgt_tokenizer = target_tokenizer)
        optimizer = SGD(model.parameters(), lr=model_dict['lr'])
        
    if model_dict['type'] == "attention_seq2seq":
        model = Attention_seq2seq(model_dict['embed_size'], model_dict['hidden_size'],src_tokenizer = source_tokenizer,\
                                  tgt_tokenizer = target_tokenizer, dropout_rate =model_dict["dropout_rate"])
        optimizer = SGD(model.parameters(), lr=model_dict['lr']) 
    
    model = model.to(device)
    model.train()
    
    src_sents,tgt_sents,src_lens = batch
    
    training_losses = []
    training_iteration = []
    training_accuracy = []
    
    
    for i in range(iterations):
        scores = model(batch)
        loss = masked_loss(scores,tgt_sents,PAD_ID)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = masked_accuracy(scores,tgt_sents,PAD_ID)
        
        training_losses.append(loss.item())
        training_iteration.append(i)
        training_accuracy.append(accuracy.item())
        if i % 50 == 0:
            print(f'Iteration {i}, loss = {round(loss.item(),4)},training accuracy = {round(accuracy.item(),4)}')
        if accuracy == 1:
            break
            
    #Printing predictions of mdoel at the ened
    print_predictions(src_sents,tgt_sents,scores,src_tokenizer = source_tokenizer,tgt_tokenizer = target_tokenizer)
    
    #Plotting the logs
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16,8),ncols=2)#,ncols=2

    ax1 = sns.lineplot(x = training_iteration,y = training_losses,label = 'Training Loss',ax = ax[0])
    ax1.set_xlabel('No. of Iterations',fontsize = 15)
    ax1.set_ylabel('Loss',fontsize = 15)
    ax1.set_title("Loss", fontsize = 18)

    ax2 = sns.lineplot(x = training_iteration,y = training_accuracy,label = 'Training Accuracy',ax = ax[1])
    ax2.set_xlabel('No. of Iterations',fontsize = 15)
    ax2.set_ylabel('Accuracy',fontsize = 15)
    ax2.set_title('Accuracy',  fontsize = 18)

    title = title_suffix
    fig.suptitle(title,size = 25)
    fig.get_tight_layout()
    #fig.show()  
    
def print_predictions(src_sents,tgt_sents,scores,src_tokenizer,tgt_tokenizer):
    """Prints the predictions of model after converting ids to words

    Args:
        src_sents (Tensor): Tensor of source ids
        tgt_sents (Tensor): Actual Tensor of target ids
        scores (Tensor): Vocab scores
        src_tokenizer (Tokenizer): Source Tokenizer
        tgt_tokenizer (Tokenizer): Target tokenizer
    """
    src_sents = src_sents[:,1:].tolist()
    tgt_sents = tgt_sents[:,1:].tolist()
    preds = torch.argmax(scores, dim=-1).tolist()
    print (tgt_sents)
    print(preds)
    end_id = tgt_tokenizer.token_to_id('</s>')
    src_words = ["".join(word.split()) for word in src_tokenizer.decode_batch(src_sents)]
    tgt_words = ["".join(word.split()) for word in tgt_tokenizer.decode_batch(tgt_sents)]
    pred_words = ["".join(word.split()) for word in tgt_tokenizer.decode_batch(preds)]
    examples = list(zip(src_words,tgt_words,pred_words))
    print(tgt_words)
    print(pred_words)
    for example in examples:
        print (f'{example[0]}   {example[1]} ------->   {example[2]}')
        
def train(vocab_dict,file_dict, model_dict, batch_size, storage_path,epochs,print_every = 100,check_every = 100,save = False):
    """Trains the model and saves it

    Args:
        vocab_dict (dict): Parameters required to create tokenizers
        file_dict (dict): Files needed to train 
        model_dict (dict): Parameters required to create model
        batch_size (int): Batch size
        storage_path (Path): Path where model is stored
        epochs (int): -NUmber of epochs to train
        print_every (int, optional): How often to print training logs. Defaults to 100.
        check_every (int, optional): How often to validate on validation set. Defaults to 100.
        save (bool, optional):Whether to save the model or not. Defaults to False.

    Returns:
        model: Model
        training log: Training log used to visualize the results
    """
    
    source_tokenizer, target_tokenizer = create_source_target_tokenizers(**vocab_dict)
    pad_id = target_tokenizer.padding['pad_id']
    
    train_dataset = TransliterationDataset(file_dict["train"],source_tokenizer, target_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn=pad_collate,shuffle =False)
    del train_dataset
    
    val_dataset = TransliterationDataset(file_dict["val"],source_tokenizer, target_tokenizer)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, collate_fn=pad_collate,shuffle = False)
    del val_dataset
    
    test_dataset = TransliterationDataset(file_dict["test"],source_tokenizer, target_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, collate_fn=pad_collate,shuffle =False)
    del test_dataset
    
    del file_dict
    
    if model_dict['type'] == "simple_seq2seq":
        model = Simple_seq2seq(model_dict['embed_size'], model_dict['hidden_size'],
                               src_tokenizer = source_tokenizer, tgt_tokenizer = target_tokenizer)
        optimizer = SGD(model.parameters(), lr=model_dict['lr'])
        
    if model_dict['type'] == "attention_seq2seq":
        model = Attention_seq2seq(model_dict['embed_size'], model_dict['hidden_size'],
                               src_tokenizer = source_tokenizer, tgt_tokenizer = target_tokenizer, dropout_rate = model_dict["dropout_rate"])
        optimizer = SGD(model.parameters(), lr=model_dict['lr'])       
          
    model = model.to(device)
    epoch_len = len(train_loader)
    training_losses = []
    training_iteration = []
    validation_losses = []
    validation_accuracy = []
    training_accuracy = []
    validation_iteration = []
    
    
    for epoch in range(epochs):
        print ('\n')
        print (f'Running epoch {epoch + 1}')
        print ('\n')
        
        for i, batch in enumerate(train_loader):
            
            iteration = epoch*epoch_len + i
            model.train()
            src_sents,tgt_sents,src_lens = batch
            scores = model(batch)
            del batch,src_sents,src_lens
            loss = masked_loss(scores,tgt_sents,pad_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % print_every == 0:
                training_iteration.append(iteration)
                training_losses.append(loss.item())
                accuracy = masked_accuracy(scores,tgt_sents,pad_id)
                training_accuracy.append(accuracy.item())
                print(f'Iteration {i}, loss = {round(loss.item(),4)},training accuracy = {round(accuracy.item(),4)}')
            
            del scores,tgt_sents
            
            if i % check_every == 0:
                validation_iteration.append(iteration)
                val_loss,val_accuracy = valid_step(model,val_loader)
                validation_losses.append(val_loss)
                validation_accuracy.append(val_accuracy)
                print(f'Iteration {i}, validation loss = {round(val_loss,4)},validation accuracy = {round(val_accuracy,4)}')
        
                
        if save:
            PATH = storage_path /  f'model_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict(),'loss': loss,}, PATH)

            
                    
    training_log = {'training_losses' : training_losses,
                    'training_accuracy'  : training_accuracy,
                    'training_iteration' : training_iteration,
                    'validation_losses' : validation_losses,
                    'validation_accuracy' : validation_accuracy,
                    'validation_iteration' : validation_iteration,
                    }

    return model,training_log

def valid_step(model,data_loader):
    """Performs the validation step

    Args:
        model (model): Model
        data_loader (DataLoader): Valildation data loader

    Returns:
        val_loss: Validation loss
        val_accuracy: Validation Accuracy
    """
    model.eval()
    val_loss = 0
    val_accuracy = 0
    
    for iteration ,(batch) in enumerate(data_loader):
        with torch.no_grad():
            src_sents,tgt_sents,src_lens = batch
            scores = model(batch)
            del batch,src_sents,src_lens
            
            loss = masked_loss(scores,tgt_sents,PAD_ID)
            accuracy = masked_accuracy(scores,tgt_sents,PAD_ID)
            del scores,tgt_sents
            
            val_loss += loss.item()
            val_accuracy += accuracy.item()
    
    val_loss /= (iteration+1)
    val_accuracy /= (iteration+1)
        
    return val_loss,val_accuracy

def plot_training_log(training_log,title):
    """Visualize the training metrics

    Args:
        training_log (dict): Dict consisting of training and validation metrics
        title (str): Name of the experiment
    """
    training_losses  = training_log['training_losses']
    training_iteration  = training_log['training_iteration']
    validation_losses  = training_log['validation_losses']
    training_accuracy  = training_log['training_accuracy']
    validation_accuracy  = training_log['validation_accuracy']
    validation_iteration  = training_log['validation_iteration']
    

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16,8),ncols=2)#,ncols=2

    ax1 = sns.lineplot(x = training_iteration,y = training_losses,label = 'Training Loss',ax = ax[0])
    ax1 = sns.lineplot(x = validation_iteration,y = validation_losses,label = 'Validation Loss',ax = ax[0])
    ax1.set_xlabel('No. of Iterations',fontsize = 15)
    ax1.set_ylabel('Loss',fontsize = 15)
    ax1.set_title("Loss", fontsize = 18)

    ax2 = sns.lineplot(x = training_iteration,y = training_accuracy,label = 'Training Accuracy',ax = ax[1])
    ax2 = sns.lineplot(x = validation_iteration,y = validation_accuracy,label = 'Validation Accuracy',ax = ax[1])
    ax2.set_xlabel('No. of Iterations',fontsize = 15)
    ax2.set_ylabel('Accuracy',fontsize = 15)
    ax2.set_title('Accuracy',  fontsize = 18)

    
    fig.suptitle(title,size = 25)
    fig.get_tight_layout()
    #fig.show()        
