# Modify from https://github.com/S-Abdelnabi/awt
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
import data
import lang_model

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden
from fb_semantic_encoder import BLSTMEncoder

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00003,
                    help='initial learning rate')
parser.add_argument('--disc_lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')
parser.add_argument('--fixed_length', type=int, default=0,
                    help='whether to use a fixed input length (bptt value)')
parser.add_argument('--dropout_transformer', type=float, default=0.1,
                    help='dropout applied to transformer layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash,
                    help='path to save the final model')
parser.add_argument('--save_interval', type=int, default=20,
                    help='saving models regualrly')
					
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

#message arguments
parser.add_argument('--msg_len', type=int, default=64,
                    help='The length of the binary message')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The total number of messages')
parser.add_argument('--msg_in_mlp_layers', type=int, default=1,
                    help='message encoding FC layers number')
parser.add_argument('--msg_in_mlp_nodes', type=list, default=[],
                    help='nodes in the MLP of the message')

#transformer arguments
parser.add_argument('--attn_heads', type=int, default=4,
                    help='The number of attention heads in the transformer')
parser.add_argument('--encoding_layers', type=int, default=3,
                    help='The number of encoding layers')
parser.add_argument('--shared_encoder', type=bool, default=True,
                    help='If the message encoder and language encoder will share weights')

#adv. transformer arguments
parser.add_argument('--adv_attn_heads', type=int, default=4,
                    help='The number of attention heads in the adversary transformer')
parser.add_argument('--adv_encoding_layers', type=int, default=3,
                    help='The number of encoding layers in the adversary transformer')

#gumbel softmax arguments
parser.add_argument('--gumbel_temp', type=int, default=0.5,
                    help='Gumbel softmax temprature')

#Adam optimizer arguments
parser.add_argument('--scheduler', type=int, default=1,
                    help='whether to schedule the lr according to the formula in: Attention is all you need')
parser.add_argument('--warm_up', type=int, default=6000,
                    help='number of linear warm up steps')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='Adam beta1 parameter')
parser.add_argument('--beta2', type=float, default=0.98,
                    help='Adam beta2 parameter')
parser.add_argument('--eps', type=float, default=1e-9,
                    help='Adam eps parameter')
#GAN arguments
parser.add_argument('--msg_weight', type=float, default=25,
                    help='The factor multiplied with the message loss')

#fb InferSent semantic loss 
parser.add_argument('--use_semantic_loss', type=int, default=1,
                    help='whether to use semantic loss')
parser.add_argument('--glove_path', type=str, default='sent_encoder/GloVe/glove.840B.300d.txt',
                    help='path to glove embeddings')
parser.add_argument('--infersent_path', type=str, default='sent_encoder/infersent2.pkl',
                    help='path to the trained sentence semantic model')
parser.add_argument('--sem_weight', type=float, default=40,
                    help='The factor multiplied with the semantic loss')
					
#language loss
parser.add_argument('--use_lm_loss', type=int, default=1,
                    help='whether to use language model loss')
parser.add_argument('--lm_weight', type=float, default=1,
                    help='The factor multiplied with the lm loss')
parser.add_argument('--lm_ckpt', type=str, default='WT2_lm.pt',
                    help='path to the fine tuned language model')
					
#reconstruction loss
parser.add_argument('--use_reconst_loss', type=int, default=1,
                    help='whether to use language reconstruction loss')
parser.add_argument('--reconst_weight', type=float, default=1,
                    help='The factor multiplied with the reconstruct loss')

#lang model params.
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize_lm', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti_lm', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute_lm', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
#GAN arguments			
parser.add_argument('--discr_interval', type=int, default=1,
                    help='when to update the discriminator')
parser.add_argument('--autoenc_path', type=str, default='',
                    help='path of the autoencoder path to use as init to the generator, in case the model is pretrained as autoencoder only')
parser.add_argument('--gen_weight', type=float, default=2,
                    help='The factor multiplied with the gen loss')

parser.add_argument('--eval_train', action='store_true', default=False,
                    help='eval on train dataset')
parser.add_argument('--eval_valid', action='store_true', default=False,
                    help='eval on valid dataset')


args = parser.parse_args()
args.tied = True
args.tied_lm = True
suffix = f"{args.data.split('/')[-1]}{'_train' if args.eval_train else ''}{'_valid' if args.eval_valid else ''}{'_test' if not args.eval_train and not args.eval_valid else ''}"

os.makedirs(f"./data/output_{args.save}{suffix}/data")
os.makedirs(f"./data/output_{args.save}{suffix}/data_out")
os.makedirs(f"./data/output_{args.save}{suffix}/msgs")
os.makedirs(f"./data/output_{args.save}{suffix}/msgs_out")
os.makedirs(f"./data/output_{args.save}{suffix}/msgs_out_sig")

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_load(fn):
    global model_gen, criterion, criterion_reconst, optimizer_gen
    with open(fn+'_gen.pt', 'rb') as f:
        model_gen, criterion, criterion_reconst, optimizer_gen = torch.load(f,map_location='cpu')
		
import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

criterion = None
criterion_reconst = None

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word


## global variable for the number of steps ( batches) ##
step_num = 1
discr_step_num = 1


all_msgs = generate_msgs(args)


#convert word ids to text	
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list

### conventions for real and fake labels during training ###
real_label = 1
fake_label = 0

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10, klass="test"):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    # model_disc.eval()
		
    total_loss_gen = 0
    total_loss_disc = 0
    total_loss_msg = 0
    total_loss_sem = 0
    total_loss_reconst = 0
    total_loss_lm = 0
    
    total_encoding_time = 0.
    total_decoding_time = 0.
	
    batches_count = 0
    i = 0
    code_num = 0

    with open(os.path.join(args.data, f"{klass}.txt"), "r") as f:
        seq_lens = [len([y for y in x.strip().split() if y]) for x in f.readlines()]
    if os.path.exists(os.path.join(args.data, f"{klass}_identifiers.txt")):
        with open(os.path.join(args.data, f"{klass}_identifiers.txt"), "r") as f:
            identifiers = [list(map(lambda y: corpus.dictionary.word2idx[y], filter(lambda y: y in corpus.dictionary.word2idx, x.strip().split()))) for x in f.readlines()]
    else:
        identifiers = [[] for _ in seq_lens]

    while i < data_source.size(0):
        # data, msgs, targets = get_batch_different(data_source, i, args,all_msgs, evaluation=True)
        if not args.fixed_length:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = 0
            while seq_len < 1 and code_num < len(seq_lens):
                seq_len += seq_lens[code_num]
                code_num += 1
            if code_num >= len(seq_lens) and seq_len == 0:
                break
            # if seq_len == 0:
            #     with open(f"data/output_{args.save}/data/data{batches_count}.txt", "w") as f:
            #         f.write('')
            #     with open(f"data/output_{args.save}/data_out/data_out{batches_count}.txt", "w") as f:
            #         f.write('')
            #     with open(f"data/output_{args.save}/msgs/msgs{batches_count}.txt", "w") as f:
            #         f.write('0 0 0 0')
            #     with open(f"data/output_{args.save}/msgs_out/msgs_out{batches_count}.txt", "w") as f:
            #         f.write('1 1 1 1')
            #     with open(f"data/output_{args.save}/msgs_out_sig/msgs_out_sig{batches_count}.txt", "w") as f:
            #         f.write('1 1 1 1')
            #     continue
            ori_seq_len = seq_len
            # seq_len = min(10, seq_len)
            data, msgs, targets = get_batch_different(data_source, i, args,all_msgs, seq_len=seq_len, evaluation=True)
        else:
            seq_len = args.bptt
            ori_seq_len = seq_len
            data, msgs, targets = get_batch_different(data_source, i, args, all_msgs, seq_len=None, evaluation=True)
        #get a batch of fake (edited) sequence from the generator
        start_time = time.time()
        identifiers_mask = [[0 if x not in identifiers[code_num - 1] else 0 for x in range(ntokens)]]
        identifiers_mask = torch.tensor(identifiers_mask, dtype=int).cuda()
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp, identifiers_mask=identifiers_mask)
        total_encoding_time += time.time() - start_time
        start_time = time.time()
        # print(data.shape, fake_data_emb.shape, seq_len)
        msg_out = model_gen.forward_msg_decode(fake_data_emb)
        total_decoding_time += time.time() - start_time
        #get prediction (and the loss) of the discriminator on the real sequence. First gen the embeddings from the generator
        # data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)
        # real_out = model_disc(data_emb)
        # label = torch.full( (data.size(1),1), real_label)
        # if args.cuda:
        #     label = label.cuda()
        # errD_real = criterion(real_out,label.float())
        #get prediction (and the loss) of the discriminator on the fake sequence.
        # fake_out = model_disc(fake_data_emb.detach())
        # label.fill_(fake_label)
        # errD_fake = criterion(fake_out,label.float())
        # errD = errD_real + errD_fake

        #generator loss
        # label.fill_(real_label) 
        # errG_disc = criterion(fake_out,label.float())

        #msg loss of the generator
        msg_loss = criterion(msg_out, msgs)
        with open(f"data/output_{args.save}{suffix}/data/data{batches_count}.txt", "w") as f:
            f.write(''.join([' '.join(x) for x in convert_idx_to_words(data)]))
        with open(f"data/output_{args.save}{suffix}/data_out/data_out{batches_count}.txt", "w") as f:
            data_out = torch.argmax(fake_one_hot, dim=-1)
            f.write(''.join([' '.join(x) for x in convert_idx_to_words(data_out)]))
        np.savetxt(f"data/output_{args.save}{suffix}/msgs/msgs{batches_count}.txt", msgs.detach().cpu().numpy())
        np.savetxt(f"data/output_{args.save}{suffix}/msgs_out/msgs_out{batches_count}.txt", msg_out.detach().cpu().numpy())
        np.savetxt(f"data/output_{args.save}{suffix}/msgs_out_sig/msgs_out_sig{batches_count}.txt", torch.nn.Sigmoid()(msg_out.detach().cpu()).numpy())

        #reconstruction loss 
        reconst_loss = criterion_reconst(fake_data_prob,data.view(-1))
        total_loss_reconst += reconst_loss.data

        # total_loss_gen +=  errG_disc.data
        # total_loss_disc +=  errD.data
        total_loss_msg += msg_loss.data
        batches_count = batches_count + 1
        i += ori_seq_len
    if args.use_semantic_loss: 
        total_loss_sem = total_loss_sem.item()
    if args.use_lm_loss: 
        total_loss_lm = total_loss_lm.item() 		
    
    print("Encoding Time:", total_encoding_time / batches_count)
    print("Decoding Time:", total_decoding_time / batches_count)

    return total_loss_reconst.item()/batches_count, total_loss_gen / batches_count, total_loss_msg.item() / batches_count, total_loss_sem / batches_count, total_loss_lm/batches_count, total_loss_disc / batches_count


# Load the best saved model.
model_load(args.save)
if args.cuda:
    model_gen = model_gen.cuda()
    # model_disc = model_disc.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()


# Run on test data.
if args.eval_train:
    train_data = batchify(corpus.train, test_batch_size, args)
    test_loss_reconst, test_loss_gen, test_loss_msg, test_loss_sem, test_loss_lm, test_loss_disc = evaluate(train_data, test_batch_size, "train")
elif args.eval_valid:
    valid_data = batchify(corpus.valid, test_batch_size, args)
    test_loss_reconst, test_loss_gen, test_loss_msg, test_loss_sem, test_loss_lm, test_loss_disc = evaluate(valid_data, test_batch_size, "valid")
else:
    test_loss_reconst, test_loss_gen, test_loss_msg, test_loss_sem, test_loss_lm, test_loss_disc = evaluate(test_data, test_batch_size)

print('-' * 89)
print('| End of training | test gen loss {:5.2f} | test disc loss {:5.2f} | test msg loss {:5.5f} | test sem loss {:5.2f} | test reconst loss {:5.2f} | test lm loss {:5.2f}'.format(test_loss_gen, test_loss_disc, test_loss_msg, test_loss_sem, test_loss_reconst, test_loss_lm))
print('-' * 89)

