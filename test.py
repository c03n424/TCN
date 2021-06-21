import argparse,utils
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils import *
from model import *
import pickle
from random import randint
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=150, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='/home/visionlab/Wenjing/TCN_Test/TCN/word_cnn/data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')

parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate (default: 0.1)')

parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)

eval_batch_size = 50
#-----------------------------------------------------------------------------------
# data processing
ExpName = 'V1_0.1_600'
#学习速度0.1 jiezhe 40  jixu xunlian
T_train = 24 * 7 * 2 
T_predict = 24 * 7
test_split_p = 0.2

datapath = '/home/visionlab/Wenjing/data/'
house_data =  datapath + str(T_train) + '_' + str(T_predict) + '/'

result_path = '/home/visionlab/Wenjing/result/'
figure_save_path = result_path + '/' + ExpName +  '/img/'
eval_path = result_path + '/' + ExpName +  '/Eval_img/'
writer_path = result_path + '/' + ExpName +'/log/'
model_path = result_path + '/' + ExpName +'/model_' + str(args.nhid) + '_' + str(args.levels) + '_' + str(args.lr) + '/'
tar_path = result_path + '/' + ExpName + '/code.tar.gz'

if  not os.path.exists(eval_path):
    os.makedirs(eval_path)

if  not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)

if  not os.path.exists(writer_path):
    os.makedirs(writer_path)

if  not os.path.exists(model_path):
    os.makedirs(model_path)

writer = SummaryWriter(writer_path)
syn_train_csv = house_data + '/all_data.csv'
utils.make_targz(tar_path, '.')

print(syn_train_csv)
# Load Synthetic dataset
train_dataset, test_dataset = utils.get_data_from_csv(syn_dir=house_data, read_from_csv=syn_train_csv,  test_split=test_split_p)

syn_train_dl  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
syn_test_dl   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#-----------------------------------------------------------------------------------

num_chans = [args.nhid] * args.levels

k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = False

model = MY_TCN(1, num_chans, dropout=dropout, kernel_size=k_size)


# May use adaptive softmax to speed up training
criterion = nn.CrossEntropyLoss()
l2_loss = nn.MSELoss().cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


print('******************************')
print('******************************')
print('************* bix {:5d} ************'.format(len(syn_train_dl)))
print('******************************')
print('******************************')
def evaluate():
    model.eval()
    total_loss = 0
    iter_flag = 0
    print('************* bix {:5d} ************'.format(len(syn_test_dl)))

    with torch.no_grad():
        for bix, data in enumerate(syn_test_dl):
                train_sequence, index = data
                train_sequence =  train_sequence.cuda()
                input = train_sequence[:,0:24 * 7 * 2 ]
                targets = train_sequence[:,24 * 7 * 2:train_sequence.shape[1]]
                optimizer.zero_grad()
                output = model(input)

                loss = l2_loss(output, targets)
                total_loss += loss.item()
                iter_flag +=1
                print(bix)
                # utils.point_plot(targets, output, eval_path, 0, bix)
                # if bix > 20


        return total_loss / iter_flag

if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        path = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/'
        path__20 = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/20_predict_T.pkl'
        writer1 = SummaryWriter(path)
        model.load_state_dict(torch.load(path__20, map_location={'cuda:1':'cuda:0'}))
        model.cuda()
        avg_loss = evaluate()
        writer1.add_scalar('avg_eva_loss', avg_loss, 1)
        print('*****************************************************')

        path__40 = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/40_predict_T.pkl'
        model.load_state_dict(torch.load(path__40, map_location={'cuda:1':'cuda:0'}))
        model.cuda()
        avg_loss = evaluate()
        writer1.add_scalar('avg_eva_loss', avg_loss, 2)
        print('*****************************************************')


        path__60 = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/60_predict_T.pkl'
        model.load_state_dict(torch.load(path__60, map_location={'cuda:1':'cuda:0'}))
        model.cuda()
        avg_loss = evaluate()
        writer1.add_scalar('avg_eva_loss', avg_loss, 3)
        print('*****************************************************')

        
        path__80 = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/80_predict_T.pkl'
        model.load_state_dict(torch.load(path__80, map_location={'cuda:1':'cuda:0'}))
        model.cuda()
        avg_loss = evaluate()
        writer1.add_scalar('avg_eva_loss', avg_loss, 4)
        print('*****************************************************')



        path__100 = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/100_predict_T.pkl'
        model.load_state_dict(torch.load(path__100, map_location={'cuda:1':'cuda:0'}))
        model.cuda()
        avg_loss = evaluate()
        writer1.add_scalar('avg_eva_loss', avg_loss, 5)
        print('*****************************************************')


        path__120 = '/home/visionlab/Wenjing/result/V11_0.1_600/model_600_4_0.1/120_predict_T.pkl'
        model.load_state_dict(torch.load(path__120, map_location={'cuda:1':'cuda:0'}))
        model.cuda()
        avg_loss = evaluate()
        writer1.add_scalar('avg_eva_loss', avg_loss, 6)
        print('*****************************************************')






    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
