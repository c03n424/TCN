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

parser.add_argument('--lr', type=float, default=0.0002,
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

eval_batch_size = 100
#-----------------------------------------------------------------------------------
# data processing
ExpName = 'V1_0.001_600_start_144_104_136_21'
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

writer_flag = 0
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
# model_PKL_path = '/home/visionlab/Wenjing/result/V1_0.1_600/model_600_4_0.1/144_predict_T.pkl'
# model_PKL_path = '/home/visionlab/Wenjing/result/V1_0.1_600_start_144/model_600_4_0.1/104_predict_T.pkl'
#model_PKL_path = '/home/visionlab/Wenjing/result/V1_0.001_600_start_144_104/model_600_4_0.001/136_predict_T.pkl'

model_PKL_path = '/home/visionlab/Wenjing/result/V1_0.001_600_start_144_104_136/model_600_4_0.0002/21_predict_T.pkl'



model.load_state_dict(torch.load(model_PKL_path))

model.cuda()

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
    with torch.no_grad():
        for bix, data in enumerate(syn_train_dl):
                train_sequence, index = data
                train_sequence =  train_sequence.cuda()
                input = train_sequence[:,0:24 * 7 * 2 ]
                targets = train_sequence[:,24 * 7 * 2:train_sequence.shape[1]]
                optimizer.zero_grad()
                output = model(input)

                loss = l2_loss(output, targets)
                total_loss += loss.item()
                iter_flag +=1
                # utils.point_plot(targets, output, eval_path, 0, bix)
                if iter_flag==1:
                    utils.point_plot(targets, output, figure_save_path, epoch, bix)
        # break      
    print('.......evaluate done....................')
    return total_loss / iter_flag




def train():
    # Turn on training mode which enables dropout.
    global train_data
    global writer_flag
    model.train()
    total_loss = 0
    start_time = time.time()
    flag = 0
    print('the training data length is ' + str(len(syn_train_dl)))
    for bix, data in enumerate(syn_train_dl):
        train_sequence, index = data
        train_sequence =  train_sequence.cuda()
        input = train_sequence[:,0:24 * 7 * 2 ]
        targets = train_sequence[:,24 * 7 * 2:train_sequence.shape[1]]
        optimizer.zero_grad()
        output = model(input)
        train_loss = l2_loss(output, targets)
        writer.add_scalar('train_loss', train_loss.item(), writer_flag)

        flag +=1
        writer_flag+=1
        total_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        # print(bix)
        print('| epoch {:3d} | bix/length {:4d}/{:4d} | train_loss {:5.10f} |'.format(epoch, bix, len(syn_train_dl), train_loss))

     


    elapsed = time.time() - start_time
    print('******************************************************************')
    print('******************************************************************')
    print('| epoch {:3d} | bix  | avg_loss {:5.10f} | elapsed {:5.5f}'.format(
                epoch, bix, total_loss /flag, elapsed * 1000))
    print('******************************************************************')
    print('******************************************************************')
    total_loss = 0
    start_time = time.time()



if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            print('______________________________')
            print('________ epoch {:3d} _________'.format(epoch))
            print('______________________________')
            epoch_start_time = time.time()
            train()
            torch.save(model.state_dict(), model_path + str(epoch) +  '_predict_T.pkl')
            avg_eval = evaluate()
            writer.add_scalar('AVG_eva_loss', avg_eval, epoch)
            print('^^^^^{:3d}epoch done'.format(epoch))

        


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
