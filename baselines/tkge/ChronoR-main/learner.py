import argparse
from typing import Dict
import torch
from torch import optim
from datasets import TemporalDataset
from optimizers import TKBCOptimizer
from models import ChronoR
from regularizers import N3, Lambda3
import os
import time


parser = argparse.ArgumentParser(
    description="ChronoR"
)
parser.add_argument(
    '--dataset', type=str, default='ICEWS14',
    help="Dataset name"
)

parser.add_argument(
    '--model', default='ChronoR', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=1600, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--k', default=3, type=int,
    help="k-dimensional real space"
)
parser.add_argument(
    '--ratio', default=0.1, type=float,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=0.1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)


args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          k = args.k,
          ratio = args.ratio,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          device = 'cpu'
          ):


    root = 'results/'+ dataset +'/' + model
    modelname = model
    datasetname = dataset

    # rank是秩,默认为2000
    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/k{:.0f}/ratio{:.2f}/batch{:.0f}/emb_reg{:.5f}/time_reg{:.5f}/'.format(rank,learning_rate,k,ratio,batch_size, emb_reg, time_reg))
    # 确保目录存在
    # 如果 exist_ok 为 False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
    # 如果 exist_ok 为 True，则在目标目录已存在的情况下不会触发 FileExistsError 异常
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=False)

    # 创建空文件
    # with open(os.path.join(PATH, 'result.txt'), 'w') as f:
    #     pass
    # begin time
    begin_time = time.gmtime()
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S",begin_time)
    f = open(os.path.join(PATH, 'result.txt'), 'a+')
    f.write("[begin time:{}]".format(begin_time))
    f.write('\nrank: {:.0f}\tlr: {:.4f}\tk: {:.0f}\tratio: {:.2f}\tbatch: {:.0f}\temb_reg: {:.5f}\ttime_reg: {:.5f}'.format(rank, learning_rate, k, ratio, batch_size, emb_reg, time_reg))
    f.close()
    dataset = TemporalDataset(dataset,device=device)
    '''
    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps
    '''
    sizes = dataset.get_shape()
    # 其实就等于model = ChronoR(sizes, rank, no_time_emb=args.no_time_emb)
    model = {
        'ChronoR': ChronoR(sizes, rank, k, ratio, no_time_emb=args.no_time_emb,device=device)
    }[model]
    model = model.to(device)


    opt = optim.Adagrad(model.parameters(), lr=learning_rate)

    print("Start training process: ", modelname, "on", datasetname, "using", "rank =", rank, "k = ", k, "ratio = ", ratio, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg)

    emb_reg = N3(emb_reg)
    time_reg = Lambda3(time_reg)
  
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}
    # 最多200轮
    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        '''
        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)
        def get_train(self):
            # 将(s,r,o,t)和(o,r^-1,s,t)一起得到
        '''
        # 即shape (num of train * 2,4)
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()

        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=batch_size,device=device
        )

        optimizer.epoch(examples)
       
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:

            if dataset.interval: 
                valid, test = [
                    avg_both(*dataset.eval(model, split, -1))
                    for split in ['valid', 'test']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])

            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results
            f = open(os.path.join(PATH, 'result.txt'), 'a+')
            f.write("\n[Epoch:{}]-TRAIN : ".format(epoch + 1))
            f.write(str(train))
            f.write("\n[Epoch:{}]-VALID : ".format(epoch + 1))
            f.write(str(valid))
            f.close()
            # early-stop with patience
            mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 10:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            if not dataset.interval:
                curve['train'].append(train)
    
                print("\t TRAIN: ", train)
            print("\t VALID : ", valid)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)

    # end time
    end_time = time.gmtime()
    end_time = time.strftime("%Y-%m-%d %H:%M:%S",end_time)
    f = open(os.path.join(PATH, 'result.txt'), 'a+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.write("\n[end time:{}]".format(end_time))
    f.close()

if __name__ == '__main__':
    cuda_id = int(os.environ.get('CUDA_DEVICE', '0'))
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    learn(device=device)


