from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1,device='cpu'
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size] # batch_size, 4
                        # (batch_size,k,rank)
                        q = self.get_queries(these_queries)
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        # if hasattr(torch.cuda, 'empty_cache'):
                        #     torch.cuda.empty_cache()
                        scores = torch.einsum('bik,jkr->bjir', q, rhs).diagonal(dim1=-2,dim2=-1).sum(dim=-1)
                        # block_size = 10
                        # scores = torch.zeros((these_queries.shape[0], self.sizes[2])).to(self.device)
                        # for i in range(0,batch_size,block_size):
                        #     trace = torch.einsum('bik,jkr->bjir', q[i*block_size:(i+1)*block_size], rhs).diagonal(dim1=-2,dim2=-1).sum(dim=-1)
                        #     scores[i*block_size:(i+1)*block_size] = trace

                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            # 事实的o的id
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks



class ChronoR(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int, k: int, ratio: float, no_time_emb=False, init_size: float = 1e-2, device = 'cpu'):
        super(ChronoR, self).__init__()
        self.device = device
        self.sizes = sizes
        # rank指实体嵌入维度，ratio指n_r / n_t
        self.rank = rank
        # k就是论文中的k
        self.k = k
        # t_rank即time映射的维度
        self.t_rank = int(rank / (1 + ratio))

        #  the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from N(0,1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], k * rank, sparse=True),
            nn.Embedding(sizes[1], k * (rank - self.t_rank), sparse=True),
            nn.Embedding(sizes[3], k * self.t_rank, sparse=True),
            nn.Embedding(sizes[1], k * rank, sparse=True),
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

    @staticmethod
    def has_time():
        return True
	
    def score(self, x):
        # x shape:(batch_size,4)
        # head (batch_size,k * rank)
        batch_size = x.shape[0]
        head = self.embeddings[0](x[:, 0])
        relations = self.embeddings[1](x[:, 1]) 
        relations_2 = self.embeddings[3](x[:, 1]) 
        tail = self.embeddings[0](x[:, 2])
        times = self.embeddings[2](x[:, 3])
        # 由于tr(A^TB)=tr(AB^T)，所以直接反向构造
        head = head.reshape((batch_size, self.k, -1))
        relations = relations.reshape((batch_size, self.k, -1))
        relations_2 = relations_2.reshape((batch_size, self.k, -1))
        tail = tail.reshape((batch_size, self.k, -1))
        times = times.reshape((batch_size, self.k, -1))
        r_t = torch.cat((relations, times), dim=2)
        # shape: (batch_size,k,k)
        trace = torch.bmm(head * r_t * relations_2, tail.transpose(1,2))
        score = torch.diagonal(trace, dim1=-2, dim2=-1).sum(dim=-1,keepdim=True)
        # score shape: (batch_size,1)
        return score
	
    def forward(self, x):
        # x shape:(batch_size,4)
        batch_size = x.shape[0]
        head = self.embeddings[0](x[:, 0])
        relations = self.embeddings[1](x[:, 1]) 
        relations_2 = self.embeddings[3](x[:, 1]) 
        tail = self.embeddings[0](x[:, 2])
        times = self.embeddings[2](x[:, 3])
        # right 每一行相当于一个词典,即每个实体嵌入成的向量
        # right shape: (num_eneity, rank , k)
        right = self.embeddings[0].weight.reshape((self.sizes[0],self.k,self.rank)).transpose(1,2)
        
        head = head.reshape((batch_size, self.k, -1))
        relations = relations.reshape((batch_size, self.k, -1))
        relations_2 = relations_2.reshape((batch_size, self.k, -1))
        tail = tail.reshape((batch_size, self.k, -1))
        times = times.reshape((batch_size, self.k, -1))
        r_t = torch.cat((relations, times), dim=2)
        # l shape: (batch_size,k,rank)
        l = head * r_t * relations_2
        # score shape: (batch_size, self.sizes[0])
        score = torch.einsum('bik,jkr->bjir', l, right).diagonal(dim1=-2,dim2=-1).sum(dim=-1)
        # score = torch.zeros((batch_size, self.sizes[0])).to(self.device)
        # for i in range(0,batch_size,block_size):
        #     trace = torch.einsum('bik,jkr->bjir', l[i*block_size:(i+1)*block_size], right).diagonal(dim1=-2,dim2=-1).sum(dim=-1)
        #     score[i*block_size:(i+1)*block_size] = trace

        # @矩阵乘法
        # 第一个元组，手动计算(s,r,?,t),得到score(batch_size,num_entities)
        # 第二个，Λ4 regulation
        # 第三个元组time shape (num_time, k * t_rank)
        # 因为头尾实体集合相同，都是用的embeddings[0]映射，所以用right矩阵相乘后得到所有的score的预测
        return (score), (
                   torch.linalg.norm(head,ord=4,dim=1),
                   torch.linalg.norm(r_t,ord=4,dim=1),
                   torch.linalg.norm(relations_2,ord=4,dim=1),
                   torch.linalg.norm(tail,ord=4,dim=1),
               ),  self.embeddings[2].weight[:-1].reshape((self.sizes[3],self.k,-1)) if self.no_time_emb else self.embeddings[2].weight.reshape((self.sizes[3],self.k,-1))


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        # chunk_size = num_entities
        # return (num_entities, k, rank)
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size].reshape((-1, self.k, self.rank)).transpose(1,2)

    def get_queries(self, queries: torch.Tensor):
        # queries shape: (batch_size, 4)
        batch_size = queries.shape[0]
        head = self.embeddings[0](queries[:, 0])
        relations = self.embeddings[1](queries[:, 1])
        relations_2 = self.embeddings[3](queries[:, 1])
        times = self.embeddings[2](queries[:, 3])
        
        head = head.reshape((batch_size, self.k, -1))
        relations = relations.reshape((batch_size, self.k, -1))
        relations_2 = relations_2.reshape((batch_size, self.k, -1))
        times = times.reshape((batch_size, self.k, -1))
        r_t = torch.cat((relations, times), dim=2)
        # l shape: (batch_size,rank,k)
        l = head * r_t * relations_2
        return l

