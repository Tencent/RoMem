# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch

_DEVICE = None

def get_device():
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
        else:
            _DEVICE = torch.device('cpu')
    return _DEVICE

def shredFacts(facts): #takes a batch of facts and shreds it into its columns
    dev = get_device()
    heads      = torch.tensor(facts[:,0]).long().to(dev)
    rels       = torch.tensor(facts[:,1]).long().to(dev)
    tails      = torch.tensor(facts[:,2]).long().to(dev)
    years = torch.tensor(facts[:,3]).float().to(dev)
    months = torch.tensor(facts[:,4]).float().to(dev)
    days = torch.tensor(facts[:,5]).float().to(dev)
    return heads, rels, tails, years, months, days
