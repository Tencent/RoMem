### Note
Unofficial implementation. The code use the loss function with reciprocal learning, which is different from the original paper's method.


### Datasets
NOTE: I only test the codes on the dataset---ICEWS14.

```
python process_icews.py

python process_gdelt.py
```

This will create the files required to compute the filtered metrics.

### Reproducing results of ChronoR

run the following commands

```
python learner.py --dataset ICEWS14 --rank 800 --k=3 --ratio=0.1 --emb_reg 0.01 --time_reg 0.01
python learner.py --dataset ICEWS14 --rank 1600 --k=2 --ratio=0.1 --emb_reg 0.01 --time_reg 0.01
```

I got the following results:

|     |  MRR    | hits@1 | hits@3 | hits@ 10 |
|-----|---------|--------|--------|----------|
| k=3 |  0.612  |  0.529 | 0.661  |   0.762  |
| k=2 |  0.617  |  0.536 | 0.664  |   0.768  |

### QA
If you have any questions or some measures to improve the scores, please contact me.

