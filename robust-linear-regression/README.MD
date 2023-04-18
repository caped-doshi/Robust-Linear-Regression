We will give scripts for each file with their default parameters

First install the necessary requirements in a Python 3 environment. 
```
pip install -r requirements.txt
```

Subquantile Minimization
```
python SubQuantile.py --num_trials 5 ^
  --num_iters 64 ^
  --noise 0.1 ^
  --noise_type oblivious ^
  --quantile 0.9 ^
  --dataset drug ^
  --n 2000 ^
  --d 200
```

Sever
```
python sever.py --num_trials 5 ^
  --num_iters 64 ^
  --reg 2 ^
  --noise 0.1 ^
  --noise_type oblivious ^
  --dataset drug ^
  --n 2000 ^
  --d 200
```

Consistent Robust Regression
```
python crr.py --num_trials 5 ^
  --num_iters 3000 ^
  --noise 0.1 ^
  --noise_type oblivious ^
  --dataset drug ^
  --n 2000 ^
  --d 200
```

Robust Risk Minimization
```
python RMM.py --num_trials 5 ^
  --max_iters 50 ^
  --tol 0.02 ^
  --dist_max_iters 10000 ^
  --dist_tol 0.001 ^
  --noise 0.1 ^
  --noise_type oblivious ^
  --dataset drug ^
  --n 2000 ^
  --d 200
```

Sklearn Models (RANSAC, Ridge, Quantile, and Genie Regression). Genie runs Ridge with knowledge of where clean data is.
```
python smart.py -- method Ridge ^
  --num_trials 5 ^
  --num_iters 32 ^
  --noise 0.1 ^
  --noise_type oblivious ^
  --dataset drug ^
  --n 2000 ^
  --d 200
```
