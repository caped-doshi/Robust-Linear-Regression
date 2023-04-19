We will give scripts for each file with their default parameters

First install the necessary requirements in a Python 3 environment. 
```
pip install -r requirements.txt
```

Subquantile Minimization
```
usage: SubQuantile.py [-h] [--num_trials NUM_TRIALS] [--num_iters NUM_ITERS] [--noise NOISE] [--noise_type NOISE_TYPE]
                      [--quantile QUANTILE] [--dataset DATASET] [--n N] [--d D]

options:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        run how many times
  --num_iters NUM_ITERS
                        how many iterations of algorithm
  --noise NOISE         noise ratio in range (0, 1)
  --noise_type NOISE_TYPE
                        oblivious, adaptive, or feature
  --quantile QUANTILE   what quantile level to minimize over
  --dataset DATASET     dataset; drug, cal_housing, abalone, or synthetic
  --n N                 samples for synthetic data
  --d D                 dim for synthetic data
```

Sever
```
usage: sever.py [-h] [--num_trials NUM_TRIALS] [--iters ITERS] [--reg REG] [--p P] [--noise NOISE]
                [--noise_type NOISE_TYPE] [--dataset DATASET] [--n N] [--d D]

options:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        run how many times
  --iters ITERS         iterations of algorithm
  --reg REG             regularizer
  --p P                 fraction of outlier to remove (0:1)]
  --noise NOISE         noise ratio in range (0, 1)
  --noise_type NOISE_TYPE
                        oblivious, adaptive, or feature
  --dataset DATASET     dataset; drug, cal_housing, abalone, or synthetic
  --n N                 samples for synthetic data
  --d D                 dim for synthetic data
```

Consistent Robust Regression
```
usage: crr.py [-h] [--num_trials NUM_TRIALS] [--num_iters NUM_ITERS] [--noise NOISE] [--noise_type NOISE_TYPE]
              [--dataset DATASET] [--n N] [--d D]

options:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        run how many times
  --num_iters NUM_ITERS
                        how many iterations of algorithm
  --noise NOISE         noise ratio in range (0, 1)
  --noise_type NOISE_TYPE
                        oblivious, adaptive, or feature
  --dataset DATASET     dataset; drug, cal_housing, abalone, or synthetic
  --n N                 samples for synthetic data
  --d D                 dim for synthetic data
```

Robust Risk Minimization
```
usage: RRM.py [-h] [--num_trials NUM_TRIALS] [--max_iters MAX_ITERS] [--tol TOL] [--dist_max_iters DIST_MAX_ITERS]
              [--dist_tol DIST_TOL] [--noise NOISE] [--noise_type NOISE_TYPE] [--dataset DATASET] [--n N] [--d D]

options:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        run how many times
  --max_iters MAX_ITERS
                        max iterations of algorithm
  --tol TOL             tolerence for algorithm convergence
  --dist_max_iters DIST_MAX_ITERS
                        max iterations of distribution optimization
  --dist_tol DIST_TOL   tolerence for distribution optimization convergence
  --noise NOISE         noise ratio in range (0, 1)
  --noise_type NOISE_TYPE
                        oblivious, adaptive, or feature
  --dataset DATASET     dataset; drug, cal_housing, abalone, or synthetic
  --n N                 samples for synthetic data
  --d D                 dim for synthetic data
```

Sklearn Models (RANSAC, Ridge, Quantile, and Genie Regression). Genie runs Ridge with knowledge of where clean data is.
```
usage: sklearn_models.py [-h] [--method METHOD] [--num_trials NUM_TRIALS] [--num_iters NUM_ITERS] [--noise NOISE]
                         [--noise_type NOISE_TYPE] [--dataset DATASET] [--n N] [--d D]

options:
  -h, --help            show this help message and exit
  --method METHOD       RANSAC, Ridge, Quantile, or Genie
  --num_trials NUM_TRIALS
                        run how many times
  --num_iters NUM_ITERS
                        how many iterations of algorithm
  --noise NOISE         noise ratio in range (0, 1)
  --noise_type NOISE_TYPE
                        oblivious, adaptive, or feature
  --dataset DATASET     dataset; drug, cal_housing, abalone, or synthetic
  --n N                 samples for synthetic data
  --d D                 dim for synthetic data
```

STIR
```
usage: stir.py [-h] [--num_trials NUM_TRIALS] [--num_iters NUM_ITERS] [--noise NOISE] [--noise_type NOISE_TYPE]
               [--dataset DATASET] [--n N] [--d D]

options:
  -h, --help            show this help message and exit
  --num_trials NUM_TRIALS
                        run how many times
  --num_iters NUM_ITERS
                        how many iterations of algorithm
  --noise NOISE         noise ratio in range (0, 1)
  --noise_type NOISE_TYPE
                        oblivious, adaptive, or feature
  --dataset DATASET     dataset; drug, cal_housing, abalone, or synthetic
  --n N                 samples for synthetic data
  --d D                 dim for synthetic data
```
