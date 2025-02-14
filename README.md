# MatrixFactorization

#### Methods Implemented

1. [Explicit Feedback Matrix Factorization](models/als.py)
2. [Explicit Feedback MF with Bias Vectors](models/als_explicit_bias.py)
3. [Implicit Feedback MF with Bias Vectors](models/als_implicit_bias_vector.py)
4. [Implicit Feedback MF with Confidence Bias](models/als_implicit_confidence_bias.py)

#### Resource - [Link](https://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/)
#### Dataset - [Movielens 1M](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset/data)

Evaluation Metric Used - 

1. Hit rate @ k - Used on Implicit MF on Training set
2. Custom Evaluation using mean bucketized ranks - Used on Implicit MF on Test Set
3. RMSE - Used on Training set of Explicit MF

To run the code :
```
python3 main.py --configs/<your_choice.yaml>
```

## Results

- The values are based on a custom metric. Values closer to 0 represents a perfect alighnment while closer to 4 represents worst alignment. You can see the implementation in [eval.py](models/eval.py)

Implicit MF with Bias vector - 

n-factors = 25

| λ \ α  | 0.0  | 0.01 | 0.05  | 0.1  | 1  |
|--------|------|------|------|------|------|
| 0.1    |0.914 | 0.901| 0.893| 0.926| 0.975|
| 0.5    |0.904 | 0.905| 0.903| 0.914| 0.959|
| 1.0    |0.890 | 0.896| 0.894| 0.909| 0.941|


n-factors = 50

| λ \ α  | 0.0  | 0.01 | 0.05  | 0.1  | 1  |
|--------|------|------|------|------|------|
| 0.1    |0.914 | 0.908| 0.917| 0.918| 0.966|
| 0.5    |0.904 | 0.880| 0.903| 0.914| 0.959|
| 1.0/2.0    |0.899 | 0.901| 0.898| 0.899| 0.918|


<!-- 
n-factors = 100

| λ \ α  | 0.0  | 0.01 | 0.05  | 0.1  | 1  |
|--------|------|------|------|------|------|
| 0.1    |0.914 | 0.901| 0.893| 0.926| 0.975|
| 0.5    |0.904 | 0.905| 0.903| 0.914| 0.959|
| 1.0    |0.926 | |      |      |      | -->


Implicit MF with Bias in Confidence - 

n-factor = 25

| λ \ α  | 0.0  | 0.01 | 0.05  | 0.1  | 1  |
|--------|------|------|------|------|------|
| 0.1    |0.900| 0.896| 0.894| 0.892| 0.895|
| 0.5    |0.888 | 0.892| 0.891| 0.891| 0.888|
| 1.0/2.0    |0.905 | 0.898 | 0.879| 0.889| 0.896|


n-factor = 50

| λ \ α  | 0.0  | 0.01 | 0.05  | 0.1  | 1  |
|--------|------|------|------|------|------|
| 0.1    |1.03| 0.982| 0.970| 0.893| 0.101|
| 0.5    |0.897| 0.899| 0.878| 0.881| 0.898|
| 1.0/2.0    |0.897 | 0.906| 0.910| 0.895| 0.912|

<!-- works good with even very high alpha -->

Note : The results present are not optimized and the hyperparameters chosen is just educated guess. For full evaluation a Grid Search is needed.

