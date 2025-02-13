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

To run the experiment:

```
python3 main.py --config configs/<your_choice>.yaml
```

