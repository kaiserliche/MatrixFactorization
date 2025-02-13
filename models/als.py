import numpy as np
from tqdm import tqdm
import os

class Model:
    def __init__(self, config):
        self.config = config
        self.savePath = config['save_path']
        self.factor = config['n_factors']
        self.n_iter = config['n_iter']
        self.lamda = config['lamda']
        self.tol = config.get('tol', 1e-8)

    def train(self,
                user_rating_matrix, 
                user_interaction_matrix, 
                user_genre_matrix,
                testdf
                ):
        # import ipdb; ipdb.set_trace()
        self.U = user_rating_matrix
        self.n_users, self.n_items = self.U.shape
        self.X = np.random.random((self.n_users, self.factor))
        self.Y = np.random.random((self.n_items, self.factor))
        
        print("Starting to train:")
        prev_rmse = float('inf')
        for i in tqdm(range(self.n_iter)):
            self.step()
            current_rmse = self.evaluate(self.U)
            print(f" RMSE on training set at {i}th iter ----> ", current_rmse)
            if abs(prev_rmse - current_rmse) < self.tol:
                break
            prev_rmse = current_rmse

        return self.X, self.Y

    def evaluate(self, U):
        mask = U > 0
        prediction = self.predict()
        # prediction = np.round(np.clip(prediction, 0, 5))
        error = U[mask] - prediction[mask]
        return np.sqrt(np.mean(error**2))

    def predict(self):
        return self.X @ self.Y.T #R^

    def step(self):

        XTX = self.X.T @ self.X + self.lamda * np.eye(self.factor)
        for i in (range(self.n_items)):
            self.Y[i] = np.linalg.solve(XTX, self.X.T @ self.U[:,i])

        YTY = self.Y.T @ self.Y + self.lamda * np.eye(self.factor)
        for u in (range(self.n_users)):
            self.X[u] = np.linalg.solve(YTY, self.Y.T@self.U[u])


    def save(self):
        os.makedirs(self.config['save_path'], exist_ok=True)
        np.save(f"{self.config['save_path']}/X.npy", self.X)
        np.save(f"{self.config['save_path']}/Y.npy", self.Y)

    def load(self):
        self.X = np.load(f"{self.config['save_path']}/X.npy")
        self.Y = np.load(f"{self.config['save_path']}/Y.npy")