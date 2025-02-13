import numpy as np
from tqdm import tqdm

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
        self.U = user_rating_matrix
        self.n_users, self.n_items = self.U.shape
        self.X = np.random.random((self.n_users, self.factor))
        self.Y = np.random.random((self.n_items, self.factor))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        print("Starting to train:")
        prev_rmse = float('inf')
        for i in tqdm(range(self.n_iter)):
            self.step()
            current_rmse = self.evaluate(self.U)
            print(f" RMSE on training set at {i}th iter ----> ", current_rmse)
            if abs(prev_rmse - current_rmse) < self.tol:
                break
            prev_rmse = current_rmse

        return self.X, self.Y, self.user_bias, self.item_bias

    def evaluate(self, U):
        mask = U > 0
        prediction = self.predict()
        error = U[mask] - prediction[mask]
        return np.sqrt(np.mean(error**2))

    def predict(self):
        return self.user_bias[:, None] + self.item_bias[None, :] + self.X @ self.Y.T

    def step(self):
        for i in (range(self.n_items)):
            mask = self.U[:, i] > 0### masking to ensure correct updates in user and item matrix. without this, bias vectors will be "biased" to users who have not interacted
            X_tilde = np.c_[np.ones(np.sum(mask)), self.X[mask]] 
            # import ipdb; ipdb.set_trace()
            r_beta_i = self.U[mask, i] - self.user_bias[mask]
            reg_matrix = self.lamda * np.eye(self.factor + 1)
            theta = np.linalg.solve(X_tilde.T @ X_tilde + reg_matrix, X_tilde.T @ r_beta_i)
            self.item_bias[i], self.Y[i] = theta[0], theta[1:]
        
        for u in (range(self.n_users)):
            mask = self.U[u, :] > 0
            Y_tilde = np.c_[np.ones(np.sum(mask)), self.Y[mask]]
            r_gamma_u = self.U[u, mask] - self.item_bias[mask]
            reg_matrix = self.lamda * np.eye(self.factor + 1)
            theta = np.linalg.solve(Y_tilde.T @ Y_tilde + reg_matrix, Y_tilde.T @ r_gamma_u)
            self.user_bias[u], self.X[u] = theta[0], theta[1:]

    def save(self):
        np.save(f"{self.config['save_path']}/X.npy", self.X)
        np.save(f"{self.config['save_path']}/Y.npy", self.Y)
        np.save(f"{self.config['save_path']}/user_bias.npy", self.user_bias)
        np.save(f"{self.config['save_path']}/item_bias.npy", self.item_bias)

    def load(self):
        self.X = np.load(f"{self.config['save_path']}/X.npy")
        self.Y = np.load(f"{self.config['save_path']}/Y.npy")
        self.user_bias = np.load(f"{self.config['save_path']}/user_bias.npy")
        self.item_bias = np.load(f"{self.config['save_path']}/item_bias.npy")
