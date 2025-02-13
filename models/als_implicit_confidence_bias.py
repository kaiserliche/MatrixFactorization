import numpy as np
from tqdm import tqdm
from models.abstract import AbstractModel
import os

class Model:
    def __init__(self, config):
        self.config = config
        self.savePath = config['save_path']
        self.factor = config['n_factors']
        self.n_iter = config['n_iter']
        self.lamda = config['lamda']
        self.alpha = config.get('alpha', 40)  # Confidence scaling factor alpha
        self.tol = config.get('tol', 1e-8)
        self.k = config.get('k', [1,5,10,20])

    def train(self, user_rating_matrix, user_interaction_matrix, user_genre_matrix, testdf):
        self.U = user_rating_matrix
        self.C = user_interaction_matrix
        self.G = user_genre_matrix
        self.testdata = testdf
        # self.preprocess()
        self.prereq()
        self.n_users, self.n_items = self.U.shape

        self.X = np.random.random((self.n_users, self.factor))
        self.Y = np.random.random((self.n_items, self.factor))
        self.P = (self.U > 0).astype(float)  # Binary presence matrix

        print("Starting training...")
        print("Note : Hit rates are on train data and other metrics are on Test data")
        for i in tqdm(range(self.n_iter)):
            self.step()
            metrics = self.evaluate(self.U, self.k)  
            print(f"Hit Rate@1 = {metrics['Hit Rate@1']:.4f}, "
                f"Hit Rate@5 = {metrics['Hit Rate@5']:.4f}, "
                f"Hit Rate@10 = {metrics['Hit Rate@10']:.4f}, "
                f"Hit Rate@20 = {metrics['Hit Rate@20']:.4f}")
            
            metric = self.evaluate_preference_by_rating()
            print(f"Mean preference distance from actual prediction for {i}th iter on test data ----> {metric}")

            metric2 = self.evaluate_bucket_distance()
            print(f"Mean bucket distance from actual prediction for {i} th iter on test data----> {metric2}")

        return self.X, self.Y


    def prereq(self):
        self.popularity = np.sum(self.U > 0, axis=0)  # Movie popularity (# of ratings)
        user_rating_sum = np.sum(self.U, axis=1)  # Sum of ratings per user
        user_rating_count = np.sum(self.U > 0, axis=1)  # Count of ratings per user

        user_avg_rating = np.where(user_rating_count > 0, user_rating_sum / user_rating_count, 0)  # User's avg rating
        movie_avg_rating = np.where(self.popularity > 0, np.sum(self.U, axis=0) / self.popularity, 0)  # Movie avg rating

        print('Generating confidence matrix')
        # self.C = np.load("C.npy")
        # return
        for userid in tqdm(range(self.U.shape[0])):
            for movieid in range(self.U.shape[1]):
                rating = self.U[userid][movieid]
                if rating == 0:
                    confidence = 1
                else:
                    interactions = self.C[userid][movieid]
                    user_bias_factor = user_avg_rating[userid] / np.max(user_avg_rating) if np.max(user_avg_rating) > 0 else 1
                    item_bias_factor = movie_avg_rating[movieid] / np.max(movie_avg_rating) if np.max(movie_avg_rating) > 0 else 1
                    confidence = 1 + self.alpha*(interactions)* (self.popularity[movieid]/np.sum(self.popularity) )*(user_bias_factor + item_bias_factor)/2
                self.C[userid][movieid] = confidence

    def predict(self):
        return self.X @ self.Y.T  # Removed bias terms, only MF

    def step(self):
        # Update item factors
        for i in (range(self.n_items)):
            mask = self.U[:, i] > 0
            if np.sum(mask) == 0:
                continue  # Skip items with no interactions
            C_i = np.diag(self.C[mask, i])
            X_tilde = self.X[mask]
            reg_matrix = self.lamda * np.eye(self.factor)
            theta = np.linalg.solve(X_tilde.T @ C_i @ X_tilde + reg_matrix, X_tilde.T @ C_i @ self.P[mask, i])
            self.Y[i] = theta
        
        # Update user factors
        for u in (range(self.n_users)):
            mask = self.U[u, :] > 0
            if np.sum(mask) == 0:
                continue  # Skip users with no interactions
            C_u = np.diag(self.C[u, mask])
            Y_tilde = self.Y[mask]
            reg_matrix = self.lamda * np.eye(self.factor)
            theta = np.linalg.solve(Y_tilde.T @ C_u @ Y_tilde + reg_matrix, Y_tilde.T @ C_u @ self.P[u, mask])
            self.X[u] = theta

    
    def evaluate(self, U, k_values=[1, 5, 10, 20]):
        num_users = U.shape[0]
        hit_rates = {k: [] for k in k_values}
        predictions = self.predict()
        
        for user in range(num_users):
            # Get actual items & sort by rating (highest-rated items first)
            actual_items = np.where(U[user] > 0)[0]
            if len(actual_items) == 0:
                continue
            
            actual_items = actual_items[np.argsort(-U[user, actual_items])]  # Sort by rating (descending)

            # Get top-K predictions
            predicted_ranking = np.argsort(-predictions[user])  # Sort predictions in descending order
            
            for k in k_values:
                top_k_items = predicted_ranking[:k]
                hit = np.any(np.isin(top_k_items, actual_items))  # Check if any actual item is in top-K
                hit_rates[k].append(hit)
        
        # Compute average hit rate for each K
        avg_hit_rates = {f"Hit Rate@{k}": np.mean(hit_rates[k]) for k in k_values}
        
        return avg_hit_rates

    def save(self):
        os.makedirs(self.config['save_path'], exist_ok=True)
        np.save(f"{self.config['save_path']}/X.npy", self.X)
        np.save(f"{self.config['save_path']}/Y.npy", self.Y)

    def load(self):
        self.X = np.load(f"{self.config['save_path']}/X.npy")
        self.Y = np.load(f"{self.config['save_path']}/Y.npy")
    

    def evaluate_bucket_distance(self):
        user_distances = []
        predictions = self.predict()
        unique_users = np.unique(self.testdata['userid'].values)
        for user in unique_users:
            train_items = np.nonzero(self.U[user])[0]
            test_items = self.testdata.loc[self.testdata["userid"] == user, 'movieid'].values
            if test_items.size == 0 or train_items.size == 0:
                    continue
            
            all_items = np.concatenate([train_items, test_items])
            all_ratings = np.concatenate([self.U[user, train_items], self.testdata.loc[self.testdata['userid'] == user, "rating"].values])

            item_rating = {}
            for item, rating in zip(all_items, all_ratings):
                item_rating[item] = rating
            
            sorted_actual_items = np.array(sorted(all_items, key=lambda item: item_rating[item], reverse=True))   
            item_preds = predictions[user, all_items]
            indices_preds = np.argsort(-item_preds)

            sorted_items_predicted = []
            for i in indices_preds:
                sorted_items_predicted.append(all_items[i])
            
            np.array(sorted_items_predicted)
            distances = []
            for test in test_items:
                actual_rating = item_rating[test]
                index_in_pred = np.where(sorted_items_predicted == test)[0][0]
                actual_item = sorted_actual_items[index_in_pred]
                predicted_rating = item_rating[actual_item]
                distance = abs(actual_rating - predicted_rating)
                distances.append(distance)
            if distances:
                user_distances.append(np.mean(distances))
            
        return np.mean(user_distances) if user_distances else None

    def evaluate_preference_by_rating(self):
        test_df = self.testdata
        U = self.U
        predictions = self.predict()
        preference_deviations = []
        unique_users = np.unique(test_df["userid"].values)

        for user in unique_users:
            train_items = np.nonzero(U[user])[0]
            test_items = test_df.loc[test_df["userid"] == user, "movieid"].values

            if test_items.size == 0 or train_items.size == 0:
                continue

            train_ratings = U[user, train_items]
            train_prefs = predictions[user, train_items]

            unique_ratings, rating_sums = np.unique(train_ratings, return_counts=True)
            rating_bucket_means = {r: np.mean(train_prefs[train_ratings == r]) for r in unique_ratings}

            test_prefs = predictions[user, test_items]
            test_ratings = test_df.loc[test_df["userid"] == user, "rating"].values

            valid_mask = np.isin(test_ratings, unique_ratings)
            test_bucket_means = np.array([rating_bucket_means[r] for r in test_ratings[valid_mask]])
            deviations = np.abs(test_prefs[valid_mask] - test_bucket_means)

            preference_deviations.extend(deviations)

        return np.log1p(np.mean(preference_deviations)) if preference_deviations else None
