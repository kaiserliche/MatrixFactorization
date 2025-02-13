import numpy as np


### scratch implementation of 
def evaluate_bucket_distance(self):
    user_distances = []
    predictions = self.predict()
    unique_users = np.unique(self.testdata['userid'].values)
    for user in unique_users:
        train_items = np.nonzero(self.U[user])[0]
        test_items = self.testdata.loc[self.testdata["userid"] == user, 'movieid'].values
        if test_items.size == 0 or train_items.size == 0:
                continue
        
        all_items = np.c_[[train_items, test_items]]
        all_ratings = np.c_[[self.U[user, train_items], self.testdata.loc[self.testdata['userid'] == user, "rating"].values]]

        item_rating = {}
        for item, rating in zip(all_items, all_ratings):
             item_rating[item] = rating
        
        sorted_actual_items = sorted(all_items, key=lambda item: item_rating[item], reverse=True)        
        item_preds = predictions[user, all_items]
        indices_preds = np.argsort(-item_preds)
        sorted_items_predicted = []
        for i in indices_preds:
             sorted_items_predicted.append(all_items[i])
        
        np.array(sorted_items_predicted)

        for test in test_items:
             actual_rating = item_rating[test]
             index_in_pred = np.where(sorted_items_predicted == test)[0][0]
             actual_item = sorted_actual_items[index_in_pred]
             predicted_rating = item_rating[actual_item]
             distance = abs(actual_rating - predicted_rating)
             distance.append(distance)
        if distance:
             user_distances.append(np.mean(distance))
        
        return np.mean(user_distances) if user_distances else None