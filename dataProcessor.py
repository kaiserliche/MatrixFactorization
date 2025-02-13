import pandas as pd
import numpy as np
import kagglehub

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        path = kagglehub.dataset_download(self.config['rawPath'])
        rcolumns = self.config['rcols']
        mcolumns = self.config['mcols']
        if self.config['use_cache']:
            self.cache_load()
            return
        self.ratings = pd.read_table(f"{path}/{self.config['ratingFile']}", names = rcolumns, sep = self.config['sep'], encoding = "latin1", engine='python')
        self.movies = pd.read_table(f"{path}/{self.config['movieFile']}", names = mcolumns, sep = self.config['sep'], encoding = "latin1", engine='python')
        print('Data loaded')

    def preprocess_data(self):
        # import ipdb; ipdb.set_trace()
        if self.config["use_cache"]:
            return
        ratings = self.ratings
        movies = self.movies
        ratings = ratings[ratings['rating'] > 0]
        mergedData = ratings.merge(movies, on="movieid")
        # import ipdb; ipdb.set_trace()
        unique_movie_ids = sorted(mergedData["movieid"].unique())
        movie_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}
        mergedData['movieid'] = mergedData['movieid'].map(movie_id_mapping)
        mergedData['userid'] = mergedData['userid'] - 1
        mergedData.drop(["timestamp", "movie"], axis = 1,inplace= True)
        genres_ = mergedData['genres'].values
        genres = set()
        for gen in genres_.tolist():
            gen_ = gen.split("|")
            for gen__ in gen_:
                genres.add(gen__)
        genlookup = {}
        for i, genre in enumerate(sorted(list(genres))):
            genlookup[genre] = i
        
        def convert(genrestr):
            gens = genrestr.split("|")
            arr = []
            for g in gens:
                arr.append(genlookup[g])
            return arr
        mergedData["genres"] = mergedData["genres"].apply(convert)
        movie_counts = mergedData["movieid"].value_counts()
        movies_with_multiple_ratings = movie_counts[movie_counts > 5].index  # Movies rated by multiple users , hardcoded 5
        df_filtered = mergedData[mergedData["movieid"].isin(movies_with_multiple_ratings)]

        np.random.seed(self.config['seed'])
        test_indices = df_filtered.sample(frac=1 - self.config["train_size"], random_state=self.config["seed"]).index
        test_df= mergedData.loc[test_indices].copy()
        train_df = mergedData.drop(test_indices).copy()
        train_user_movie_matrix = train_df.pivot(index='userid', columns='movieid', values='rating').fillna(0)
        
        mergedData_exploded = train_df.explode("genres")
        user_genre_matrix = mergedData_exploded.groupby(["userid", "genres"]).size().unstack(fill_value=0)
        ### creating user interaction matrix based on genre
        def get_interaction(userid, genres):
            return sum(train_user_movie_matrix.loc[userid, gen] for gen in genres if gen in user_genre_matrix.columns)
        train_df["interaction"] = train_df.apply(lambda row: get_interaction(row["userid"], row["genres"]), axis=1)
        user_interaction_matrix = train_df.pivot(index='userid', columns='movieid', values='interaction').fillna(0)

        self.U = train_user_movie_matrix.values
        self.G = user_genre_matrix.values
        self.C = user_interaction_matrix.values
        self.testdata = test_df
        # self.split_data()
        self.save_data()


    def split_data(self):
        #### split movie data into train and test
        self.train = self.data.sample(frac=self.config['train_size'],
                                       random_state=self.config['seed'])
        self.test = self.data.drop(self.train.index)

        print('Data split completed')
        self.U_train = self.train.pivot(index='user_id', columns='item_id', values='rating').fillna(0).values
        self.U_test = self.test.pivot(index='user_id', columns='item_id', values='rating').fillna(0).values
        print("User-Item matrix created")
        self.save_data()

        
    def save_data(self):
        np.save(self.config["URM"], self.U)
        np.save(self.config["UIM"], self.C)
        np.save(self.config["UGM"], self.G)
        self.testdata.to_csv(self.config["testdf"])

    def cache_load(self):
        self.U = np.load(self.config["URM"])
        self.C = np.load(self.config["UIM"])
        self.G = np.load(self.config["UGM"])
        self.testdata = pd.read_csv(self.config["testdf"])
        print("Used cached data, loading complete !")