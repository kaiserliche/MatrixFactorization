import pandas as pd
from argparse import ArgumentParser
import yaml
from dataProcessor import DataLoader
from models import als, als_explicit_bias, als_implicit_bias_vector, als_implicit_confidence_bias

MODELS = {
    'als': als.Model, 
    'als_with_bias_explicit': als_explicit_bias.Model, #explicit MF with bias vector
    'als_with_bias_implicit' : als_implicit_bias_vector.Model, #implicit MF with bias vector
    'als_with_confidence_matrix_implicit': als_implicit_confidence_bias.Model ### implicit MF with bias handled in confidence equation
}

def main(config):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataProcessor = DataLoader(config['data'])
    dataProcessor.load_data()
    dataProcessor.preprocess_data()
    print('Data processing completed')

    model = MODELS[config['model']['name']](config['model'])
    model.train(
                user_rating_matrix = dataProcessor.U, 
                user_interaction_matrix = dataProcessor.C, 
                user_genre_matrix = dataProcessor.G,
                testdf = dataProcessor.testdata
                )
    model.save()
    print('Model evaluation completed')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = args.config
    main(config)
    