# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import pandas as pd
import torch
from classifier import Classifier
from PIL import Image
from torchvision import transforms


@click.command()
@click.argument('trained_model_filepath', type=click.Path(),
                default='models/trained_model.pth')
@click.argument('raw_images_folder', type=click.Path(),
                default='data/raw/to_predict')
@click.argument('predictions_filepath', type=click.Path(),
                default='data/processed/predictions.csv')


def main(trained_model_filepath, raw_images_folder, predictions_filepath):
    """ Predicts raw images located in the RAW_IMAGES_FOLDER using the pre-trained
        model in TRAINED_MODEL_FILEPATH """
    logger = logging.getLogger(__name__)
    logger.info('Creating predictions using a pre-trained neural network')

    # Load the trained model
    project_dir = Path(__file__).resolve().parents[2]
    model = Classifier()
    state_dict = torch.load(project_dir.joinpath(trained_model_filepath))
    model.load_state_dict(state_dict)
    model.eval()                     # Sets the model to evaluation mode

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    # Run through each image in the RAW_IMAGES_FOLDER
    # and predict with the network
    with torch.no_grad():  # Turn off gradients for prediction
        temp = []
        for path, subdirs, files in \
                os.walk(project_dir.joinpath(raw_images_folder)):
            for name in files:
                # Open and transform the image
                raw_image = Image.open(os.path.join(path, name))
                trans_image = transform(raw_image)
                pred_loader = torch.utils.data.DataLoader(trans_image,
                                                          batch_size=1,
                                                          shuffle=False)

                # Forward pass
                data_point = next(iter(pred_loader))
                log_ps = model(data_point)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)

                # Keep track of the file name, class
                # probabilities and predicted class
                temp.append(
                    {
                        'File_Name': name,
                        'Predicted_Class': top_class.item(),
                        'Probability_0': ps[0, 0].item(),
                        'Probability_1': ps[0, 1].item(),
                        'Probability_2': ps[0, 2].item(),
                        'Probability_3': ps[0, 3].item(),
                        'Probability_4': ps[0, 4].item(),
                        'Probability_5': ps[0, 5].item(),
                        'Probability_6': ps[0, 6].item(),
                        'Probability_7': ps[0, 7].item(),
                        'Probability_8': ps[0, 8].item(),
                        'Probability_9': ps[0, 9].item()
                    }
                )
        # Store the file name, class probabilities and predicted class
        df = pd.DataFrame(temp)
        df.to_csv(project_dir.joinpath(predictions_filepath),
                  index=False, encoding='utf-8')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

