# -*- coding: utf-8 -*-
import logging
import sys

import click
import optuna
import plotly.io as pio

from src.models.train_model import optuna_objective, train_model


@click.command()
@click.argument("data_filepath", type=click.Path(), default="data")
@click.argument(
    "trained_model_filepath", type=click.Path(), default="models/trained_model.pth"
)
@click.argument(
    "training_statistics_filepath", type=click.Path(), default="data/processed/"
)
@click.argument(
    "training_figures_filepath", type=click.Path(), default="reports/figures/"
)
@click.argument("epoch", type=int, default=10)
@click.argument("lr", type=float, default=0.001)
@click.option(
    "-uo",
    "--use_optuna",
    type=bool,
    default=False,
    help="Set True to run Optuna cross validation (default=False)",
)
def train_model_command_line(
    data_filepath,
    trained_model_filepath,
    training_statistics_filepath,
    training_figures_filepath,
    epoch,
    lr,
    use_optuna,
):
    """ Trains the neural network using MNIST training data """
    if use_optuna:
        # Pruners automatically stop unpromising trials at the early
        # stages of the training (a.k.a., automated early-stopping)
        print("Using median pruning")

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )

        # Set up the median stopping rule as the pruning condition
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
            direction="maximize",
        )
        study.optimize(
            lambda trial: optuna_objective(
                trial,
                data_filepath=data_filepath,
                trained_model_filepath=trained_model_filepath,
                training_statistics_filepath=training_statistics_filepath,
                training_figures_filepath=training_figures_filepath,
                epochs=epoch,
            ),
            n_trials=3,
        )

        # Plot optimization history of all trials in a study
        fig = optuna.visualization.plot_optimization_history(study)
        pio.write_image(fig, "optuna_optimization_history.pdf")

        # Plot intermediate values of all trials in a study -
        # Visualize the learning curves of the trials
        fig = optuna.visualization.plot_intermediate_values(study)
        pio.write_image(fig, "optuna_accuracy_curve.pdf")

        # Plot the high-dimensional parameter relationships in a study
        fig = optuna.visualization.plot_parallel_coordinate(study)
        pio.write_image(fig, "optuna_high_dim_par_relationships.pdf")

        # Plot the parameter relationship as contour plot in a study
        fig = optuna.visualization.plot_contour(study)
        pio.write_image(fig, "optuna_contour_par_relationships.pdf")

        # Visualize individual hyperparameters as slice plot
        fig = optuna.visualization.plot_slice(study)
        pio.write_image(fig, "optuna_individual_pars.pdf")

        # Plot hyperparameter importances
        fig = optuna.visualization.plot_param_importances(study)
        pio.write_image(fig, "optuna_individual_par_importance.pdf")

        # Plot the objective value EDF (empirical distribution function) of a study
        fig = optuna.visualization.plot_edf(study)
        pio.write_image(fig, "optuna_edf.pdf")
        fig.show()

    else:
        _ = train_model(
            data_filepath,
            trained_model_filepath,
            training_statistics_filepath,
            training_figures_filepath,
            epoch,
            lr,
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_model_command_line()
