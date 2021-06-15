# -*- coding: utf-8 -*-
from azureml.core import Experiment, Workspace


def main():
    # Get the workspace and experiment
    ws = Workspace.from_config()
    experiment = Experiment.list(workspace=ws,
                                 experiment_name='Test-Experiments')
    print(experiment)
    experiment = experiment[0]

    # Cancel all runs which are in running state
    for run in experiment.get_runs():
        if run.status == "Running":
            print('Cancel a running expriment with run Id: {}'.format(run.id))
            run.cancel()

    # Check the status of each run again
    for run in experiment.get_runs():
        print(run.id)
        print(run.status)

if __name__ == '__main__':
    main()
