# -*- coding: utf-8 -*-
import azureml.core
from azureml.core import Experiment, Workspace


def main():
    print('Testing Azure ML with a standalone Python script')
   
    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

    # Display compute resources in workspace
    print("Compute resources in the workspace:")
    for compute_name in ws.compute_targets:
        compute = ws.compute_targets[compute_name]
        print("\t", compute.name, ':', compute.type)

    # Create an Azure ML experiment in the workspace
    experiment = Experiment(workspace=ws,
                            name="Test-Experiment-Standalone-Scripts")

    # Start logging data from the experiment, obtaining
    # a reference to the experiment run
    run = experiment.start_logging()
    print("Starting experiment:", experiment.name)

    # Just test logging 1
    run.log('TEST', 1)
    print('Testing logging 1 in the experiment run')

    # Complete the run
    run.complete()
    print('Completed running the expriment')

if __name__ == '__main__':
    main()
