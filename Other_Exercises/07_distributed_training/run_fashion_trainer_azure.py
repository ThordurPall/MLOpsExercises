# -*- coding: utf-8 -*-
import os

from azureml.core import (Environment, Experiment, Model, ScriptRunConfig,
                          Workspace, ComputeTarget)
from azureml.core.conda_dependencies import CondaDependencies


def main():
    # Create a Python environment for the experiment
    env = Environment("experiment-fashion-trainer")

    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print('Ready to use Azure ML to work with {}'.format(ws.name))

    # Set the compute target
    compute_target = ComputeTarget(ws, 'MLOpsGPU')
    print('Ready to use compute target: {}'.format(compute_target.name))

    # Display compute resources in workspace
    print("Compute resources in the workspace:")
    for compute_name in ws.compute_targets:
        compute = ws.compute_targets[compute_name]
        print("\t", compute.name, ':', compute.type)

    # Ensure the required packages are installed
    packages = CondaDependencies.create(conda_packages=['pip'],
                                        pip_packages=['azureml-defaults',
                                                      'torch', 'torchvision'])
    env.python.conda_dependencies = packages

    # Create a script config for making the data set
    script_config = ScriptRunConfig(source_directory='.',
                                    script='fashion_trainer.py',
                                    environment=env,
                                    compute_target=compute_target)

    # Create and submit the experiment
    experiment = Experiment(workspace=ws, name='fashion-trainer')
    run = experiment.submit(config=script_config)

    # Block until the experiment run has completed
    run.wait_for_completion()
    print('Finished running the fashion trainer script')


if __name__ == '__main__':
    main()
