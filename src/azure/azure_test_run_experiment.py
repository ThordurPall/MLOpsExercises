# -*- coding: utf-8 -*-
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.conda_dependencies import CondaDependencies


def main():
    # Create a Python environment for the experiment
    # env = Environment("experiment_test_env")
    env = Environment("experiment-test-MLFlow-env")

    # Ensure the required packages are installed
    #  (here pip and Azure ML defaults)
    packages = CondaDependencies.create(conda_packages=['pip'],
                                        pip_packages=['mlflow', 'azureml-mlflow'])
                                        # pip_packages=['azureml-defaults'])
    env.python.conda_dependencies = packages

    # Create a script config
    experiment_folder = './src/azure'
    script_config = ScriptRunConfig(source_directory=experiment_folder,
                                    script='azure_test_experiment_script_MLFlow.py',
                                    # script='azure_test_experiment_script.py',
                                    environment=env)
    
    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print('Ready to use Azure ML to work with {}'.format(ws.name))

    # Create and submit the experiment
    experiment = Experiment(workspace=ws, name='test-experiment-MLFlow') # name='test-experiment')
    run = experiment.submit(config=script_config)
    run.wait_for_completion()


if __name__ == '__main__':
    main()
