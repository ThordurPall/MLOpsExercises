# -*- coding: utf-8 -*-
from azureml.core import Run


def main():
    print('Testing Azure ML using a separate experiment Python script')

    # Get the experiment run context. That is, retrieve the experiment
    # run context when the script is run
    run = Run.get_context()

    # Simply test logging 2
    run.log('TEST', 2)
    print('Testing logging 2 in the experiment run')

    # Complete the run
    run.complete()
    print('Completed running the expriment')

if __name__ == '__main__':
    main()
