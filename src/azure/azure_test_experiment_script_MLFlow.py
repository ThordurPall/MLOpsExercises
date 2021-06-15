# -*- coding: utf-8 -*-
import mlflow


def main():
    # Start the MLflow experiment
    with mlflow.start_run():
        print('Testing Azure ML using a separate experiment Python script and MLFlow')

        # Simply test logging 3
        mlflow.log_metric('TEST', 3)
        print('Testing logging 3 in the experiment run')
        print('Completed running the expriment')

if __name__ == '__main__':
    main()
