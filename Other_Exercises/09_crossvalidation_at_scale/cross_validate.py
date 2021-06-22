import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid, train_test_split

np.random.seed(123)
OPTUNA = True

data = datasets.load_digits()
X, y = data["data"], data["target"]
N = X.shape[0]
print(f"Datamatrix size {X.shape}. Label size {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")

if not OPTUNA:
    kf = KFold(n_splits=5)

    N_ESTIMATORS = [1, 5, 10, 20, 50, 100, 200]
    MAX_DEPTH = [1, 5, 10, 20, 100]

    kf = KFold(n_splits=5)

    # we are going to do a full grid search
    params = ParameterGrid({"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH})
    print(params)

    scores = []
    for p in params:
        c = RandomForestClassifier(**p)
        scores.append([])
        for train_index, val_index in kf.split(X_train):
            x_t, x_v = X_train[train_index], X_train[val_index]
            y_t, y_v = y_train[train_index], y_train[val_index]

            c.fit(x_t, y_t)

            preds = c.predict(x_v)

            acc = accuracy_score(y_v, preds)

            scores[-1].append(acc)

    scores_mean = [np.mean(s) for s in scores]
    scores_std = [np.std(s) for s in scores]

    idx = np.argmax(scores_mean)
    print(f"Best parameter combination: {params[idx]}")
    classifier = RandomForestClassifier(**params[idx])
    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_test)
    final_acc = accuracy_score(y_test, preds)
    print(f"Final score to report: {final_acc}")

##################### Here starts the exercise #####################
else:
    import optuna

    def objective(trial):
        """Define thes objective function to be minimized"""
        # fill in this. Given a trial it should
        # 1. suggest a set of hyperparameters (HINT: use trial.suggest_discrete_uniform)
        n_estimators = trial.suggest_discrete_uniform("n_estimators", 1, 200, 1)
        max_depth = trial.suggest_discrete_uniform("max_depth", 1, 100, 1)
        # n_estimators = trial.suggest_int("n_estimators", 1, 200)
        # max_depth = trial.suggest_int("max_depth", 1, 100)

        # 2. train a random forest using the hyperparameters
        c = RandomForestClassifier(
            n_estimators=int(n_estimators), max_depth=int(max_depth)
        )
        accuracies = []
        kf = KFold(n_splits=5)
        for train_index, val_index in kf.split(X_train):
            x_t, x_v = X_train[train_index], X_train[val_index]
            y_t, y_v = y_train[train_index], y_train[val_index]

            # Train the random forest
            c.fit(x_t, y_t)

            # 3. evaluate the trained random forest
            # Predict on the validation set and compute accuracy
            preds = c.predict(x_v)
            acc = accuracy_score(y_v, preds)
            accuracies.append(acc)

        # Compute the mean accuarcy over the k folds
        scores_mean = np.mean(accuracies)

        # Set an objective function to minimize
        val_acc = scores_mean
        return val_acc

    grid_sampler = False
    comparison = True
    if grid_sampler:
        # 5.2. Optuna will by default do baysian optimization when sampling the hyperparameters.
        # However, since this example is quite simple, perform a full grid search
        search_space = {
            "n_estimators": list(range(1, 200, 10)),
            "max_depth": list(range(1, 100, 10)),
        }
        print(search_space)
        study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(search_space), direction="maximize",
        )
        study.optimize(objective, n_trials=100)

    elif comparison:
        # 5.3. Compare the performance of a single optuna run using baysian optimization with
        #  n_trials=10 with a exhaustive grid search that have search through all hyperparameters
        n_trials = 10
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        search_space = {
            "n_estimators": list(range(1, 200, 1)),
            "max_depth": list(range(1, 100, 1)),
        }
        print(search_space)
        study = optuna.create_study(
            sampler=optuna.samplers.GridSampler(search_space), direction="maximize"
        )
        study.optimize(objective, n_trials=n_trials)

    else:
        # call the optimizer
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
