from typing import Dict, Any
from pathlib import Path
import numpy as np
import torch

from src.models.PKDR.Trainer import RKHS_Trainer
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate


def fit_model(train_dataset, hyperparams):
    """
    Fit the RKHS model using cross-validation and evaluate it.

    Args:
        - cfg (object): Configuration object containing dataset and model details.
        - treatment (np.array): Array of treatment effects.
        - train_dataset (tuple): Training dataset.
        - test_dataset (tuple): Testing dataset.
        - hyperparams (dict): Dictionary of hyperparameters for model training.

    Returns:
        - ATE_h (float): Estimated Average Treatment Effect using h-test.
        - ATE_q (float): Estimated Average Treatment Effect using q-test.
        - ATE_dr (float): Estimated Average Treatment Effect using doubly robust test.
    """
    rkhs_train = RKHS_Trainer(train_dataset, **hyperparams)
    rkhs_train.fit_h_cv()

    # rkhs_train.fit_q_cv(type='kde')
    # elif cfg.dataset == 'high':
    rkhs_train.fit_q_cv(type='cnf', hyperparams=hyperparams)

    return rkhs_train


def pkdr_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                     one_mdl_dump_dir: Path, random_seed: int = 42, verbose: int = 0):
    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")
    print(data_config)
    print(model_param)
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    torch.manual_seed(random_seed)

    rkhs_train = fit_model(train_data, model_param)



    ATE_dr = rkhs_train._drtest(test_data.treatment, train_data)
    ATE_dr =np.array(ATE_dr).reshape(-1,1)

    ATE_q = rkhs_train._qtest(test_data.treatment, train_data)
    ATE_q = np.array(ATE_q).reshape(-1,1)


    ATE_dr = preprocessor.postprocess_for_prediction(ATE_dr)
    ATE_q = preprocessor.postprocess_for_prediction(ATE_q)

    oos_loss = 0.0
    if test_data.structural is not None:
        oos_loss: float = np.mean((ATE_dr - test_data_org.structural) ** 2)
        q_loss: float = np.mean((ATE_q - test_data_org.structural) ** 2)
        print('standard loss:', oos_loss)
        print('q loss:', q_loss)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(ATE_dr - test_data_org.structural))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), ATE_dr)
    return oos_loss

