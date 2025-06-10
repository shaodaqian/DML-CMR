# from __future__ import annotations
from typing import Dict, Any
import sys
import torch
from pathlib import Path
from torch.utils.data import random_split,ConcatDataset

from src.models.DML_PCL.model import Treatment,train_treatment,train_Ymodel,Ymodel,Response,train_response,train_response_kfold
from src.models.DML_PCL.nn.utils import ResponseDataset,device

import numpy as np
from src.data.ate import generate_train_data_ate, generate_test_data_ate, generate_val_data_ate,get_preprocessor_ate
from src.data.ate.data_class import PVTestDataSetTorch


def dml_pcl_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                     one_mdl_dump_dir: Path, random_seed: int = 42, verbose: int = 0):
    dump_dir = one_mdl_dump_dir.joinpath(f"{random_seed}")
    print(data_config)
    print(model_param)
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data_org = generate_val_data_ate(data_config=data_config, rand_seed=9999)
    test_data_org = generate_test_data_ate(data_config=data_config)

    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    val_data = preprocessor.preprocess_for_val_input(val_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    print(train_data.treatment.shape)
    print(train_data.outcome.shape)
    print(train_data.treatment_proxy.shape)
    print(train_data.outcome_proxy.shape)

    torch.manual_seed(random_seed)

    hidden = [128,256,32]
    N=train_data.outcome.shape[0]
    print('number of data: ',N)
    # epochs = int(1000000./float(N))+100 # heuristic to select number of epochs
    # epochs: 200 to 800

    # dropout_rate: 0.08 to 0.03

    # dropout_rate = min(400. / (4000. + N), 0.5)

    use_image=False
    image_shape = None
    if data_config.get("name")=="dsprite":
        use_image=True
        image_shape = (-1,1, 64, 64)


    n_components = model_param.get("n_components", 10)
    loss = 'mixture_of_gaussians'
    dropout_rate = model_param.get("dropout", 0.05)
    w_decay = model_param.get("weight_decay", 0.0001)
    response_w_decay=model_param.get("response_decay", 0.0001)
    lr = model_param.get("lr", 0.0005)
    epochs= model_param.get("n_epoch", 100)
    batch_size = model_param.get("batch_size", 128)
    verbose=1



    # treatment: np.ndarray
    # treatment_proxy: np.ndarray
    # outcome_proxy: np.ndarray
    # outcome: np.ndarray
    # backdoor: Optional[np.ndarray]

    # z: treatment_proxy
    # x: treatment
    # t: outcome proxy
    # y: outcome


    # train_data = ResponseDataset(z,x,t,y)
    # train_data = z, x, t, y
    print('treatment_proxy stats: ',train_data.treatment_proxy.mean(), train_data.treatment_proxy.std())
    print('treatment stats: ',train_data.treatment.mean(), train_data.treatment.std())
    print('outcome_proxy stats: ',train_data.outcome_proxy.mean(), train_data.outcome_proxy.std())
    print('outcome stats: ',train_data.outcome.mean(), train_data.outcome.std())

    train_dataset = ResponseDataset((train_data.treatment_proxy, train_data.treatment, train_data.outcome_proxy, train_data.outcome))
    valid_data = (val_data.treatment_proxy, val_data.treatment, val_data.outcome_proxy, val_data.outcome)

    treat_dim = train_data.treatment_proxy.shape[-1] + train_data.treatment.shape[-1]



    treatment_model = Treatment(treat_dim, hidden, dropout_rate, n_components,use_image=use_image,image_shape=image_shape)
    treatment_model = train_treatment(treatment_model, train_dataset, valid_data, batch_size, loss,
                                      epochs, lr, w_decay, verbose=verbose)
    treatment_model.eval()

    Y_model = Ymodel(treat_dim, hidden, dropout_rate,use_image=use_image,image_shape=image_shape)
    Y_model = train_Ymodel(Y_model, train_dataset, valid_data, batch_size, epochs, lr, w_decay, verbose=verbose)
    Y_model.eval()

    # x, z, t, y, g_true = datafunction(int(n), 2)
    # epochs=500
    # batch_size=100
    n_samples = 1
    resp_dim = train_data.treatment.shape[-1] + train_data.outcome_proxy.shape[-1]
    dml = False
    standard_response = Response(resp_dim, hidden, dropout_rate,use_image=use_image,image_shape=image_shape)
    standard_response = train_response(standard_response, treatment_model, Y_model, train_dataset, valid_data, dml,
                                       batch_size, epochs, lr, response_w_decay, n_samples, verbose=verbose)
    standard_response.eval()

    dml = True
    dml1_response = Response(resp_dim, hidden, dropout_rate,use_image=use_image,image_shape=image_shape)
    dml1_response = train_response(dml1_response, treatment_model, Y_model, train_dataset, valid_data, dml, batch_size,
                                   epochs, lr, response_w_decay, n_samples, verbose=verbose)
    dml1_response.eval()


    # k_fold = 10
    # print("k_fold")
    # datasets = random_split(train_dataset, [1 / k_fold] * k_fold)
    # treatments = []
    # y_models = []
    # for fold in range(k_fold):
    #     train_fold = []
    #     for d in range(k_fold):
    #         if d != fold:
    #             train_fold.append(datasets[d])
    #
    #     train_fold = ConcatDataset(train_fold)
    #     treatment_model = Treatment(treat_dim, hidden, dropout_rate, n_components, use_image=use_image, image_shape=image_shape)
    #     treatment_model = train_treatment(treatment_model, train_fold, valid_data, batch_size, loss, epochs, lr,
    #                                       w_decay, verbose=verbose)
    #     treatment_model.eval()
    #
    #     # epochs = 500
    #     Y_model = Ymodel(treat_dim, hidden, dropout_rate, use_image=use_image, image_shape=image_shape)
    #     Y_model = train_Ymodel(Y_model, train_fold, valid_data, batch_size, epochs, lr, w_decay, verbose=verbose)
    #     Y_model.eval()
    #
    #     treatments.append(treatment_model)
    #     y_models.append(Y_model)
    #     print("finished fold: ", fold)
    #     sys.stdout.flush()
    #
    # dml = True
    # dml_k_response = Response(resp_dim, hidden, dropout_rate, use_image=use_image, image_shape=image_shape)
    # dml_k_response = train_response_kfold(dml_k_response, treatments, y_models, datasets, valid_data, dml, batch_size,
    #                                       epochs, lr, response_w_decay, n_samples, verbose=verbose)
    # dml_k_response.eval()



    test_data_t = PVTestDataSetTorch.from_numpy(test_data)

    test_treat=test_data_t.treatment.to(device)

    out_proxy=torch.tensor(train_data.outcome_proxy).to(device)

    with torch.no_grad():
        standard_pred=np.array([np.mean(standard_response(test_treat[i].unsqueeze(0).repeat(out_proxy.shape[0],1),
                                                          out_proxy).cpu().numpy(),axis=0) for i in range(test_treat.shape[0])])

        dml_pred = np.array([np.mean(dml1_response(test_treat[i].unsqueeze(0).repeat(out_proxy.shape[0],1),
                                                  out_proxy).cpu().numpy(),axis=0) for i in range(test_treat.shape[0])])

        # dml_k_pred = np.array([np.mean(dml_k_response(test_treat[i].unsqueeze(0).repeat(out_proxy.shape[0],1),
        #                                           out_proxy).cpu().numpy(),axis=0) for i in range(test_treat.shape[0])])



    standard_pred = preprocessor.postprocess_for_prediction(standard_pred)
    dml_pred = preprocessor.postprocess_for_prediction(dml_pred)
    # dml_k_pred = preprocessor.postprocess_for_prediction(dml_k_pred)

    oos_loss = 0.0
    dml_loss = 0.0
    dml_k_loss = 0.0
    if test_data.structural is not None:
        oos_loss: float = np.mean((standard_pred - test_data_org.structural) ** 2)
        print('standard loss:', oos_loss)
        dml_loss: float = np.mean((dml_pred - test_data_org.structural) ** 2)
        print('dml loss:', dml_loss)
        # dml_k_loss: float = np.mean((dml_k_pred - test_data_org.structural) ** 2)
        # print('dml k loss:', dml_k_loss)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(standard_pred - test_data_org.structural))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), dml_pred)
    sys.stdout.flush()

    return [oos_loss,dml_loss,dml_k_loss]
