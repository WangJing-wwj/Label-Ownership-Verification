"""
Trains a downstream (user) model on a dataset with input labels.
"""

from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, get_n_classes, pick_poisoner,\
                                        construct_user_dataset
from modules.base_utils.util import extract_toml, get_train_info, mini_train, load_model,\
                                    needs_big_ims, slurmify_path, softmax,clf_eval
from modules.base_utils.SCALE_UP import SCALE_UP

def run(experiment_name, module_name,config_file, **kwargs):
    """
    Runs user model training and saves metrics.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """
    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name,config_file)

    user_model_flag = args["user_model"]
    trainer_flag = args["trainer"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    soft = args.get("soft", True)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    alpha = args.get("alpha", None)
    train_pct=args.get("train_pct",1.0)
    sample_number=args.get("sample_number",50)
    scale_start=args.get("scale_start",1.0)
    scale_stop=args.get("scale_stop",4.001)
    scale_step=args.get("scale_step",0.0025)
    input_path = slurmify_path(args["input_labels"], slurm_id)
    true_path = slurmify_path(args.get("true_labels", None), slurm_id)
    output_path = slurmify_path(args["output_dir"], slurm_id)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("Building datasets...")
    poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label)

    big_ims = needs_big_ims(user_model_flag)
    _, distillation, test, poison_test, _ =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, big=big_ims)

    labels_syn = torch.tensor(np.load(input_path))        

    if alpha > 0:
        assert true_path is not None
        y_true = torch.tensor(np.load(true_path))
        labels_d = softmax(alpha * y_true + (1 - alpha) * labels_syn)
    else:
        labels_d = softmax(labels_syn)

    if not soft:
        labels_d = labels_d.argmax(dim=1)

    user_dataset = construct_user_dataset(distillation, labels_d)

    # Train user model
    print("Training user model...")
    n_classes = get_n_classes(dataset_flag)
    model_retrain = load_model(user_model_flag, n_classes)

    batch_size, epochs, optimizer_retrain, scheduler = get_train_info(
        model_retrain.parameters(), trainer_flag, batch_size,
        epochs, optim_kwargs, scheduler_kwargs
    )
    model_retrain, clean_metrics, poison_metrics = mini_train(
        model=model_retrain,
        train_data=user_dataset,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=optimizer_retrain,
        scheduler=scheduler,
        epochs=epochs,
        expert_id=None,
        record=True
    )

    # model_retrain.load_state_dict(torch.load("/home/ZhaoZQ/repos/MyCodes/FLIP/out/checkpoints/cifar100_r18_1xs_acc_0.8054_asr0.98.pth"))
    
    # Verification Preparation
    dataset =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims,spc=True)
    
    defense=SCALE_UP(model_retrain)


    # Soft Label
    print("Probability Label Verification...")
    delta_P,t_stat,p_value=defense.probability_available_test(target_label,dataset,sample_number)
    print(f"poisoner_falg is {poisoner_flag}\n ","t_stat,p_value,avg_delta_probability are ",t_stat,p_value,delta_P)

    # Save results
    print("Saving results...")
    np.save(output_path + f"{dataset_flag}_{user_model_flag}_{poisoner_flag}_asr.npy", poison_metrics)
    np.save(output_path + f"{dataset_flag}_{user_model_flag}_{poisoner_flag}_acc.npy", clean_metrics)
    np.save(output_path + f"{dataset_flag}_{user_model_flag}_{poisoner_flag}_labels.npy", labels_d.numpy())
    clean_test_acc = clf_eval(model_retrain, test)[0]
    poison_test_acc = clf_eval(model_retrain, poison_test.poison_dataset)[0]
    torch.save(model_retrain.state_dict(), output_path + f"{dataset_flag}_{user_model_flag}_{poisoner_flag}_acc_{clean_test_acc}_asr_{poison_test_acc}.pth")

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
