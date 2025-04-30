"""
Optimizes logit labels given expert trajectories using trajectory matching.
"""
from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes
from modules.base_utils.util import extract_toml, get_module_device, get_mtt_attack_info,\
                                    load_model, either_dataloader_dataset_to_both, temp_dataloader,make_pbar,\
                                    needs_big_ims, slurmify_path, clf_loss, softmax,\
                                    total_mse_distance
from modules.generate_labels.utils import coalesce_attack_config, extract_experts,\
                                          extract_labels, sgd_step

def run(experiment_name, module_name, **kwargs):
    """
    Optimizes and saves poisoned logit labels.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    input_pths = args["input_pths"]
    opt_pths = args["opt_pths"]
    expert_model_flag = args["expert_model"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    expert_config = args.get('expert_config', {})
    config = coalesce_attack_config(args.get("attack_config", {}))

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build datasets and initialize labels
    print("Building datasets...")
    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)

    big_ims = needs_big_ims(expert_model_flag)
    _, _, _, _, mtt_dataset =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims)
    n_classes= get_n_classes(dataset_flag)
    labels = extract_labels(mtt_dataset.distill, config['one_hot_temp'], n_classes)
    labels_init = torch.stack(extract_labels(mtt_dataset.distill, 1, n_classes))
    labels_syn = torch.stack(labels).requires_grad_(True)

    # Load expert trajectories
    print("Loading expert trajectories...")
    expert_starts, expert_opt_starts = extract_experts(
        expert_config,
        input_pths,
        config['iterations'],
        expert_opt_path=opt_pths
    )

    # Optimize labels
    print("Training...")
    n_classes = get_n_classes(dataset_flag)
    pre_computed_scores_exist=False

    student_model = load_model(expert_model_flag, n_classes)
    expert_model = load_model(expert_model_flag, n_classes)

    device = get_module_device(student_model)

    batch_size, epochs, optimizer_expert, optimizer_labels = get_mtt_attack_info(
        expert_model.parameters(),
        labels_syn,
        config['expert_kwargs'],
        config['labels_kwargs'],
        batch_size=batch_size,
        epochs=epochs
    )

    mtt_dataloader, _ = either_dataloader_dataset_to_both(mtt_dataset,
                                                          batch_size=batch_size)
   
    
    single_batch_loader = temp_dataloader(mtt_dataset, batch_size=1)
    losses = []
    with make_pbar(total=config['iterations'] * len(mtt_dataset)) as pbar:
        for i in range(config['iterations']):
            for x_t, y_t, x_d, y_true, idx in mtt_dataloader: #脏图  脏标签  干净图 真实标签 索引
                y_d = labels_syn[idx]  #y_d是合成标签
                x_t, y_t, x_d, y_d = x_t.to(device), y_t.to(device), x_d.to(device), y_d.to(device)

                # Load parameters
                checkpoint = torch.load(expert_starts[i])
                expert_model.load_state_dict(checkpoint)
                student_model.load_state_dict({k: v.clone() for k, v in checkpoint.items()})
                expert_start = [v.clone() for v in expert_model.parameters()]

                optimizer_expert.load_state_dict(torch.load(expert_opt_starts[i]))
                state_dict = torch.load(expert_opt_starts[i])

                # Take a single expert / poison step
                expert_model.train()
                expert_model.zero_grad()
                loss = clf_loss(expert_model(x_t), y_t)
                loss.backward()
                optimizer_expert.step()
                expert_model.eval()

                # Train a single student step
                student_model.train()
                student_model.zero_grad()
                loss = clf_loss(student_model(x_d), softmax(y_d))
                grads = torch.autograd.grad(loss, student_model.parameters(), create_graph=True)

                # Calculate loss
                param_loss = torch.tensor(0.0).to(device)
                param_dist = torch.tensor(0.0).to(device)

                for initial, student, expert, grad, state in zip(expert_start,
                                                                 student_model.parameters(),
                                                                 expert_model.parameters(),
                                                                 grads,
                                                                 state_dict['state'].values()):
                    student_update = sgd_step(student, grad, state, state_dict['param_groups'][0])

                    param_loss += total_mse_distance(student_update, expert)
                    param_dist += total_mse_distance(initial, expert)

                # Add Regularization and calculate loss
                reg_term = lam * torch.linalg.vector_norm(softmax(labels_syn) - labels_init, ord=1, axis=1).mean()
                grand_loss = (param_loss / param_dist) + reg_term
                g_loss = grand_loss.item()

                # Optimize labels and learning rate
                optimizer_labels.zero_grad()
                grand_loss.backward()
                optimizer_labels.step()
                losses.append(g_loss)

    if pre_computed_scores_exist:
        mask = torch.zeros(len(labels_syn), dtype=torch.bool)
    
        for round_num in range(3):  # 分成3个轮次
            print(f"Round {round_num + 1}...")
    
            # 用x_d和label_syn来完整地训练模型var_theta
            model_a = load_model(expert_model_flag, n_classes).to(device)
            optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(150):  # 训练模型var_theta
                model_a.train()
                for x_t, y_t, x_d, y_true, idx in mtt_dataloader:
                    x_d = x_d.to(device)
                    y_d = labels_syn[idx].to(device)
                    
                    optimizer_a.zero_grad()
                    outputs = model_a(x_d)
                    loss = criterion(outputs, torch.argmax(y_d, dim=1))
                    loss.backward()
                    optimizer_a.step()
                    break
                
            # 测试模型A，找出Mask为0的样本中所有会被模型A错误分类的样本
            model_a.eval()
            incorrect_indices = []
            with torch.no_grad():
                for x_t, y_t, x_d, y_true, idx in single_batch_loader:
                    if mask[idx]:  # 跳过Mask为1的样本
                        continue
                    pred = torch.argmax(model_a(x_d.to(device)), dim=1)
                    if pred.item() != torch.argmax(y_true).item():
                        incorrect_indices.append(idx)
            
            cscores_path = "./precomputed_scores/forgetting_CIFAR10.npy"
            cscores = np.load(cscores_path)
            cscores = torch.zeros(len(labels_syn))
            incorrect_scores = [(idx, cscores[idx]) for idx in incorrect_indices]
            incorrect_scores.sort(key=lambda x: x[1])  
    
            num_to_select = max(1, len(incorrect_scores) // (10 // 3))  
            selected_indices = [idx for idx, _ in incorrect_scores[:num_to_select]]
            
            with torch.no_grad():
                for idx in selected_indices:
                    labels_syn[idx].requires_grad = False 
                    labels_syn[idx] = labels_init[idx] 
                mask[idx] = True
    
            
            print("Re-matching with updated labels...")
            with make_pbar(total=config['iterations'] * len(selected_indices)) as pbar:
                for i in range(config['iterations']):
                    for x_t, y_t, x_d, y_true, idx in mtt_dataloader:
                        # Prepare data
                        y_d = labels_syn[idx]
                        x_t, y_t, x_d, y_d = x_t.to(device), y_t.to(device), x_d.to(device), y_d.to(device)
                        
                        # Load parameters
                        checkpoint = torch.load(expert_starts[i])
                        expert_model.load_state_dict(checkpoint)
                        student_model.load_state_dict({k: v.clone() for k, v in checkpoint.items()})
                        expert_start = [v.clone() for v in expert_model.parameters()]
    
                        optimizer_expert.load_state_dict(torch.load(expert_opt_starts[i]))
                        state_dict = torch.load(expert_opt_starts[i])
                        
                        # Take a single expert / poison step
                        expert_model.train()
                        expert_model.zero_grad()
                        loss = clf_loss(expert_model(x_t), y_t)
                        loss.backward()
                        optimizer_expert.step()
                        expert_model.eval()
        
                        # Train a single student step
                        student_model.train()
                        student_model.zero_grad()
                        loss = clf_loss(student_model(x_d), softmax(y_d))
                        grads = torch.autograd.grad(loss, student_model.parameters(), create_graph=True)
        
                        # Calculate loss
                        param_loss = torch.tensor(0.0).to(device)
                        param_dist = torch.tensor(0.0).to(device)
        
                        for initial, student, expert, grad, state in zip(expert_start,
                                                                         student_model.parameters(),
                                                                         expert_model.parameters(),
                                                                         grads,
                                                                         state_dict['state'].values()):
                            student_update = sgd_step(student, grad, state, state_dict['param_groups'][0])
        
                            param_loss += total_mse_distance(student_update, expert)
                            param_dist += total_mse_distance(initial, expert)
        
                        # Add Regularization and calculate loss
                        reg_term = lam * torch.linalg.vector_norm(softmax(labels_syn) - labels_init, ord=1, axis=1).mean()
                        grand_loss = (param_loss / param_dist) + reg_term
                        g_loss = grand_loss.item()
                        
                        # Optimize labels and learning rate
                        optimizer_labels.zero_grad()
                        grand_loss.backward()
                        optimizer_labels.step()
                    
            
    # Save results
    print("Saving results...")
    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
