
import json
import torch

from litgpt.utils import chunked_cross_entropy

from whittle.eval.utils import convert_and_evaluate

def plot_validation_metrics(model, val_dataloader, eval, sampler):
    # compute loss for superent
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_loss(model, val_dataloader, eval)
    middle_config = sampler.get_medium_sub_network()
    model.set_sub_network(**middle_config)
    val_loss_medium = compute_loss(model, val_dataloader, eval)
    model.reset_super_network()
    smallest_config = sampler.get_smallest_sub_network()
    model.set_sub_network(**smallest_config)
    val_loss_smallest = compute_loss(model, val_dataloader, eval)
    model.reset_super_network()
    return val_loss_largest, val_loss_medium, val_loss_smallest

def plot_accuracies(model, sampler, dataset, checkpoint_dir):
    model.eval()
    model.reset_super_network()
    val_loss_largest = compute_accuracy(model, dataset, checkpoint_dir)
    middle_config = sampler.get_medium_sub_network()
    model.set_sub_network(**middle_config)
    val_loss_medium = compute_accuracy(model, dataset, checkpoint_dir)
    model.reset_super_network()
    smallest_config = sampler.get_smallest_sub_network()
    model.set_sub_network(**smallest_config)
    val_loss_smallest = compute_accuracy(model, dataset, checkpoint_dir)
    model.reset_super_network()
    return val_loss_largest, val_loss_medium, val_loss_smallest

def compute_loss(model, val_dataloader, eval):
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )

    val_loss = losses.mean()
    return val_loss

def compute_accuracy(model, dataset, checkpoint_dir):
    convert_and_evaluate(
        model,
        out_dir=checkpoint_dir,
        device=None,
        dtype=torch.float32,
        tasks=dataset,
        batch_size=16,  # Test for non-positive integer
    )
    with open(str(checkpoint_dir / "results.json")) as f:
        results = json.load(f)
    acc = results["results"][dataset]["acc_norm,none"]
    return acc