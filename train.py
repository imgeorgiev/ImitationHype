import hydra, os, wandb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torch.distributions as dist

from utils import seeding
from visualizer import visualize_policy

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Neutral", call_pdb=True
)


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    dataset_name = job_config["dataset"]["name"]
    model_name = job_config["model"]["_target_"].split(".")[-1]
    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{model_name}_{dataset_name}_sweep_{job_config['seed']}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{model_name}_{dataset_name}"
        notes = wandb_cfg["notes"]  # force user to make notes
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        sync_tensorboard=False,
        monitor_gym=False,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):

    assert torch.cuda.is_available(), "CUDA is not available"
    assert __debug__, "Python is running with optimizations enabled"

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]

    seeding(cfg.seed, False)

    device = torch.device(cfg.device)

    # load data
    dataset = instantiate(cfg.dataset)
    cfg.model.qpos_dim = dataset.qpos_dim
    cfg.model.act_len = dataset.traj_len

    if cfg.run_wandb:
        create_wandb_run(cfg.wandb, OmegaConf.to_container(cfg, resolve=True))

    # Determine the sizes of the splits
    train_size = int(cfg.train_data_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.stack(x),
        num_workers=cfg.num_dataloaders,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda x: torch.stack(x),
        num_workers=cfg.num_dataloaders,
    )

    # load agent
    model = instantiate(cfg.model)
    model.to(device)

    optimizer = instantiate(cfg.optimizer, model.parameters())

    if cfg.lr_scheduler:
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=cfg.num_epochs * len(dataloader),
        )
    else:
        lr_scheduler = None

    if cfg.checkpoint:
        model.load(cfg.checkpoint)

    # train agent
    for epoch in range(cfg.num_epochs):
        running_loss = 0.0
        running_kld_loss = 0.0
        running_norm = 0.0
        running_act_std = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                optimizer.zero_grad()
                batch = batch.to(device)
                img = batch["images"]
                qpos = batch["qpos"]
                act = batch["act"]
                act_hat, style_dist = model(img, qpos, act)
                # mask to remove end of trajectory and unused end effectors
                mask = act != 0
                loss = F.l1_loss(act, act_hat, reduction="none")
                loss = (loss * mask).mean()
                if style_dist:
                    kld_loss = dist.kl_divergence(style_dist, dist.Normal(0, 1))
                    loss += kld_loss.mean() * cfg.kl_weight
                    running_kld_loss += kld_loss.mean().item()
                loss.backward()
                running_norm += clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                running_loss += loss.item()
                running_act_std += act_hat.std(dim=1).mean().item()
                if lr_scheduler:
                    lr_scheduler.step()
                tepoch.set_postfix(loss=loss.item())
            running_loss /= len(dataloader)
            running_kld_loss /= len(dataloader)
            running_norm /= len(dataloader)
            running_act_std /= len(dataloader)
            time_per_batch = tepoch.format_dict["elapsed"] / len(dataloader)

        stats = {
            "train/loss": running_loss,
            "train/grad_norm": running_norm,
            "train/act_std": running_act_std,
            "train/time_per_batch": time_per_batch,
        }

        if style_dist:
            stats.update({"train/kld_loss": running_kld_loss})

        if lr_scheduler:
            stats.update({"train/lr": lr_scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch + 1} loss: {running_loss:.2f}")

        if epoch % cfg.val_freq == 0:
            with torch.no_grad():
                val_loss = 0.0
                with tqdm(val_dataloader, unit="batch") as tepoch:
                    for batch in tepoch:
                        batch = batch.to(device)
                        img = batch["images"]
                        qpos = batch["qpos"]
                        act = batch["act"]
                        act_hat, _ = model(img, qpos)
                        mask = act != 0
                        loss = F.l1_loss(act, act_hat, reduction="none")
                        loss = (loss * mask).mean()
                        val_loss += loss.item()
                    val_loss /= len(val_dataloader)
            stats.update({"val/loss": val_loss})
            print(f"Validation loss: {val_loss:.2f}")

        if epoch % cfg.eval_freq == 0:
            print("Video evaluation")
            video_dir = os.path.join(logdir, "eval_videos")
            os.makedirs(video_dir, exist_ok=True)
            batch = dataset.get_full_seq(torch.randint(0, len(dataloader), (1,)).item())
            video_path = visualize_policy(model, batch, video_dir, device, epoch)
            stats.update({"video": wandb.Video(video_path)})

        if epoch % cfg.save_freq == 0:
            print("Saving model")
            model_dir = os.path.join(logdir, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_name = cfg["model"]["_target_"].split(".")[-1]
            dataset_name = cfg["dataset"]["name"]
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(model_dir, f"{model_name}_{dataset_name}_e{epoch}.pt"),
            )

        if cfg.run_wandb:
            wandb.log(stats)

    if cfg.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
