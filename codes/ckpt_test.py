import os
from tqdm import tqdm

from torch.nn.functional import softmax
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from codes.utils.functions import setup_parser
from codes.model.lightning_demel import LightningForDEMEL
from codes.utils.dataset import DataModuleForDEMEL


class ValidationErrorAnalysisCallback(Callback):
    def __init__(self, output_dir=" ", batch_size=32, device="cuda:0"):
        super().__init__()
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.preds = []
        self.targets = []
        self.logits = []
        self.sample_ids = []
        self.entropy = None

    def on_test_start(self, trainer: Trainer, pl_module):
        os.makedirs(self.output_dir, exist_ok=True)
        pl_module.to(self.device)
        val_dataloader = trainer.datamodule.val_dataloader()

        pl_module.eval()
        pl_module.on_validation_start()
        outputs_list = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch.data = {k: v.to(self.device) for k, v in batch.data.items()}
                outputs = pl_module.validation_step(batch, 0)
                outputs_list.append(outputs)

                self.logits.append(outputs["logits"].detach())
                self.preds.append(torch.argmax(outputs["logits"], dim=1).detach())
                self.targets.append(outputs["targets"].detach())

        pl_module.validation_epoch_end(outputs_list)

        self.logits = torch.cat(self.logits, dim=0)
        self.preds = torch.cat(self.preds, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        error_mask = self.preds != self.targets
        error_indices = torch.where(error_mask)[0].cpu().numpy()
        self.entropy = -torch.sum(softmax(self.logits, dim=1) * torch.log(softmax(self.logits, dim=1) + 1e-10), dim=1)
        entropy_range = self.find_optimal_entropy_range(error_indices, len(error_indices))
        pl_module.entropy_l_thresh = entropy_range[0]
        pl_module.entropy_h_thresh = entropy_range[1]

        del self.logits, self.preds, self.targets
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = setup_parser()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    data_module = DataModuleForDEMEL(args)
    lightning_model = LightningForDEMEL.load_from_checkpoint(
        ' ',
        strict=False).to(torch.device('cuda:0'))
    logger = pl.loggers.CSVLogger("./runs", name=args.run_name, flush_logs_every_n_steps=30)

    error_analysis_callback = ValidationErrorAnalysisCallback(
        output_dir=" ",
        device="cuda:0"
    )
    ckpt_callbacks = ModelCheckpoint(monitor='Val/mrr', save_weights_only=True, mode='max')
    early_stop_callback = EarlyStopping(monitor="Val/mrr", min_delta=0.00, patience=5, verbose=True, mode="max")

    trainer = pl.Trainer(**args.trainer,
                         deterministic=True, logger=logger, default_root_dir="./runs",
                         callbacks=[error_analysis_callback, ckpt_callbacks, early_stop_callback])
    trainer.test(lightning_model, datamodule=data_module)
