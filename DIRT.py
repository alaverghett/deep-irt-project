import pandas as pd
import torch
import pytorch_lightning as pl
import wandb
import argparse

from os.path import join
from torch import nn
from torch.nn import functional as f
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

DATA_PATH = "Z:/Code/item-response-theory/project/data"
CSV_NAME = "Fraction_23.csv"
DOCX_NAME = "Fraction Test_English.xlsx"
DEVICE = "cuda"


def LoadItemResponses() -> torch.Tensor:
    item_responses = pd.read_csv(join(DATA_PATH, CSV_NAME))
    # transpose so that the tensor is of shape (num_items * num_responses)
    item_responses = item_responses.sample(frac=0.8, random_state=1)
    item_responses = torch.Tensor(item_responses.T.to_numpy())
    # item_responses = item_responses.long()
    print(item_responses.shape)
    return item_responses


# return pre-computed sentence embeddings
def LoadItemContent(encoder: str) -> torch.Tensor:
    item_content = pd.read_excel(join(DATA_PATH, DOCX_NAME), index_col=0)
    item_encoder = SentenceTransformer(encoder)
    items_raw = list(item_content["Item Content"])
    item_embeddings = torch.Tensor(item_encoder.encode(items_raw))
    return item_embeddings


# the deep irt model requires a custom activation function
# corresponding to the 1pl function
# minibatching should be done along items, NOT respondents
# the acutal loss is just MSE or MAE
# BETA MUST BE A SCALAR TO CALCULATE THE ACTIVATION CORRECTLY
def onePL_activation(theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # scaling constant
    D = 1.7

    p_theta = 1 / (1 + torch.exp(-D * (theta - beta)))
    return p_theta


# the beta paramter must be kept in the range [-4,4]
# the function to do this comes from the DIRT paper
def beta_activation(beta: torch.Tensor) -> torch.Tensor:
    sigmoid = nn.Sigmoid()
    beta = 8 * (sigmoid(beta) - 0.5)
    return beta


# load betas into a dataloader
# but not thetas, they are all updated at once
# whereas loss function only accepts a single beta
def prepare_dataloader(y_true: torch.Tensor, beta: torch.Tensor) -> DataLoader:
    data_dict = {
        "beta": beta,
        "y_true": y_true
    }
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # do NOT change batch size
    return dataloader


# TODO: import eval metrics
class Deep_IRT(pl.LightningModule):
    def __init__(
        self,
        loss,
        lr: float,
        weight_decay: float,
        betas_shape: int,
        num_respondents: int,
        drop_p: float,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.thetas = self.InitThetas(num_respondents)
        self.loss = loss

        # encode theta
        self.fcTheta = nn.Linear(in_features=self.thetas.shape[0], out_features=self.thetas.shape[0])
        self.fcdrop = nn.Dropout(p=drop_p)

        # encode beta
        self.fcBeta = nn.Linear(in_features=betas_shape, out_features=betas_shape)
        self.actBeta = beta_activation

        # 1pl activatioon
        self.finalAct = onePL_activation
    
    # initalize an ability vector (theta) from a normal distribution
    # the vector must be of length equal to the number of test takers
    # it functions similarity to the hidden layer in an RNN
    def InitThetas(self, num_respondents: int) -> torch.Tensor:
        return torch.normal(mean=0, std=1, size=(num_respondents,), requires_grad=False).to(DEVICE)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.weight_decay
        )
        return optimizer

    # the training_setp, val_step, and test_step functions are required for a torch model
    # they handle the logic for the training, validation, and test loops, respectively
    # batch_idx is not used and only included for API compatibility
    # it correspondings to the index of the batch in the dataloader
    def training_step(self, batch, batch_idx):
        labels = batch["y_true"].squeeze(0)
        logits = self.forward(batch["beta"])
        
        # logits = torch.unsqueeze(logits, 0)
        train_loss = self.loss(logits, labels)

        # in regression, loss is also the validation metric
        # accuracy is not well defined, so don't calculate
        self.log("train_loss", train_loss.item())
        self.log("beta_weight", self.fcBeta.weight.item())
        self.log("theta_weight", torch.mean(self.fcTheta.weight).item())
        return train_loss

    def validiation_step(self, batch, batch_idx):
        labels = torch.tensor(batch["y_true"]).squeeze(0)
        logits = self.forward(self.thetas, batch["beta"])
        
        # logits = torch.unsqueeze(logits, 0)
        val_loss = self.loss(logits, labels)

        # in regression, loss is also the validation metric
        # accuracy is not well defined, so don't calculate
        self.log("val_loss", val_loss.item())
        return val_loss

    def test_step(self, batch, batch_idx):
        labels = torch.tensor(batch["y_true"]).squeeze(0)
        logits = self.forward(self.thetas, batch["beta"])
        
        # logits = torch.unsqueeze(logits, 0)
        test_loss = self.loss(logits, labels)

        # in regression, loss is also the validation metric
        # accuracy is not well defined, so don't calculate
        self.log("test_loss", test_loss.item())
        return test_loss

    # the computation graph is defined in forward
    # actual execution is handeled by torch lightning
    # so there will be no calls to backward, optimizer step, or no_grad for the validation setps
    # the framework takes care of all of that for us
    def forward(
        self,
        betas: torch.Tensor,
    ):
        # 1. take the mean of beta along dim 1
        betas = torch.mean(betas, 1, keepdim=False)

        # 1. fully connected layer + non-linearity for theta and beta
        self.thetas = self.fcTheta(self.thetas)
        betas = self.fcBeta(betas)
        betas = self.actBeta(betas)

        # 2. pass to final non-linearity, the 1pl
        p_i = self.finalAct(self.fcdrop(self.thetas), betas)

        self.thetas.detach_()

        return p_i # thetas are a hidden state like in RNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float)
    parser.add_argument('--weight_decay',type=float)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--drop_p',type=float)
    parser.add_argument('--encoder', type=str, default="all-MiniLM-L12-v2")
    args = parser.parse_args()
    config = {
        "loss": 'Binary Cross Entropy',
        "lr": args.lr,
        "weight_decay": args.weight_decay, # momentum
        "epochs": args.epochs,
        "encoder": args.encoder,
        "wandb_log_freq": 1,
        "drop_p": args.drop_p
    }
    run = wandb.init(config=config)
    
    item_responses = LoadItemResponses().to(DEVICE)
    item_embeddings = LoadItemContent(config["encoder"]).to(DEVICE)

    # thetas are not part of the dataloader
    # because they are a separate parameter being estimated
    theta = InitThetas(item_responses.shape[1])
    train_dataloader = prepare_dataloader(item_responses, item_embeddings)


    # TODO: for now train_dataset = val_dataset
    # must fix later
    val_dataloader = train_dataloader
    irt_model = Deep_IRT(
        # TODO: move to config in wandb
        thetas=theta,
        loss=nn.BCELoss(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        num_respondents=item_responses.shape[1],
        betas_shape=1, # must be 1
        drop_p=config["drop_p"]
    )

    wandb_logger = pl.loggers.WandbLogger(project="item_response_theory")
    # TODO: early stopping callback?
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./model_checkpoints/{config['encoder']}/",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    wandb_logger.watch(irt_model, log="all", log_freq=config["wandb_log_freq"])
    trainer = pl.Trainer(
        accelerator=DEVICE,
        logger=wandb_logger,
        log_every_n_steps=config["wandb_log_freq"],
        max_epochs=config["epochs"],
        # add the callback if you want early stopping
        # early_stop_callback
        callbacks=[checkpoint_callback],
    )
    trainer.fit(irt_model, train_dataloader, val_dataloader)

    # TODO:
    # if config["use_test_set"]:
    #     trainer.test(dataloaders=test_dataloader)

    wandb.finish()
