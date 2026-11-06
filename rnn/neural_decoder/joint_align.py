from .augmentations import GaussianSmoothing
from edit_distance import SequenceMatcher
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

class JointAligner(pl.LightningModule):
    def __init__(
        self,
        pretrain_model,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        dropout,
        strideLen,
        kernelLen, 
        gaussianSmoothWidth,
        whiteNoiseSD, 
        constantOffsetSD,
        bidirectional,
        outer_lr, 
        optimizer_type,
        use_scheduler,
        l2_decay,
        momentum,
        nesterov,
        gamma,
        stepSize,
        single_layer=True,
        align_hidden=None
    ):
        """
        GRU decoder for MAML learning
        
        Args:
            nerual_dim: Neural feature input dimension
            n_classes: Number of char classes, 26 characters + 5 punctuation marks
            hidden_dim: Hidden dimension for GRU layers
            layer_dim: Number of GRU hidden layers
            dropout: Dropout rate
            strideLen: Stride length for sliding window
            kernelLen: Kernel length for sliding window, history length
            gaussianSmoothWidth: Size of the gaussian kernel
            whiteNoiseSD: White noise standard deviation
            constantOffsetSD: Constant offset standard deviation
            bidirectional: GRU type, use bidirectional GRU or not
            first_order: Use first order MAML or not
            inner_lr: Inner loop learning rate for gradient descent
            outer_lr: Outer loop learning rate
            optimizer_type: Outer loop optimizer
            use_scheduler: Apply scheduler on outer loop optimizer or not
            l2_decay: l2 regularization weight decay
            momentum: Momentum, if use SGD optimizer
            nesterov: Nesterov, if use SGD optimizer
            gamma: Gamma of the scheduler, if use scheduler
            stepSize: Step size of the scheduler, if use scheduler
            inner_steps: Inner loop steps
        """
        super().__init__()

        # load original model weights
        pre_model = torch.load(pretrain_model)
        state_dict = pre_model["state_dict"]
        new_state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.constantOffsetSD = constantOffsetSD
        self.whiteNoiseSD = whiteNoiseSD
        self.loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        self.outer_lr = outer_lr
        self.optimizer_type = optimizer_type
        self.use_scheduler = use_scheduler
        self.l2_decay = l2_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.gamma = gamma
        self.stepSize = stepSize

        self.testLoss = []
        self.testCER = []

        # Aligner, linear layer
        if single_layer:
            self.aligner = nn.Linear(neural_dim, neural_dim)
        else:
            self.aligner = nn.Sequential(
                nn.Linear(neural_dim, align_hidden),
                nn.ReLU(), 
                nn.Linear(align_hidden, neural_dim)
                )

        # GRU layers, neural decoder
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

        # Use pretrained weights
        self.load_state_dict(new_state_dict, strict=False)

        # Freeze all the paramters, except aligner
        for param in self.parameters():
            param.requires_grad = False
        for param in self.aligner.parameters():
            param.requires_grad = True


    def forward(self, neuralInput):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        neuralInput = self.aligner(neuralInput)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(neuralInput, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                neuralInput.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                neuralInput.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
    

    def training_step(self, batch, batch_idx):
        X, y, X_len, y_len, _ = batch

        # Noise augmentation is faster on GPU
        if self.whiteNoiseSD > 0:
            X += torch.randn(X.shape, device=self.device) * self.whiteNoiseSD

        if self.constantOffsetSD > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=self.device)
                * self.constantOffsetSD
            )

        # Compute prediction error
        pred = self.forward(X)
        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.kernelLen) / self.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)
        if self.use_scheduler:
            schedulers = self.lr_schedulers()
            cur_lr = schedulers.optimizer.param_groups[-1]["lr"]
        else:
            cur_lr = self.optimizers().param_groups[0]['lr']
        self.log_dict({"train/predictionLoss": loss, "train/learning_rate": cur_lr}, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return {"loss": loss, "pred": pred, "y":y}
    
    
    def validation_step(self, batch, batch_idx): 
        X, y, X_len, y_len, _ = batch

        total_edit_distance = 0
        total_seq_length = 0

        pred = self.forward(X)
        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.kernelLen) / self.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        adjustedLens = ((X_len - self.kernelLen) / self.strideLen).to(
            torch.int32
        )
        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                pred[iterIdx, 0 : adjustedLens[iterIdx], :].clone().detach(),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(
                y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
            )

            matcher = SequenceMatcher(
                a=trueSeq.tolist(), b=decodedSeq.tolist()
            )
            total_edit_distance += matcher.distance()
            total_seq_length += len(trueSeq)

        avgDayLoss = loss
        cer = total_edit_distance / total_seq_length

        self.log_dict({"val/predictionLoss": avgDayLoss, "val/ser": cer}, sync_dist=True, prog_bar=True, rank_zero_only=True)

        return {"loss": loss, "pred": pred, "y":y}


    def configure_optimizers(self):
        # Outer loop optimizer
        if self.optimizer_type == 'SGD':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.outer_lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.l2_decay,
            )
        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(
            self.parameters(),
            lr=self.outer_lr,
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=self.l2_decay,
        )

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.stepSize,
                gamma=self.gamma,
            )
            return {"optimizer":optimizer, 
                    "lr_scheduler":
                        {
                        "scheduler": scheduler,
                        "interval": "step",
                        }
                    }
        else:
            return {"optimizer":optimizer}
