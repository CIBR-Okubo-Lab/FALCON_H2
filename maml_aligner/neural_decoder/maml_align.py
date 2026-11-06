from .augmentations import GaussianSmoothing
from edit_distance import SequenceMatcher
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.func import functional_call
from torch.autograd import grad as autograd_grad

class MAMLAligner(pl.LightningModule):
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
        first_order,
        inner_lr, 
        outer_lr, 
        optimizer_type,
        use_scheduler,
        l2_decay,
        momentum,
        nesterov,
        gamma,
        stepSize,
        inner_steps,
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

        self.first_order = first_order
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.optimizer_type = optimizer_type
        self.use_scheduler = use_scheduler
        self.l2_decay = l2_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.gamma = gamma
        self.stepSize = stepSize

        self.inner_steps = inner_steps
        self.testLoss = []
        self.testCER = []

        # MAML aligner, linear layer
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
    
    # One MAML optimization step
    def training_step(self, batch, batch_idx):
        # original parameters
        params = {n: p for n, p in self.named_parameters()}

        n_task = len(batch['support'])
        k_tr = batch['support'][0][0].shape[0]
        k_ts = batch['query'][0][0].shape[0]

        # inner loop
        def task_loss(batch_train, batch_test, params):
            X_tr, y_tr, X_len_tr, y_len_tr, _ = batch_train
            X_ts, y_ts, X_len_ts, y_len_ts, _ = batch_test

            # Noise augmentation is faster on GPU
            if self.whiteNoiseSD > 0:
                X_tr += torch.randn(X_tr.shape, device=self.device) * self.whiteNoiseSD
                # TODO: do we need to add noise on test dataset?
                X_ts += torch.randn(X_ts.shape, device=self.device) * self.whiteNoiseSD

            if self.constantOffsetSD > 0:
                X_tr += (
                    torch.randn([X_tr.shape[0], 1, X_tr.shape[2]], device=self.device)
                    * self.constantOffsetSD
                )
                X_ts += (
                    torch.randn([X_ts.shape[0], 1, X_ts.shape[2]], device=self.device)
                    * self.constantOffsetSD
                )  
            def inner_loss(params, x, y, x_len, y_len):
                # with torch.backends.cudnn.flags(enabled=False):
                pred = functional_call(self, params, x)
                log_probs = torch.permute(pred.log_softmax(2), [1, 0, 2])  # (T, N, C)
                adjustedLens = ((x_len - self.kernelLen) / self.strideLen).to(torch.int32)
                ctc_loss = nn.functional.ctc_loss(
                        log_probs, y, adjustedLens, y_len, blank=0, reduction="mean", zero_infinity=True
                    )
                return ctc_loss
            # ------ Inner Loop ------ #
            for _ in range(self.inner_steps):
                loss = inner_loss(params, X_tr, y_tr, X_len_tr, y_len_tr)
                grads = autograd_grad(loss, 
                                      [p for p in params.values() if p.requires_grad], 
                                      create_graph=not self.first_order)
                updated_params = {}
                trainable_keys = [k for k, p in params.items() if p.requires_grad]
                for key, grad in zip(trainable_keys, grads):
                    updated_params[key] = params[key] - self.inner_lr * grad
                params.update(updated_params)
            
            # After inner loop
            pred_ts = functional_call(self, params, X_ts)
            log_probs_ts = torch.permute(pred_ts.log_softmax(2), [1, 0, 2])
            adjustedLens_ts = ((X_len_ts - self.kernelLen) / self.strideLen).to(torch.int32)
            ctc_loss_ts = nn.functional.ctc_loss(
                    log_probs_ts,  # (T, N, C)
                    y_ts,         # (N, S)
                    adjustedLens_ts,  # (N,)
                    y_len_ts,     # (N,)
                    blank=0, reduction="mean", zero_infinity=True
                )
            return ctc_loss_ts

        # Compute meta-loss across all tasks
        maml_loss = torch.stack([task_loss(batch['support'][task_id], batch['query'][task_id], params) for task_id in range(n_task)]).mean()

        # ------ Outer Loop ------ #
        # PL will automatically do this
        
        if self.use_scheduler:
            schedulers = self.lr_schedulers()
            cur_lr = schedulers.optimizer.param_groups[-1]["lr"]
        else:
            cur_lr = self.optimizers().param_groups[0]['lr']
        self.log_dict({"train/predictionLoss": maml_loss, "train/learning_rate": cur_lr}, 
                      batch_size=n_task*(k_tr+k_ts),
                      on_step=True, 
                      on_epoch=False, 
                      prog_bar=True, 
                      sync_dist=False,  # for single GPU
                      rank_zero_only=True)
        return maml_loss # Lightning will change it to {"loss": maml_loss}
    
    def validation_step(self, batch, batch_idx): 
        self.train()
        torch.set_grad_enabled(True) # for fast adaptation
        n_task = len(batch['support']) # number of validation days
        k_val_fs = batch['support'][0][0].shape[0]
        k_val = batch['query'][0][0].shape[0]
        params = {n: p for n, p in self.named_parameters()}

        def task_loss(batch_fs, batch_val, params):
            task_edit_distance = 0
            task_seq_length = 0
            X_fs, y_fs, X_len_fs, y_len_fs, _ = batch_fs
            X_val, y_val, X_len_val, y_len_val, _ = batch_val
            def inner_loss(params, x, y, x_len, y_len):
                #with torch.backends.cudnn.flags(enabled=False):
                # use functional call
                pred = functional_call(self, params, x)
                # CTC loss
                log_probs = torch.permute(pred.log_softmax(2), [1, 0, 2])  # (T, N, C)
                adjustedLens = ((x_len - self.kernelLen) / self.strideLen).to(torch.int32)
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,  # (T, N, C)
                    y,         # (N, S)
                    adjustedLens,  # (N,)
                    y_len,     # (N,)
                    blank=0, reduction="mean", zero_infinity=True
                )
                return ctc_loss
            # ------ Inner Loop ------ #
            for _ in range(self.inner_steps):
                loss = inner_loss(params, X_fs, y_fs, X_len_fs, y_len_fs)
                grads = autograd_grad(loss, 
                                      [p for p in params.values() if p.requires_grad], 
                                      create_graph=not self.first_order)
                updated_params = {}
                trainable_keys = [k for k, p in params.items() if p.requires_grad]
                for key, grad in zip(trainable_keys, grads):
                    updated_params[key] = params[key] - self.inner_lr * grad
                params.update(updated_params)
            # After inner loop
            # Compute final validation loss with fast weights
            with torch.no_grad():
                pred_val = functional_call(self, params, X_val)
                log_probs_val = torch.permute(pred_val.log_softmax(2), [1, 0, 2])
                adjustedLens_val = ((X_len_val - self.kernelLen) / self.strideLen).to(torch.int32)
                ctc_loss_val = nn.functional.ctc_loss(
                        log_probs_val,  # (T, N, C)
                        y_val,         # (N, S)
                        adjustedLens_val,  # (N,)
                        y_len_val,     # (N,)
                        blank=0, reduction="mean", zero_infinity=True
                    )
            for iterIdx in range(pred_val.shape[0]):
                decodedSeq = torch.argmax(
                    pred_val[iterIdx, 0 : adjustedLens_val[iterIdx], :].clone().detach(),
                    dim=-1,
                )  # [num_seq,]
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(
                    y_val[iterIdx][0 : y_len_val[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                task_edit_distance += matcher.distance()
                task_seq_length += len(trueSeq)
            return ctc_loss_val, task_edit_distance, task_seq_length
        
        # CTC loss for all tasks, each task use an average CTC loss
        val_losses = []
        edit_distances = []
        seq_lengths = []
        for task_id in range(n_task):
            ctc_loss_val, task_edit_distance, task_seq_length = task_loss(batch['support'][task_id], batch['query'][task_id], params)
            val_losses.append(ctc_loss_val.item())
            edit_distances.append(task_edit_distance)
            seq_lengths.append(task_seq_length)
        avgDayLoss = np.mean(val_losses)
        cer = sum(edit_distances) / sum(seq_lengths)

        self.log_dict({"val/predictionLoss": avgDayLoss, "val/cer": cer}, 
                      batch_size=n_task*(k_val_fs+k_val),
                      sync_dist=False, # False
                      prog_bar=True, 
                      rank_zero_only=True)
        return {"val_loss": avgDayLoss, "val_cer": cer}

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
