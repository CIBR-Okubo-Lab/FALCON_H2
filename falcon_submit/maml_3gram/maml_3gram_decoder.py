r"""
    MAMLGRU decoder for h2 dataset, with LLM API
"""

import math
import numbers
import numpy as np
import os
from pathlib import Path
import pickle
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
from typing import List
import utils.lmDecoderUtils as lmDecoderUtils

from falcon_challenge.config import FalconConfig
from falcon_challenge.interface import BCIDecoder


ac_scale = 0.8
n_best = 10
llm_weight = 1
rnn_weight = 0.5
lmDir = "./language_model"
ngramDecoder = lmDecoderUtils.build_lm_decoder(lmDir, acoustic_scale=ac_scale, nbest=n_best, beam=17)

def rearrange_handwriting_logits(logits, includeSpaceSymbol=True):
    char_range = list(range(1, 27))
    if includeSpaceSymbol:
        symbol_range = [27, 28, 31, 30, 29]
    else:
        symbol_range = [28, 31, 30, 29]
    logits_new = []
    for i in range(len(logits)):
        logit = logits[i]
        logit_new = logit[:, [0] + symbol_range + char_range]
        logits_new.append(logit_new)
    return logits_new


from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', cache_dir='./gpt2_xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl', cache_dir='./gpt2_xl')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token



def h2_relabel_hotfix(dandi_tag):
    if dandi_tag.startswith('ses'): # e.g. ses-20220518
        evaluator_tag = dandi_tag.split('-')[1]
        evaluator_tag = evaluator_tag[:4] + '.' + evaluator_tag[4:6] + '.' + evaluator_tag[6:]
        return evaluator_tag
    return dandi_tag

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")

class MAMLAligner(pl.LightningModule):
    def __init__(
        self,
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
    

def loadModel(modelPath, device="cuda", last=True):
    # load pl model
    if last:
        pl_model = torch.load(modelPath+"/last.ckpt", map_location=device)
    else:
        pl_model = torch.load(modelPath+"/modelWeights", map_location=device)
    
    # load hyperparameters
    with open(os.path.join(modelPath, "args"), "rb") as handle:
        args = pickle.load(handle)
    state_dict = pl_model["state_dict"]

    if "single_layer" not in args.keys():
        print("Single layer")
        single_layer = True
        align_hidden = None
    else:
        single_layer = args["single_layer"]
        align_hidden = args["align_hidden"]
    
    model = MAMLAligner(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        dropout=0, # close dropout when fast adaptation
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"], 
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        first_order=args["first_order"],
        inner_lr=args["inner_lr"],
        outer_lr=args["outer_lr"],
        optimizer_type=args["optimizer_type"],
        use_scheduler=args["use_scheduler"],
        l2_decay=args["l2_decay"], 
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        inner_steps=args["inner_steps"],
        single_layer=single_layer,
        align_hidden=align_hidden,
    ).to(device)

    model.load_state_dict(state_dict)
    return model

CHAR_DEF = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z',
            '>', ',', "'", '~', '?']

def charToId(c):
    return CHAR_DEF.index(c)

class MAMLLMDecoder(BCIDecoder):
    r"""
        Load a MAML GRU decoder, add LLM
    """

    def __init__(
            self,
            task_config: FalconConfig,
            model_path: str,
            batch_size: int = 1,
            device = 'cuda'
        ):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self._task_config = task_config
        self.batch_size = batch_size
        self.model = loadModel(modelPath=model_path, device=device, last=True)
        self.path = model_path

        with open(model_path+"/avg_std", 'rb') as f:
            avg_and_std = pickle.load(f)
            self.x_mean = avg_and_std['avg']
            self.x_std = avg_and_std['std']

        context_limit = 5000 # limit for save one trial neural data
        self.observation_buffer = torch.zeros((
            context_limit,
            self.batch_size,
            task_config.n_channels
        ), dtype=torch.uint8, device=device)

    def reset_buffer(self):
        self.set_steps = 0
        self.observation_buffer.zero_()

    def reset(self, dataset_tags: List[Path] = [""]):
        r"""
            Apply every batch (1 session data), before applying predict and ondone
            Load the avg and std for day-wise normalization
            
            Also apply after each ondone step
            Reset observation buffer
        """
        # batch size is 1, so just load the one tag
        dataset_tag = self._task_config.hash_dataset(dataset_tags[0])
        dataset_tag = h2_relabel_hotfix(dataset_tag)
        if dataset_tag not in self.x_mean:
            raise ValueError(f"Dataset tag {dataset_tag} not found in calibration set {self.x_mean.keys()} - did you calibrate on this dataset?")
        self.local_x_mean = torch.tensor(self.x_mean[dataset_tag], device=self.model.device)
        self.local_x_std = torch.tensor(self.x_std[dataset_tag], device=self.model.device)
        self.local_weights = torch.load(self.path+'/'+dataset_tag+'_weights.pth')
        self.reset_buffer()

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts

        """
        self.observe(neural_observations)

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        self.observation_buffer[self.set_steps] = torch.as_tensor(neural_observations, dtype=torch.uint8, device=self.model.device)
        self.set_steps += 1

    def on_done(self, dones: np.ndarray):
        spikes_in = self.observation_buffer[:self.set_steps]
        # Normalization
        # Transform to 2 dimension: (timesteps, batch, neural dim) -> (timesteps, neural dim)
        spikes_2d = spikes_in.squeeze(dim=1)
        spikes_normed = (spikes_2d - self.local_x_mean) / (self.local_x_std + 1e-8) # normalization
        # Transform to 3 dimension: (timesteps, neural dim) -> (batch, timesteps, neural dim)
        # batch first to apply gru decoder
        spikes_3d = torch.unsqueeze(spikes_normed, dim=0)
        # Model output
        self.model.load_state_dict(self.local_weights, strict=False)
        pred = self.model(spikes_3d) # shape: (batch, timesteps, n_token:31+1)
        # Generate sentence (greedy search)
        
        decodedProb = pred[0].cpu().detach()
        decodedProb = rearrange_handwriting_logits([decodedProb], includeSpaceSymbol=True)
        

        nbest = lmDecoderUtils.lm_decode(ngramDecoder,
                                         decodedProb[0],
                                         blankPenalty=np.log(11),
                                         returnNBest=True,
                                         rescore=False,)


        decoded, _ = lmDecoderUtils.gpt2_lm_decode(model, tokenizer, nbest, rnn_weight, lengthPenlaty=0, alpha=llm_weight, returnConfidence=True)
        decoded = decoded.replace(' ', '>')
        decoded = decoded.replace('.', '~')
        decoded = decoded.strip()

        self.reset_buffer()
        return decoded