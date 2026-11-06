# This is a stripped-down version of the sklearn_sample.Dockerfile in the same directory. 
# This file is intended to be a template for users to create their own Dockerfiles for EvalAI submissions.
# Comments are provided to indicate which lines are mandatory for a valid submission and which are submission-specific.

# NOTE: The environment variables PREDICTION_PATH, PREDICTION_PATH_LOCAL, GT_PATH, , and EVAL_DATA_PATH are used by the evaluator and should not be changed.
# The other environment variables could be hardcoded in the runfile, but it is recommended to set them in the Dockerfile for clarity.

# Dockerfile flow for EvalAI for a specific track

# The user prepares a Docker image with this file and decoder defined locally
# The user then directly pushes this image to EvalAI with the eval-ai cli.
# see e.g. https://eval.ai/web/challenges/challenge-page/1615/submission (You may have to register for the challenge)

# RECOMMENDED: Use a base Docker image that has the necessary dependencies for your decoder. The one provided is a good choice for PyTorch-based decoders.
# Note: If you're using TF/Jax, you may want to use a different base image.
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# OPTIONAL: Users should install additional decoder-specific dependencies here (e.g. RUN pip install package_name)
ADD requirements_3gram.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --no-deps pytorch-lightning==2.2.4
RUN pip install --no-deps torchmetrics

# Install language model
ADD LanguageModelDecoder/ LanguageModelDecoder/
WORKDIR /workspace/LanguageModelDecoder/runtime/server/x86
RUN rm -rf kaldi utils bin decoder websocket patch grpc frontend
RUN ln -s ../../core/kaldi kaldi && \
    ln -s ../../core/utils utils && \
    ln -s ../../core/bin bin && \
    ln -s ../../core/decoder decoder && \
    ln -s ../../core/websocket websocket && \
    ln -s ../../core/patch patch && \
    ln -s ../../core/grpc grpc && \
    ln -s ../../core/frontend frontend
RUN python setup.py install
WORKDIR /workspace
ADD language_model/ language_model/
ADD gpt2_xl/ gpt2_xl/
ADD utils/ utils/

# MANDATORY: Leave these lines as they are. This will install an up-to-date copy of the falcon-challenge package into the Docker image.
RUN rm -f /etc/apt/sources.list.d/*.list 2>/dev/null || true
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/snel-repo/falcon-challenge.git
RUN pip install -e falcon-challenge
RUN pip uninstall -y numpy
RUN pip install "numpy<2.0"


# MANDATORY: Leave these environment variables as they are. This is how EvalAI will find the predictions and answers during evaluation.
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH "/tmp/ground_truth.pkl"

# MANDATORY: Set EVALUATION_LOC. Use "remote" for submissions to EvalAI and "local" for local testing. This is passed to the evaluator in the runfile.
ENV EVALUATION_LOC remote
#ENV EVALUATION_LOC local

# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

ADD maml_3gram_decoder.py maml_3gram_decoder.py
ADD linear_long_25/ adapt_out/

# MANDATORY:  Add the runfile to the Docker image. This should be the path to the runfile in your local directory. This line is relevant to the sklearn_sample decoder.
ADD maml_3gram_sample.py decode.py

# MANDATORY: Set SPLIT. This specifies the dataset that your decoder is being evaluated on. This is passed to the evaluator in the runfile.
ENV SPLIT "h2"

# MANDATORY: Set PHASE. "minival" should be used for local testing and submission to the minival phase on EvalAI. "test" should be used for submissions to the test phase.
# The minival set is a small subset of the held-in sessions, and is used for local testing and debugging. The test set is the full evaluation set for all sessions.
# Note that the test set is only available remotely on EvalAI. The minival set is available locally and remotely on EvalAI.
# This is passed to the evaluator in the runfile.
#ENV PHASE "minival"
ENV PHASE "test"

# OPTIONAL: Set BATCH_SIZE for batched evaluation. Batching accelerates evaluation by processing multiple samples at once.
# Note: the batch size is an attribute of the decoder. For the example sklearn_sample decoder, the batch size is passed to the decoder's constructor in the runfile.
ENV BATCH_SIZE 1

# MANDATORY: Leave this line as it is. This is a path to the evaluation data in the remote EvalAI environment.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

# MANDATORY: Specify a command for the container to run when it is launched. This command should run the decoder on the evaluation data.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path adapt_out --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]