r"""
    MAML GRU decoder for the Falcon Challenge H2 dataset.
"""

import argparse
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from maml_3gram_decoder import MAMLLMDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False, default='./h2_meta_aligner/adapt_out'
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h2',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument('--batch-size', type=int, help='size of batch for evaluation', default=1)
    args = parser.parse_args()

    task = getattr(FalconTask, args.split)
    config = FalconConfig(
        task=task,
    )

    decoder = MAMLLMDecoder(task_config=config, model_path=args.model_path, batch_size=args.batch_size)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()