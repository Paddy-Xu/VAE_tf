import argparse
import os
import sys
current_path = os.path.abspath(__file__)
home_dir = os.path.dirname(os.path.dirname(current_path))
from train import *

def get_argparser():
    parser = argparse.ArgumentParser(usage='this is just a parser')
    parser.add_argument("args", help="Arguments passed to script",
                        nargs=argparse.REMAINDER)

    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to mpunet project folder')
    parser.add_argument("-f", help="Predict on a single file")
    parser.add_argument("-l", help="Optional single label file to use with -f")
    parser.add_argument("--dataset", type=str, default="test",
                        help="Which dataset of those stored in the hparams "
                             "file the evaluation should be performed on. "
                             "Has no effect if a single file is specified "
                             "with -f.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch_size")
    parser.add_argument("--memory_size", type=int, default=1024,
                        help="Omemory_size")
    parser.add_argument("--loss_type", type=str, default='standard kl',
                        help="Omemory_size")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--latent_dim", type=int, default=2,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--no_eval", action="store_true",
                        help="Perform no evaluation of the prediction performance. "
                             "No label files loaded when this flag applies.")
    parser.add_argument("--eval_prob", type=float, default=1.0,
                        help="Perform evaluation on only a fraction of the"
                             " computed views (to speed up run-time). OBS: "
                             "always performs evaluation on the combined "
                             "predictions.")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--save_input_files", action="store_true",
                        help="Save in addition to the predicted volume the "
                             "input image and label files to the output dir)")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")
    parser.add_argument("--on_val", action="store_true",
                        help="Evaluate on the validation set instead of test")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--continues", action="store_true",
                        help="Continue from a previsous, non-finished "
                             "prediction session at 'out_dir'.")
    parser.add_argument("--limit_memory", action="store_false",
                        help="limit_memory")
    return parser

def entry_func(args=None):
    parser = get_argparser()
    parsed = parser.parse_args(args)
    print(parsed.args)
    print(parsed.out_dir)
    print(parsed.project_dir)
    print(parsed.no_argmax)
    print(parsed.num_GPUs)
    print(parsed.force_GPU)
    print(parsed.l)
    print(parsed.continues)

    kwargs ={}
    kwargs['set_memory'] = parsed.limit_memory
    kwargs['batch_size'] = parsed.batch_size
    kwargs['latent_dim'] = parsed.latent_dim

    print(kwargs)

    train(**kwargs)
if __name__ == "__main__":
    #sb = '--num_GPUs 3 --num_GPUs=3 --no_argmax'
    entry_func()
