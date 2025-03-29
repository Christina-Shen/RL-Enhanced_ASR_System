#-------------------train tokenizer/ prepare json file -------------------
import sys

#from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import *
#from SentencePiece import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import yaml
import importlib
import speechbrain as sb


from hyperpyyaml import load_hyperpyyaml
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        prepare_mini_librispeech(
            data_folder=hparams["data_folder"],
            save_json_train=hparams["train_annotation"],
            save_json_valid=hparams["valid_annotation"],
            save_json_test=hparams["test_annotation"],
        )

    # Train tokenizer
    hparams["tokenizer"]()
# def train_tokenizer(yaml_file):

#     if not hparams["skip_prep"]:
#         prepare_mini_librispeech(
#             data_folder=hparams["data_folder"],
#             save_json_train=hparams["train_annotation"],
#             save_json_valid=hparams["valid_annotation"],
#             save_json_test=hparams["test_annotation"],
#         )

#     # Train tokenizer
#     SentencePiece(model_dir=hparams['output_folder'],   vocab_size=hparams['token_output'],annotation_train=hparams['train_annotation'],annotation_read=hparams ['annotation_read'],model_type=hparams['token_type'] # ["unigram", "bpe", "char"]
#    ,character_coverage=hparams['character_coverage'],annotation_list_to_check=[hparams['train_annotation'], hparams['valid_annotation']],annotation_format='json') #
#     return None

# yaml_file_path = './training_yaml_file/token_yaml.yaml'
# def loadfile(filepath):
#     try:
#         with open(filepath, 'r') as file:
#             data = yaml.safe_load(file)
#         return data
#     except Exception as e:
#         print(f"Error loading YAML file: {e}")
#         return None

# token_hyper=loadfile(yaml_file_path)
# train_tokenizer(token_hyper)
# print("prepare json,train token finished!")

