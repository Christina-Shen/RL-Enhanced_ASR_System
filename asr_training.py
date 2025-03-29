#-------------------training ASR system-----------------------------------


import logging
import sys

import torch

from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from DQN_agent import *
from augmentor_agent import *


import scipy.stats
from scipy.signal import stft
import numpy as np
import scipy.stats
from scipy.signal import stft

from types import SimpleNamespace
import os


logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class ASR(sb.Brain):
    def __init__(  # noqa: C901
        self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None, n_wavaug=4,n_feataug=4
    ):  
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        #----------------------------------------------------------------
        self.n_wavaug=n_wavaug
        self.n_feataug=n_feataug
        self.wav_agent=DQNAgent(state_size=256,action_size=10,n_action=4)
        self.feat_agent=DQNAgent(state_size=256,action_size=6,n_action=4)
        self.wav_rlaugment=rl_Augmenter(num_augmentations=4)
        self.feat_rlaugment=rl_Augmenter(num_augmentations=4)
        # if hparams is not None:
        #     self.hparams = SimpleNamespace(**hparams)
        self.distributed_launch = (
            os.environ.get("RANK") is not None
            and os.environ.get("LOCAL_RANK") is not None
        )
        self.wavs=None
        self.wavs_trans=None
        self.feats=None
        self.feats_trans=None
        self.wavs_action=None
        self.feats_action=None
        self.test_only=False
        self.stage_loss=0
        self.wav_instant_reward=0
        self.feat_instant_reward=0
        #-------------------------------------------------------------------
        # act/step(get memory/ learn)
    def compute_forward(self, batch, stage):
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        self.wavs,self.wavs_trans,self.feats,feats, self.feat_lens = self.prepare_features(stage, batch.sig)
        self.feats_trans=feats

        tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

        # Running the encoder (prevent propagation to feature extraction)
        encoded_signal = self.modules.encoder(feats.detach())

        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos.detach())
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, self.feat_lens
        )

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}

        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_signal)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)

        elif stage != sb.Stage.TRAIN:
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    encoded_signal, self.feat_lens
                )
            elif stage == sb.Stage.TEST:
                hyps, _, _, _ = self.hparams.test_search(
                    encoded_signal, self.feat_lens
                )

            predictions["tokens"] = hyps
        self.add_memory_learn()

        return predictions

    def is_ctc_active(self, stage):

        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def compute_wav_state(wavs, sr=16000, n_fft=512, hop_length=256):
        def time_domain_features(wav):
            features = [
                np.mean(wav),
                np.std(wav),
                np.max(wav),
                np.min(wav),
                scipy.stats.kurtosis(wav),
                scipy.stats.skew(wav),
                np.sqrt(np.mean(wav**2)),
                np.sum(np.diff(np.sign(wav)) != 0) / len(wav)
            ]
            return features

        def freq_domain_features(wav):
            _, _, Zxx = stft(wav, nperseg=n_fft, noverlap=n_fft - hop_length)
            magnitude_spectrogram = np.abs(Zxx)
            
            spectral_mean = np.mean(magnitude_spectrogram, axis=1)
            spectral_std = np.std(magnitude_spectrogram, axis=1)
            spectral_centroid = np.sum(np.arange(n_fft//2 + 1)[:, np.newaxis] * magnitude_spectrogram, axis=0) / np.sum(magnitude_spectrogram, axis=0)
            spectral_bandwidth = np.sqrt(np.sum((np.arange(n_fft//2 + 1)[:, np.newaxis] - spectral_centroid)**2 * magnitude_spectrogram, axis=0) / np.sum(magnitude_spectrogram, axis=0))
            spectral_kurtosis = scipy.stats.kurtosis(magnitude_spectrogram, axis=1)
            spectral_skewness = scipy.stats.skew(magnitude_spectrogram, axis=1)
            spectral_flatness = scipy.stats.gmean(magnitude_spectrogram, axis=1) / np.mean(magnitude_spectrogram, axis=1)
            spectral_entropy = -np.sum(magnitude_spectrogram * np.log(magnitude_spectrogram + 1e-10), axis=1)
            spectral_energy = np.sum(magnitude_spectrogram**2, axis=1)
            spectral_power = np.mean(magnitude_spectrogram**2, axis=1)
            
            features = np.concatenate([
                spectral_mean, spectral_std, spectral_centroid, spectral_bandwidth,
                spectral_kurtosis, spectral_skewness, spectral_flatness, spectral_entropy,
                spectral_energy, spectral_power
            ])
            return features

        all_features = []
        for wav in wavs:
            time_features = time_domain_features(wav)
            freq_features = freq_domain_features(wav)
            features = np.concatenate([time_features, freq_features])
            all_features.append(features)
        
        all_features = np.array(all_features)
        return all_features


    def compute_instant_reward(self,matrix_1,matrix_2):
        snr_diff = (np.var(matrix_2)/np.mean(matrix_2) ** 2) -(np.var(matrix_1)/np.mean(matrix_1) ** 2)
        var_diff = np.var(matrix_2)-np.var(matrix_1)
        reward=0.5*snr_diff+0.5*var_diff
        return reward
    def prepare_features(self, stage, wavs):
        wavs, wav_lens = wavs
        wav_states=self.compute_wav_state(wavs)
        print(wav_states.shape)
        # Add waveform augme ntation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            self.wav_action=self.wav_agent.act(wav_states)
            wav_augments = [self.hparams.wav_augmentations[i] for i in self.wav_action]
            wavs_new, wav_lens = self.wav_rlaugment(wavs, wav_lens,wav_augments)
        # Feature computation and normalization
        self.wav_instant_reward=self.compute_instant_reward(wavs)
        wav_states_trans=self.compute_wav_state(wavs)
        fea_lens = wav_lens  # Relative lengths are preserved
        # Add feature augmentation if specified.

        feats = self.hparams.compute_features(wavs)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            self.feats_action=self.feat_agent.act(feats)
            feat_augments = [self.hparams.feat_augmentations[i] for i in self.feats_action]
            feats_aug, fea_lens = self.feat_rlaugment(feats, fea_lens,feat_augments)
        feats_aug = self.modules.normalize(feats_aug, fea_lens)

        self.feat_instant_reward=self.compute_instant_reward(feats_aug)
        return wav_states,wav_states_trans,feats,feats_aug ,fea_lens

    def prepare_tokens(self, stage, tokens):
        tokens, token_lens = tokens
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                tokens = self.hparams.wav_augment.replicate_labels(tokens)
                token_lens = self.hparams.wav_augment.replicate_labels(
                    token_lens
                )
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                token_lens = self.hparams.fea_augment.replicate_labels(
                    token_lens
                )
        return tokens, token_lens

    def compute_objectives(self, predictions, batch, stage):
        # Compute sequence loss against targets with EOS
        tokens_eos, tokens_eos_lens = self.prepare_tokens(
            stage, batch.tokens_eos
        )
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Add ctc loss if necessary. The total cost is a weighted sum of
        # ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
            loss_ctc = self.hparams.ctc_cost(
                predictions["ctc_logprobs"], tokens, self.feat_lens, tokens_lens
            )
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc
        self.stage_loss=loss

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.words]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.test_wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def compute_reward(self,stage_loss):
        cer=self.cer_metric.summarize("error_rate")
        wer=self.wer_metric.summarize("error_rate")
        reward=-(0.1*stage_loss+0.5*cer+0.5*wer)+(0.5*self.feat_instant_reward+0.5*self.wav_instant_reward)
        return reward
    def add_memory_learn(self):
        reward=self.compute_reward(self.stage_loss)
        wav_ratio=(self.wav_instant_reward)/(self.feat_instant_reward+self.wav_instant_reward)
        feat_ratio=(self.feat_instant_reward)/(self.feat_instant_reward+self.wav_instant_reward)
        self.wav_rlaugment.step(self.wavs,self.wavs_action,reward*wav_ratio,self.wavs_trans)
        self.feat_rlaugment.step(self.feats,self.feats_action,reward*feat_ratio,self.feats_trans)

def dataio_prepare(hparams):
    # global audio_pipelines, text_pipelines
    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    # 將 audio_pipeline 函數移到頂層
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipelines(wav):
        
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        print("wave:", type(wav))
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # # 將 text_pipeline 函數移到頂層
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipelines(words):
        """Processes the transcriptions to generate proper labels"""

        yield words
        tokens_list = hparams["tokenizer"].encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    datasets = {}
    data_folder = hparams["data_folder"]
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipelines, text_pipelines],
            output_keys=[
                "id",
                "sig",
                "words",
                "tokens_bos",
                "tokens_eos",
                "tokens",
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting training data with ascending order makes the code much
    # faster because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = [datasets["train"].filtered_sorted(sort_key="length")]
        datasets["valid"] = [datasets["valid"].filtered_sorted(sort_key="length")]
        datasets["test"] = [datasets["test"].filtered_sorted(sort_key="length")]
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=True
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="length", reverse=True
        )
        datasets["test"] = datasets["test"].filtered_sorted(
            sort_key="length", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    return datasets

# def dataio_prepare(hparams):
#     # Define audio pipeline. In this case, we simply read the path contained
#     # in the variable wav with the audio reader.
#     @sb.utils.data_pipeline.takes("wav")
#     @sb.utils.data_pipeline.provides("sig")
#     def audio_pipeline(wav):
#         """Load the audio signal. This is done on the CPU in the `collate_fn`."""
#         sig = sb.dataio.dataio.read_audio(wav)
#         return sig

#     # Define text processing pipeline. We start from the raw text and then
#     # encode it using the tokenizer. The tokens with BOS are used for feeding
#     # decoder during training, the tokens with EOS for computing the cost function.
#     # The tokens without BOS or EOS is for computing CTC loss.
#     @sb.utils.data_pipeline.takes("words")
#     @sb.utils.data_pipeline.provides(
#         "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
#     )
#     def text_pipeline(words):
#         """Processes the transcriptions to generate proper labels"""
#         yield words
#         tokens_list = hparams["tokenizer"].encode_as_ids(words)
#         yield tokens_list
#         tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
#         yield tokens_bos
#         tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
#         yield tokens_eos
#         tokens = torch.LongTensor(tokens_list)
#         yield tokens

#     # Define datasets from json data manifest file
#     # Define datasets sorted by ascending lengths for efficiency
#     datasets = {}
#     data_folder = hparams["data_folder"]
#     data_info = {
#         "train": hparams["train_annotation"],
#         "valid": hparams["valid_annotation"],
#         "test": hparams["test_annotation"],
#     }

#     for dataset in data_info:
#         datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
#             json_path=data_info[dataset],
#             replacements={"data_root": data_folder},
#             dynamic_items=[audio_pipeline, text_pipeline],
#             output_keys=[
#                 "id",
#                 "sig",
#                 "words",
#                 "tokens_bos",
#                 "tokens_eos",
#                 "tokens",
#             ],
#         )
#         hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

#     # Sorting training data with ascending order makes the code  much
#     # faster  because we minimize zero-padding. In most of the cases, this
#     # does not harm the performance.
#     if hparams["sorting"] == "ascending":
#         datasets["train"] = datasets["train"].filtered_sorted(sort_key="length")
#         datasets["valid"] = datasets["valid"].filtered_sorted(sort_key="length")
#         datasets["test"] = datasets["test"].filtered_sorted(sort_key="length")
#         hparams["train_dataloader_opts"]["shuffle"] = False

#     elif hparams["sorting"] == "descending":
#         datasets["train"] = datasets["train"].filtered_sorted(
#             sort_key="length", reverse=True
#         )
#         datasets["valid"] = datasets["valid"].filtered_sorted(
#             sort_key="length", reverse=True
#         )
#         datasets["test"] = datasets["test"].filtered_sorted(
#             sort_key="length", reverse=True
#         )
#         hparams["train_dataloader_opts"]["shuffle"] = False

#     elif hparams["sorting"] == "random":
#         hparams["train_dataloader_opts"]["shuffle"] = True
#         pass

#     else:
#         raise NotImplementedError(
#             "sorting must be random, ascending or descending"
#         )
#     return datasets

# def train_asr(yaml_file):
 # Reading command line arguments
if __name__ == "__main__":
    print("---------------training asr start--------------------")
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
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
        sb.utils.distributed.run_on_main(
            prepare_mini_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
            },
        )
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    test_stats = asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    asr_brain.checkpointer.save_checkpoint(name="latest")
    print("---------------training asr finish-------------------")

# training asr
# #read yaml file
# 
# yaml_file_path = './training_yaml_file/asr_yaml.yaml'
# asr_hyper = load_hyperpyyaml(yaml_file_path)
# train_asr(yaml_file_path)

# print("---------------training asr finish-------------------")


