import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import random

from tqdm import tqdm

import transformers
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.data.processors.utils import DataProcessor

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

def convert_example_to_features(example, max_seq_length, is_training, max_prem_length, max_hypo_length, language):
    label = None
    if is_training:
        # Get label
        label = example.label

    input_ids_dict = {"premise": [], "hypothesis": []}
    pair_word_idxs = {"premise": [], "hypothesis": []}

    for case in example.doc_tokens.keys():
        # example.doc_tokens[case]: word list
        for word in example.doc_tokens[case]:
            start_idx = len(input_ids_dict[case])
            input_ids_dict[case] += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            pair_word_idxs[case].append([start_idx, len(input_ids_dict[case])-1])

    # =========================================================
    input_ids = [tokenizer.cls_token_id] + input_ids_dict["premise"]
    pair_word_idxs["premise"] = [[0, 0]] + [[pair[0]+1, pair[1]+1] for pair in pair_word_idxs["premise"]] #cls
    input_ids += [tokenizer.sep_token_id]
    if "roberta" in language or "camembert" in language: input_ids += [tokenizer.sep_token_id]
    token_type_ids = [0] * len(input_ids)

    pair_word_idxs["hypothesis"] = [[len(input_ids)-1, len(input_ids)-1]] + [[pair[0] + len(input_ids), pair[1] + len(input_ids)] for pair in pair_word_idxs["hypothesis"]] # sep

    input_ids += input_ids_dict["hypothesis"] + [tokenizer.sep_token_id]
    pair_word_idxs["hypothesis"] += [[len(input_ids)-1, len(input_ids)-1]] #sep

    token_type_ids = token_type_ids + [1] * (len(input_ids) - len(token_type_ids))
    position_ids = list(range(0, len(input_ids)))
    attention_mask = [1] * len(input_ids)

    assert len(input_ids) == len(attention_mask) == len(token_type_ids) == len(position_ids)
    if len(input_ids) > max_seq_length: print(len(input_ids), example.doc_tokens, example.guid)
    # padding #####################################
    paddings = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
    input_ids += paddings
    attention_mask += [0]*len(paddings)
    token_type_ids += paddings
    position_ids += paddings

    real_prem_len = len(pair_word_idxs["premise"])
    real_hypo_len = len(pair_word_idxs["hypothesis"])

    if max_prem_length < len(pair_word_idxs["premise"]): print(len(pair_word_idxs["premise"]), "prem")
    if max_hypo_length < len(pair_word_idxs["hypothesis"]): print(len(pair_word_idxs["hypothesis"]), "hypo")

    pair_word_idxs["premise"] += [[-1, -1]]*(max_prem_length - len(pair_word_idxs["premise"]))
    pair_word_idxs["hypothesis"] += [[-1, -1]] * (max_hypo_length - len(pair_word_idxs["hypothesis"]))
    example.dependency["premise"] += [[-1, -1, 0]] * (max_prem_length - len(example.dependency["premise"]))
    example.dependency["hypothesis"] += [[-1, -1, 0]] * (max_hypo_length - len(example.dependency["hypothesis"]))


    if max(sum([parse[:2] for parse in example.dependency["premise"] if parse[0] != -1], [])) != (len(example.doc_tokens["premise"])-1): 
        print(max(sum([parse[:2] for parse in example.dependency["premise"] if parse[0] != -1], [])) , (len(example.doc_tokens["premise"])-1))
        print(example.dependency["premise"], example.doc_tokens["premise"])
        assert 1 ==2
    if max(sum([parse[:2] for parse in example.dependency["hypothesis"] if parse[0] != -1], [])) != (len(example.doc_tokens["hypothesis"])-1): 
        print(example.dependency["hypothesis"], example.doc_tokens["hypothesis"])
        assert 2 ==3

    ########################################################################
    # if len(input_ids) > max_seq_length:
    #     print(example.doc_tokens)
    #     print(input_ids_dict)


    return  NLIFeatures(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,

            label = label,
            prem_word_idxs= pair_word_idxs["premise"],
            hypo_word_idxs= pair_word_idxs["hypothesis"],
            real_prem_len = real_prem_len,
            real_hypo_len=real_hypo_len,

            prem_dependency = example.dependency["premise"],
            hypo_dependency = example.dependency["hypothesis"],
    )

def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        is_training,
        return_dataset=False,
        threads=1,
        max_prem_length=0,
        max_hypo_length=0,
        tqdm_enabled=True,
        language = None,
        separate_sentence_pair=False,
):
    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:

        # annotate_ : A list of multiple features for one example
        # annotate_ = list(feature1, feature2, ...)
        annotate_ = partial(
            convert_example_to_features,
            max_seq_length=max_seq_length,
            max_prem_length =max_prem_length,
            max_hypo_length = max_hypo_length,
            is_training=is_training,
            language = language,
        )

        # features = list( feature1, feature2, feature3, ... )
        ## len(features) == len(examples)
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=16),
                total=len(examples),
                desc="convert klue nli examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    example_index = 0  ## len(features) == len(examples)
    for example_feature in tqdm(
            features, total=len(features), desc="add example index", disable=not tqdm_enabled
    ):
        if not example_feature:
            continue

        example_feature.example_index = example_index
        new_features.append(example_feature)
        example_index += 1

    features = new_features
    del new_features

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)

        all_prem_word_idxs = torch.tensor([f.prem_word_idxs for f in features], dtype=torch.long)
        all_hypo_word_idxs = torch.tensor([f.hypo_word_idxs for f in features], dtype=torch.long)

        all_real_prem_len = torch.tensor([f.real_prem_len for f in features], dtype=torch.long)
        all_real_hypo_len = torch.tensor([f.real_hypo_len for f in features], dtype=torch.long)

        all_prem_dependency = torch.tensor([f.prem_dependency for f in features], dtype=torch.long)
        all_hypo_dependency = torch.tensor([f.hypo_dependency for f in features], dtype=torch.long)

        if not is_training:
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_position_ids,
                all_prem_word_idxs, all_hypo_word_idxs, all_real_prem_len, all_real_hypo_len, 
                all_prem_dependency, all_hypo_dependency,
            )

        else:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_position_ids,
                all_labels,
                all_prem_word_idxs, all_hypo_word_idxs, all_real_prem_len, all_real_hypo_len, 
                all_prem_dependency, all_hypo_dependency,

            )

        return features, dataset
    else:
        return features


class NLIProcessor(DataProcessor):
    train_file = None
    dev_file = None

    def get_train_examples(self, data_dir, filename=None, depend_embedding = None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("NLIProcessor should be instantiated via NLIV1Processor.")

        with open(os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8") as f:
            input_data = [json.loads(d) for d in f]
            input_data = [d for d in input_data if d["gold_label" if "gold_label" in d.keys() else "label"] not in ["-", "_"]]
            #input_data = input_data[0:5]

        return self._create_examples(input_data, 'train', self.train_file if filename is None else filename)

    def get_dev_examples(self, data_dir, filename=None, depend_embedding = None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("NLIProcessor should be instantiated via NLIV1Processor.")

        with open(os.path.join(data_dir, self.dev_file if filename is None else filename), "r",
                  encoding="utf-8") as f:
            input_data = [json.loads(d) for d in f]
            input_data = [d for d in input_data if d["gold_label" if "gold_label" in d.keys() else "label"] not in ["-", "_"]]

        return self._create_examples(input_data, "dev", self.dev_file if filename is None else filename)

    def get_example_from_input(self, input_dictionary):
        # guid, premise, hypothesis
        guid=input_dictionary["id"]
        premise=input_dictionary["sentence1"]
        hypothesis=input_dictionary["sentence2"]
        gold_label=None
        label = None


        examples = [NLIExample(
            guid=guid,
            premise=premise,
            hypothesis=hypothesis,
            gold_label=gold_label,
            label=label,
        )]
        return examples

    def _create_examples(self, input_data, set_type, data_file):
        is_training = set_type == "train"
        examples = [] 

        for entry in tqdm(input_data):
            guid = entry["id"]
            premise = entry["premise"]
            hypothesis = entry["hypothesis"]
            gold_label = None
            label = None
            genre = None

            if "genre" in entry.keys(): genre = entry["genre"]
            if "heuristic" in entry.keys(): genre = entry["heuristic"]

            if is_training:
                label = entry["gold_label"].strip()
            else:
                gold_label = entry["gold_label"].strip()

            example = NLIExample(
                guid=guid,
                premise=premise,
                hypothesis=hypothesis,
                gold_label=gold_label,
                label=label,
                genre = genre, 
                data_file=data_file,
            )
            examples.append(example)
        # len(examples) == len(input_data)
        return examples


class NLIV1Processor(NLIProcessor):
    train_file = "train.jsonl"
    dev_file = "dev.jsonl"
    test_file = "test.jsonl"


class NLIExample(object):
    def __init__(
        self,
        guid,
        premise,
        hypothesis,
        gold_label=None,
        label=None,
        genre=None, 
        data_file=None,
    ):
        guid = guid
        premise = premise
        hypothesis = hypothesis
        gold_label = gold_label
        label = label

        self.guid = guid.strip()
        if genre != None: self.genre = genre.strip()
        else: self.genre = genre
        self.premise = premise["origin"].strip()
        self.hypothesis=hypothesis["origin"].strip()

        self.premise_root = premise["root"][1] if premise["root"] != None else max([p[2][1] for p in premise["parsing"]], key=[p[2][1] for p in premise["parsing"]].count)
        self.hypothesis_root= hypothesis["root"][1] if hypothesis["root"] != None else max([p[2][1] for p in hypothesis["parsing"]], key=[p[2][1] for p in hypothesis["parsing"]].count)


        # label_dict = {"neutral": 0, "contradiction":1, "entailment": 2} #snli SICK
        # label_dict = {"not_entailment": 0, "entailment": 1} # RTE
        # label_dict = {"neutral": 0, "entails": 1} # SciTail
        label_dict = {"non-entailment": 0, "entailment": 1} # hans trans_Overlap

        if gold_label in label_dict.keys():
            gold_label = label_dict[gold_label]
        elif gold_label != None: print(gold_label)
        self.gold_label = gold_label

        if label in label_dict.keys():
            label = label_dict[label]
        elif label != None: print(label)
        self.label = label

        self.doc_tokens = {"premise": [word.replace('-LRB-', "(").replace('-RRB-', ")").replace('-LSB-', "[").replace('-RSB-', "]") for word in premise["word_list"]],
                           "hypothesis": [word.replace('-LRB-', "(").replace('-RRB-', ")").replace('-LSB-', "[").replace('-RSB-', "]") for word in hypothesis["word_list"]]}

        tag_list = ['advcl', 'pcomp', 'dobj', 'preconj', 'neg', 'cop', 'det', 'partmod', 'goeswith', 'csubj', 'auxpass',
                    'parataxis', 'conj', 'abbrev', 'nsubjpass', 'prt', 'prep', 'poss', 'predet', 'num', 'iobj', 'ccomp',
                    'xcomp', 'mwe', 'nsubj', 'punct', 'amod', 'expl', 'root', 'pobj', 'prepc', 'mark', 'advmod',
                    'discourse', 'vmod', 'acomp', 'number', 'aux', 'infmod', 'dep', 'appos', 'tmod', 'nn', 'npadvmod',
                    'csubjpass', 'rcmod', 'purpcl', 'agent', 'cc', 'possessive', 'quantmod',  
                    ]

        depend2idx = {}
        idx2depend = {}
        for depend in tag_list:
            depend2idx[depend] = len(depend2idx)
            idx2depend[len(idx2depend)] = depend

        self.dependency = {"premise": [], "hypothesis": []}

        if len(premise["parsing"]) == 0:
            self.dependency["premise"] = [[premise["root"][1], premise["root"][1], depend2idx["root"]]]
        else:
            for parsing in premise["parsing"]:
                if parsing[2][1] == -1: parsing[2][1] = len(premise["word_list"]) - 1
                self.dependency["premise"].append([parsing[2][0], parsing[2][1], depend2idx[parsing[1][0]]])
        if len(hypothesis["parsing"]) == 0:
            self.dependency["hypothesis"] = [[hypothesis["root"][1], hypothesis["root"][1], depend2idx["root"]]]
        else:
            for parsing in hypothesis["parsing"]:
                if parsing[2][1] == -1: parsing[2][1] = len(hypothesis["word_list"]) - 1
                self.dependency["hypothesis"].append([parsing[2][0], parsing[2][1], depend2idx[parsing[1][0]]])

        if len(set(sum([parse[:2] for parse in self.dependency["premise"]], []))) != len(self.doc_tokens["premise"]): print(self.dependency["premise"], self.doc_tokens["premise"], self.premise)
        if len(set(sum([parse[:2] for parse in self.dependency["hypothesis"]], []))) != len(self.doc_tokens["hypothesis"]): print(self.dependency["hypothesis"], self.doc_tokens["hypothesis"], self.hypothesis)

class NLIFeatures(object):
    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            label,
            prem_word_idxs,# (batch, max_pair_sen, 2)  # token-based
            hypo_word_idxs,# (batch, max_pair_sen, 2)  # token-based
            real_prem_len,
            real_hypo_len,
            prem_dependency, hypo_dependency,

    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids

        self.label = label

        self.prem_word_idxs = prem_word_idxs
        self.hypo_word_idxs = hypo_word_idxs

        self.real_prem_len = real_prem_len
        self.real_hypo_len = real_hypo_len

        self.prem_dependency = prem_dependency
        self.hypo_dependency = hypo_dependency

class Result(object):
    def __init__(self, example_index, label_logits, gold_label=None, cls_logits=None):
        self.label_logits = label_logits
        self.example_index = example_index

        if gold_label:
            self.gold_label = gold_label
            self.cls_logits = cls_logits
