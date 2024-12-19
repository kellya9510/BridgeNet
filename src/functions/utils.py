import logging
import random
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

from src.functions.processor import (
    NLIV1Processor,
    convert_examples_to_features
)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_sklearn_score(predicts, corrects, idx2label):
    predicts = [idx2label[predict] for predict in predicts]
    corrects = [idx2label[correct] for correct in corrects]
    result = {"accuracy": accuracy_score(corrects, predicts),
              "macro_precision": precision_score(corrects, predicts, average="macro"),
              "micro_precision": precision_score(corrects, predicts, average="micro"),
              "macro_f1": f1_score(corrects, predicts, average="macro"),
              "micro_f1": f1_score(corrects, predicts, average="micro"),
              "macro_recall": recall_score(corrects, predicts, average="macro"),
              "micro_recall": recall_score(corrects, predicts, average="micro"),
              }

    if len(idx2label.keys()) == 4:label_list = ["contrasting", "reasoning", "entailment", "neutral"]
    elif len(idx2label.keys()) == 3:label_list = ["contradiction", "entailment", "neutral"]
    elif len(idx2label.keys()) == 2:label_list = ["entailment", "non-entailment"]

    #print(idx2label, corrects, predicts)
    print(classification_report(corrects, predicts, target_names= label_list, digits=4))

    for k, v in result.items():
        # result[k] = round(v, 6)
        print(k + ": " + str(v))
    return result

def load_examples(args, tokenizer, evaluate=False, output_examples=False, do_predict=False, input_dict=None):
    '''
    :param args: Hyperparameters
    :param tokenizer: tokenizer used for tokenization
    :param evaluate: True when evaluating or open testing
    :param output_examples: True when evaluating or open testing, return examples and features together if True
    :param do_predict: True when open testing
    :param input_dict: dictionary of documents and questions to be input on open test
    :return:
    examples : A list of each data in full text regardless of max_length
    features : a list of partitioned and tokenized texts according to max_length
    dataset : input ids converted into tensors that are directly used for training and testing
    '''
    input_dir = args.data_dir
    print("Creating features from dataset file at {}".format(input_dir))

    processor = NLIV1Processor()

    if do_predict:
        examples = processor.get_dev_examples(os.path.join(args.data_dir),
                                              filename=args.predict_file)
    elif evaluate:
        examples = processor.get_dev_examples(os.path.join(args.data_dir),
                                              filename=args.eval_file)
    # for training
    else:
        examples = processor.get_train_examples(os.path.join(args.data_dir),
                                                filename=args.train_file)

    features, dataset = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
        max_prem_length=args.max_prem_length,
        max_hypo_length=args.max_hypo_length,
        language=args.model_name_or_path if len(args.model_name_or_path.split("/")) == 1 else args.model_name_or_path.split("/")[-2],
    )

    ## example == feature == dataset
    return dataset, examples, features
