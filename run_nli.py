import argparse 
import os 
import logging 
from attrdict import AttrDict 
from transformers import AutoTokenizer 
from transformers import AutoConfig 
from transformers import RobertaModel, BertModel
from src.functions.utils import init_logger, set_seed

from src.model.main_functions_baseline import train, evaluate, predict
from src.model.our_basic_models import baseline_model, CAGCNModel, AffinityDiffProposedModel

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

def create_model(args):

    # Load
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir, local_files_only=True,
        # force_download=True,
    )

    config.num_labels = args.num_labels
    
    config.output_attentions = True
    args.hidden_size = config.hidden_size

    # print(config)
    # print(config.hidden_dropout_prob)

    # BertTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir, local_files_only=True,
    )
    print(tokenizer)

    if "roberta" in args.output_dir:
        language_model = RobertaModel.from_pretrained(
            args.model_name_or_path,
            config = config, 
        )

    elif "bert" in args.output_dir:
        language_model = BertModel.from_pretrained(
            args.model_name_or_path,
            config = config, 
            cache_dir=args.cache_dir, local_files_only=True,
        )

    print("tokenizer.sep_token_id: ", tokenizer.sep_token_id)
    if "baseline" in args.output_dir:
        model = baseline_model(language_model=language_model, config=config)    
    else:
        model = AffinityDiffProposedModel( 
            language_model=language_model,
            config=config,
            max_prem_length=args.max_prem_length,
            max_hypo_length=args.max_hypo_length,
            device=args.device,
            graph_dim=args.d_embed, 
            gcn_dep=args.gcn_dep, gcn_layers=args.gcn_layers,
            n_graph_attn_composition_layers=args.gcn_layers,
            sep_token_id = tokenizer.sep_token_id,
        )
    if not args.from_init_weight: model.load_state_dict(torch.load(os.path.join(args.output_dir, "model/checkpoint-{}/pytorch.pt".format(args.checkpoint))))
    # print(model)

    model.to(args.device)

    return model, tokenizer

def main(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    args.add_label_node = True if "LabelNode" in args.output_dir else False
    args.change_label = True if "change_label" in args.output_dir else False
    args.change_label = True if "LabelMeaning" in args.output_dir else False
    args.change_label = False if "OriginLabelText" in args.output_dir else True

    args.add_interact_graphs = False if "NoInteractGraph" in args.output_dir else True
    args.give_and_take_option = True if "GiveAndTakeModel" in args.output_dir else False

    print("\n\n")
    print("args.add_label_node: "+str(args.add_label_node))
    print("args.change_label: "+str(args.change_label))
    print("args.gcn_layers: "+str(args.gcn_layers))
    print("args.add_interact_graphs: "+str(args.add_interact_graphs))
    print("args.percent_special_weight: " + str(args.percent_special_weight))
    print("args.give_and_take_option: " +str(args.give_and_take_option))

    print("args.label_relation: "+str(args.label_relation))

    logger = logging.getLogger(__name__)

    init_logger()
    set_seed(args)

    print("\n\n"+args.output_dir+"\n\n")
    # call model and tokenizer
    model, tokenizer = create_model(args)

    # Running model
    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        evaluate(args, model, tokenizer, logger, epoch_idx =args.checkpoint)
    elif args.do_predict:
        predict(args, model, tokenizer)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    # Directory
    cli_parser.add_argument("--data_dir", type=str, default="./data/snli/LALParser")
    cli_parser.add_argument("--train_file", type=str, default= 'train.jsonl')
    cli_parser.add_argument("--eval_file", type=str, default='dev.jsonl')
    cli_parser.add_argument("--predict_file", type=str, default='test.jsonl')

    cli_parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    cli_parser.add_argument("--cache_dir", type=str, default="roberta-base")

    # ===================================================================================================
    
    cli_parser.add_argument("--output_dir", type=str,
                            # default="./roberta-base/snli/baseline")
                            default="./roberta-base/snli/parameter/AffinityDiffProposedModel/mix5_mean/LabelNode_wo_Pair2Label_lr2e5")

    # ------------------------------------------------------------------------------------------------------------

    cli_parser.add_argument("--num_labels", type=int, default=3)

    cli_parser.add_argument("--max_seq_length", type=int, default=512)
    cli_parser.add_argument("--max_prem_length", type=int, default=270)
    cli_parser.add_argument("--max_hypo_length", type=int, default=190)
    cli_parser.add_argument("--separate_sentence_pair", type=bool, default= False)

    cli_parser.add_argument("--label_relation", type=int, default=51)  # origin_relation: 50 + 1(virtual syntactic connection)

    cli_parser.add_argument("--both_label_relation", type=bool, default= False) 
    cli_parser.add_argument("--percent_special_weight", type=float, default= 1.0)

    cli_parser.add_argument("--num_multi_layers", type=int, default= 3)
    cli_parser.add_argument("--d_embed", type=int, default=384) 
    cli_parser.add_argument("--gcn_dep", type=float, default=0.0)
    cli_parser.add_argument("--gcn_layers", type=int, default = 3)
    # ------------------------------------------------------------------------------------------------------------
    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=2e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=32)
    cli_parser.add_argument("--eval_batch_size", type=int, default= 32)
    cli_parser.add_argument("--num_train_epochs", type=int, default=5)

    cli_parser.add_argument("--logging_steps", type=int, default=100)
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.1)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    cli_parser.add_argument("--warmup_ratio", type=int, default=0.06)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # ------------------------------------------------------------------------------------------------------------

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default= False)
    cli_parser.add_argument("--checkpoint", type=str, default="4")

    cli_parser.add_argument("--do_train", type=bool, default = False)
    cli_parser.add_argument("--do_eval", type=bool, default= False)
    cli_parser.add_argument("--do_predict", type=bool, default= True)
    cli_parser.add_argument("--draw_tsne", type=bool, default= False)
    cli_parser.add_argument("--draw_tsne_name",  type=str, default="tsne_42.png")
    cli_parser.add_argument("--draw_attn_map", type=bool, default= False)
    cli_args = cli_parser.parse_args()

    main(cli_args)
