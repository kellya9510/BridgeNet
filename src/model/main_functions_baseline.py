import os
import numpy as np
import pandas as pd
import torch
import timeit
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
)

from src.functions.utils import load_examples, set_seed, to_list, get_sklearn_score

from dgl import DGLGraph
def get_graph(dependency, add_label_node=False, give_and_take_option = False, 
              label_relation = 32, both_label_relation = False, percent_special_weight = 1.0,  device="cpu"):
    # [word-unit subordinate, word-unit dominant, syntactic structure tag index]
    # prem_dependency  = [[premise_tail, premise_head, dependency], [], ...]
    # hypo_dependency = [[hypothesis_tail, hypothesis_head, dependency], [], ...]]

    dependency["premise"] = to_list(dependency["premise"])
    dependency["hypothesis"] = to_list(dependency["hypothesis"])

    # premise
    prem_g = DGLGraph()
    dependency["premise"] = [span for span in dependency["premise"] if span[0] != -1]
    head = [span[1] for span in dependency["premise"]] + [span[0] for span in dependency["premise"]]
    tail = [span[0] for span in dependency["premise"]] + [span[1] for span in dependency["premise"]]
    if label_relation != 51: tag = [span[2] for span in dependency["premise"]] + [span[2] + 51 for span in dependency["premise"]]  # _inv
    else: tag = [span[2] for span in dependency["premise"]] + [span[2] + 52 for span in dependency["premise"]]  # _inv

    if add_label_node:
        if percent_special_weight != 1.0:
            special_weight = [1.0] * len(head)
        if both_label_relation:
            root_idx = [h for h, t in zip(head, tail) if h == t]
            root_idx = root_idx[0] if len(root_idx) != 0 else max([span[1] for span in dependency["premise"]],
                                                                  key=[span[1] for span in dependency["premise"]].count)
            link_idx = max(tail + head) + 1

            head = head + [root_idx] + [link_idx]
            tail = tail + [link_idx] + [root_idx]
            if percent_special_weight != 1.0: special_weight += [percent_special_weight] + [percent_special_weight]

            if label_relation != 51:
                tag += [label_relation, label_relation + 51]
            else:
                tag += [label_relation, label_relation + 52]


    num_node = max(tail+ head) + 1
    node_ids = torch.arange(0, num_node, dtype=torch.long)
    edge_type = torch.tensor(tag)

    _, inverse_index, count = np.unique((head, tag), axis=1, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    edge_norm = np.ones(len(head), dtype=np.float32) / degrees.astype(np.float32)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

    prem_g.add_nodes(num_node)
    prem_g.add_edges(tail, head)

    prem_g = prem_g.to(device)
    prem_g.ndata.update({'id': node_ids.to(device)})
    prem_g.edata.update({'type': edge_type.to(device), 'norm': edge_norm.to(device)})
    if add_label_node:
        if percent_special_weight != 1.0:
            prem_g.edata["special_weight"] = torch.tensor(special_weight).unsqueeze(1).to(device)

    # hypothesis
    hypo_g = DGLGraph()
    dependency["hypothesis"] = [span for span in dependency["hypothesis"] if span[0] != -1]

    head = [span[1] for span in dependency["hypothesis"]] + [span[0] for span in dependency["hypothesis"]]
    tail = [span[0] for span in dependency["hypothesis"]] + [span[1] for span in dependency["hypothesis"]]
    if label_relation != 51: tag = [span[2] for span in dependency["hypothesis"]] +  [span[2]+ 51 for span in dependency["hypothesis"]] # _inv
    else: tag = [span[2] for span in dependency["hypothesis"]] +  [span[2]+ 52 for span in dependency["hypothesis"]] # _inv

    # add label info
    if add_label_node:
        root_idx = [h for h,t in zip(head, tail) if h == t]
        root_idx = root_idx[0] if len(root_idx) != 0 else max([span[1] for span in dependency["hypothesis"]], key=[span[1] for span in dependency["hypothesis"]].count)
        if percent_special_weight != 1.0:
            special_weight = [1.0] * len(head) + [percent_special_weight] + [percent_special_weight]
        head = [h + 1 for h in head] + [root_idx] + [0]
        tail = [t + 1 for t in tail] + [0] + [root_idx]
        if label_relation != 51: tag += [label_relation, label_relation + 51]
        else: tag += [label_relation, label_relation + 52]

    num_node = max(tail+ head) + 1
    node_ids = torch.arange(0, num_node, dtype=torch.long)
    edge_type = torch.tensor(tag)
    _, inverse_index, count = np.unique((head, tag), axis=1, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    edge_norm = np.ones(len(head), dtype=np.float32) / degrees.astype(np.float32)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

    hypo_g.add_nodes(num_node)
    hypo_g.add_edges(tail, head)

    hypo_g = hypo_g.to(device)
    hypo_g.ndata.update({'id': node_ids.to(device)})
    hypo_g.edata.update({'type': edge_type.to(device), 'norm': edge_norm.to(device)})
    if add_label_node:
        if percent_special_weight != 1.0:
            hypo_g.edata["special_weight"] = torch.tensor(special_weight).unsqueeze(1).to(device)

    return (prem_g, hypo_g, )

                                    
def train(args, model, tokenizer, logger):
    max_acc = 0.6

    train_dataset, _, _ = load_examples(args, tokenizer, evaluate=False, output_examples=False)

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # t_total: total optimization step
    # Calculate the total training steps for the optimization schedule
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Apply weighted decay based on Layer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    num_warmup_steps = min(int(t_total * args.warmup_ratio), 10000)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    if not args.from_init_weight: global_step += int(args.checkpoint)

    tr_loss, current_iter = 0.0, 0

    # loss buffer initialization
    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    set_seed(args)

    epoch_idx = 0
    if not args.from_init_weight: epoch_idx += int(args.checkpoint)

    for epoch in mb:

        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs_list = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                           "prem_word_idxs", "hypo_word_idxs", "real_prem_len", "real_hypo_len",
                           ]

            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            if not "CAGCN" in args.output_dir:
                prem_graphs = []
                hypo_graphs = []
                for prem_dep, hypo_dep in zip(batch[-2], batch[-1]):
                    dependency = {"premise": prem_dep, "hypothesis":hypo_dep}
                    g = get_graph(dependency, device=args.device, add_label_node=args.add_label_node,
                                label_relation = args.label_relation,
                                both_label_relation = args.both_label_relation,
                                percent_special_weight=args.percent_special_weight,
                                give_and_take_option = args.give_and_take_option
                                )
                    prem_graphs.append(g[0])
                    hypo_graphs.append(g[1])
                inputs["prem_graphs"] = prem_graphs
                inputs["hypo_graphs"] = hypo_graphs
            else: 
                inputs["prem_span"]=batch[-2]
                inputs["hypo_span"]=batch[-1]
                
            outputs = model(**inputs)
            loss = outputs[0]
            # print(loss)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            current_iter += 1

            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step + 1), loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            # torch.cuda.empty_cache()

        epoch_idx += 1
        logger.info("***** Eval results *****")
        results = evaluate(args, model, tokenizer, logger, epoch_idx=str(epoch_idx), tr_loss=loss.item())

        if (2 < epoch_idx) and (float(results["accuracy"]) >= max_acc):
            output_dir = os.path.join(args.output_dir, "model/checkpoint-{}".format(epoch_idx))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            max_acc = results["accuracy"]
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch.pt"))
        # logger.info("Saving model checkpoint to %s", os.path.join(output_dir, "pytorch.pt"))

        mb.write("Epoch {} done".format(epoch + 1))
        
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, logger, epoch_idx = "", tr_loss = 1):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    step = -1

    # Eval!
    logger.info("***** Running evaluation {} *****".format(epoch_idx))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()

    pred_logits = torch.tensor([], dtype = torch.long).to(args.device)

    for batch in progress_bar(eval_dataloader):
        model.eval()
        step += 1
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs_list = ["input_ids", "attention_mask", "token_type_ids", "position_ids",
                           "prem_word_idxs", "hypo_word_idxs", "real_prem_len", "real_hypo_len",
                           ]

            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            if not "CAGCN" in args.output_dir:
                prem_graphs = []
                hypo_graphs = []
                for prem_dep, hypo_dep in zip(batch[-2], batch[-1]):
                    dependency = {"premise": prem_dep, "hypothesis":hypo_dep}
                    g = get_graph(dependency, device=args.device, add_label_node=args.add_label_node,
                                label_relation = args.label_relation,
                                both_label_relation = args.both_label_relation,
                                percent_special_weight=args.percent_special_weight,
                                give_and_take_option = args.give_and_take_option
                                )
                    prem_graphs.append(g[0])
                    hypo_graphs.append(g[1])
                inputs["prem_graphs"] = prem_graphs
                inputs["hypo_graphs"] = hypo_graphs 
            else: 
                inputs["prem_span"]=batch[-2]
                inputs["hypo_span"]=batch[-1]
            
            # outputs = (label_logits, )
            # label_logits: [batch_size, num_labels]
            outputs = model(**inputs)

        pred_logits = torch.cat([pred_logits,outputs[0].to(torch.long)], dim = 0)


    pred_logits= pred_logits.detach().cpu().numpy()
    pred_labels = np.argmax(pred_logits, axis=-1)
    # torch.cuda.empty_cache()
    ## gold_labels = 0 or 1 or 2
    gold_labels = [example.gold_label for example in examples]

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    if "SciTail" in args.output_dir: idx2label = {0: "neutral", 1: "entails"} 
    elif "RTE" in args.output_dir: idx2label = {0: "not_entailment", 1: "entailment"}
    else: # "snli" "SICK" 
        idx2label = {0: "neutral", 1: "contradiction", 2: "entailment"} 
    if "trans_Overlap" in args.predict_file:
        idx2label = {1: "non-entailment", 0: "entailment"} 
        pred_labels = [int(i != 0) for i in pred_labels]
        
    results = get_sklearn_score(pred_labels, gold_labels, idx2label)

    output_dir = os.path.join( args.output_dir, 'eval')

    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type ='w'

    if os.path.exists(args.model_name_or_path):
        print(args.model_name_or_path)
        eval_file_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    else: eval_file_name = "init_weight"
    output_eval_file = os.path.join(output_dir, "eval_result_{}.txt".format(eval_file_name))

    with open(output_eval_file, out_file_type, encoding='utf-8') as f:
        f.write("train loss: {}\n".format(tr_loss))
        f.write("epoch: {}\n".format(epoch_idx))
        for k in results.keys():
            f.write("{} : {}\n".format(k, results[k]))
        f.write("=======================================\n\n")
    return results


def plot_head_map(tokenizer, attn_score1, attn_score2, token_ids, output_dir, alpha=None):
    batch_special_position = torch.tensor(
        [list(filter(lambda x: token_ids[i][x] == tokenizer.sep_token_id, range(len(token_ids[i])))) for i in
         range(0, len(token_ids))])
    
    token_length = [batch_special_position[:, 0].unsqueeze(0).item(), 
                    batch_special_position[:, -2].unsqueeze(0).item(),
                    batch_special_position[:, -1].unsqueeze(0).item()]

    if "prem" in output_dir:
        token_list = [
            [tokenizer.decode(token_ids[:, i])for i in range(1, token_length[0])],
            [tokenizer.decode(token_ids[:, i])for i in range(token_length[1]+1, token_length[2])]
        ]
    else:
        token_list = [
            [tokenizer.decode(token_ids[:, i]) for i in range(token_length[1]+1, token_length[2])],
            [tokenizer.decode(token_ids[:, i]) for i in range(1, token_length[0])]
        ]
    attn_score1 = attn_score1.squeeze(0)
    attn_score2 = attn_score2.squeeze(0)
    
    sum_attn_score = attn_score1 + attn_score2 + torch.abs(attn_score1-attn_score2)
    sum_attn_score = (sum_attn_score/torch.sum(sum_attn_score)).cpu().detach().numpy()
    print(sum_attn_score)
    attn_score1 = (attn_score1/torch.sum(attn_score1)).cpu().detach().numpy()
    attn_score2 = (attn_score2/torch.sum(attn_score2)).cpu().detach().numpy()

    if not os.path.exists(os.path.join(output_dir,"attn_map")):
        os.makedirs(os.path.join(output_dir,"attn_map"))
    if not os.path.exists(os.path.join(output_dir,"attn_map",str(len(os.listdir(output_dir+"/attn_map"))))):
        new_output_dir = os.path.join(output_dir,"attn_map",str(len(os.listdir(output_dir+"/attn_map"))))
        os.makedirs(os.path.join(output_dir,"attn_map",str(len(os.listdir(output_dir+"/attn_map")))))

    if alpha is not None: gate_attn_score = (alpha * attn_score1 + (1-alpha) * attn_score2).cpu().detach().numpy()
    
    for idx, (attn_score, name) in enumerate(zip([attn_score1, attn_score2, gate_attn_score] if alpha is not None else [attn_score1, attn_score2, sum_attn_score], 
                                                 ["Affinity_Attn_Map", "Difference_Attn_Map", "Attn_Map"] if alpha is not None else ["Affinity_Attn_Map", "Difference_Attn_Map", "Sum_Attn_Map"])):
        
        fig = plt.figure(figsize = (30, 10))
        ax = fig.add_subplot(1, 1, 1)
        
        ax.pcolor(attn_score, cmap=plt.cm.YlGnBu)
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(attn_score.shape[0]) + 0.5, minor=False) # mma.shape[1] = target sequence length
        ax.set_yticks(np.arange(attn_score.shape[1]) + 0.5, minor=False) # mma.shape[0] = input sequence length

        # without this I get some extra columns rows
        ax.set_xlim(0, int(attn_score.shape[0]))
        ax.set_ylim(0, int(attn_score.shape[1]))
 
        # source words -> column labels
        ax.set_xticklabels(token_list[0], minor=False, fontsize=20)
        # target words -> row labels
        ax.set_yticklabels(token_list[1], minor=False, fontsize=20)
 
        plt.xticks(rotation=90)
        #plt.colorbar()
        ax.set_title('{} between sentence pair'.format(name))
        plt.savefig(os.path.join(new_output_dir, name+".png"))
    return "Done"


def predict(args, model, tokenizer):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True, do_predict=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    print("***** Running Prediction *****")
    print("  Num examples = %d", len(dataset))
    pred_logits = torch.tensor([]).to(args.device)
    gate_prob = []
    lp_list = []
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs_list = ["input_ids", "attention_mask", "token_type_ids", "position_ids",
                           "prem_word_idxs", "hypo_word_idxs", "real_prem_len", "real_hypo_len",
                           ]


            inputs = dict()
            for n, input in enumerate(inputs_list): inputs[input] = batch[n]

            if not "CAGCN" in args.output_dir:
                prem_graphs = []
                hypo_graphs = []
                for prem_dep, hypo_dep in zip(batch[-2], batch[-1]):
                    dependency = {"premise": prem_dep, "hypothesis":hypo_dep}
                    g = get_graph(dependency, device=args.device, add_label_node=args.add_label_node,
                                label_relation = args.label_relation,
                                both_label_relation = args.both_label_relation,
                                percent_special_weight=args.percent_special_weight,
                                give_and_take_option = args.give_and_take_option
                                )
                    prem_graphs.append(g[0])
                    hypo_graphs.append(g[1])
                inputs["prem_graphs"] = prem_graphs
                inputs["hypo_graphs"] = hypo_graphs 
            else: 
                inputs["prem_span"]=batch[-2]
                inputs["hypo_span"]=batch[-1]
            
            # outputs = (label_logits, )
            # label_logits: [batch_size, num_labels]
            outputs = model(**inputs)

        pred_logits = torch.cat([pred_logits,outputs[0]], dim = 0)
        if len(outputs) > 3: gate_prob.append(outputs[-1])
        if args.draw_attn_map: 
            plot_head_map(tokenizer,outputs[1],outputs[2], inputs["input_ids"], args.output_dir, alpha=outputs[3] if len(outputs) > 3 else None)
        if args.draw_tsne: 
            lp_list.append(outputs[3])
    if args.draw_attn_map: 
        return "check "+args.output_dir
    pred_logits= pred_logits.detach().cpu().numpy()
    pred_labels = np.argmax(pred_logits, axis=-1)
    # torch.cuda.empty_cache()
    ## gold_labels = 0 or 1 or 2 or 3
    gold_labels = [example.gold_label for example in examples]
    genre = [example.genre for example in examples]

    if args.draw_tsne: 
        lp_list = torch.cat(lp_list, dim=0)
        tsne = TSNE(random_state=args.seed)
        digits_tsne = tsne.fit_transform(lp_list.detach().cpu().numpy())
        colors = ['darkblue', '#BD3430', # hans # textflint_overlap
        # colors = ['#476A2A', '#BD3430', # scitail
        # colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525', # snli, sick textflint_addsent
                '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
        gold_color = [colors[g] for g in gold_labels]

        if ("hans" in args.predict_file) or ("trans_Overlap" in args.predict_file):
            idx2label = {0: "non-entailment", 1: "entailment"} 
            pred_new_labels = [int(i == 2) for i in pred_labels]
        else: pred_new_labels = pred_labels

        if "hans" in args.predict_file:
            genre_dict = {k: [] for k in set(genre)}
            for i,g in enumerate(genre): 
                genre_dict[g].append(i)
                # correct
                #if gold_labels[i] == pred_new_labels[i]: genre_dict[g].append(i)
            for i, (k, v) in enumerate(genre_dict.items()):
                plt.figure(i)
                plt.scatter(digits_tsne[v, 0], digits_tsne[v, 1], s=10, c=[gold_color[vi] for vi in v])
                plt.savefig(os.path.join(args.output_dir, k.strip()+"_tsne_42.png"))
            plt.figure(i+1)

        plt.scatter(digits_tsne[:, 0], digits_tsne[:, 1],s=10, c=gold_color)

        plt.savefig(os.path.join(args.output_dir, args.draw_tsne_name))
    
    if "snli" in args.output_dir: idx2label = {0: "neutral", 1: "contradiction", 2: "entailment"}
    elif "SICK" in args.output_dir: idx2label = {0: "neutral", 1: "contradiction", 2: "entailment"}
    elif "sick" in args.output_dir: idx2label = {0: "neutral", 1: "contradiction", 2: "entailment"} 
    elif "SciTail" in args.output_dir: idx2label = {0: "neutral", 1: "entails"} 
    
    if "hans" in args.predict_file: 
        idx2label = {0: "non-entailment", 1: "entailment"} 
        pred_labels = [int(i == 2) for i in pred_labels]
    if "trans_Overlap" in args.predict_file:
        idx2label = {0: "non-entailment", 1: "entailment"} 
        pred_labels = [int(i == 2) for i in pred_labels]

    print(idx2label)

    results = get_sklearn_score(pred_labels, gold_labels, idx2label)

    output_dir = os.path.join( args.output_dir, 'test-{}'.format(str(args.checkpoint)))

    out_file_type = 'a'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        out_file_type ='w'

    if os.path.exists(args.model_name_or_path):
        print(args.model_name_or_path)
        eval_file_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    else:
        eval_file_name = "init_weight"
    output_test_file = os.path.join(output_dir, "test_result_{}_incorrect.txt".format(eval_file_name))

    with open(output_test_file, out_file_type, encoding='utf-8') as f:
        print('\n\n=====================outputs=====================')
        for i, (g, p) in enumerate(zip(gold_labels, pred_labels)):
            if idx2label[g] != idx2label[p]:
                f.write("premise: {}\thypothesis: {}\tcorrect: {}\tpredict: {}\n".format(examples[i].premise,
                                                                                             examples[i].hypothesis,
                                                                                             idx2label[g],
                                                                                             idx2label[p]))
                                                             
        for k in results.keys():
            f.write("{} : {}\n".format(k, results[k]))
        f.write("=======================================\n\n")
    
    out_pair = {"premise": [], "hypothesis": [], "correct": [], "predict": []}
    if "hans" in args.predict_file: out_pair["heuristic"] = []
    for i, (g, p) in enumerate(zip(gold_labels, pred_labels)):
        for k, v in zip(out_pair.keys(), [examples[i].premise, examples[i].hypothesis, idx2label[g], idx2label[p]]):
            out_pair[k].append(v)
        if "hans" in args.predict_file: out_pair["heuristic"].append(examples[i].genre)
    df = pd.DataFrame(out_pair)
    df.to_csv(os.path.join(output_dir, "test_result_{}.csv".format(eval_file_name)), index=False)

    return results
