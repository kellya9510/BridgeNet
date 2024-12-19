import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import copy
import math

import dgl
from dgl.nn.pytorch import utils
import dgl.function as fn

from torch.nn import CrossEntropyLoss

from dgl.nn.pytorch import RelGraphConv, GraphConv, GATConv

from allennlp.nn.util import masked_softmax, masked_max, masked_mean

from src.CA_GCN.GCNLayer import GraphConvolution
from src.CA_GCN.HighWay import HighWay
from src.CA_GCN.graph_util import BottledOrthogonalLinear

class baseline_model(nn.Module):
    def __init__(self, language_model, config):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels

        # contextual Encoding
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.reset_parameters

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        prem_word_idxs=None,  # (batch, max_pair_sen, 2) 
        hypo_word_idxs=None,  
        real_prem_len=None,
        real_hypo_len=None,
        label_input_ids=None,
        label_attention_mask=None,
        label_token_type_ids=None,
        label_position_ids=None,
        label_word_idxs=None,
        prem_graphs=None,
        hypo_graphs=None,
        labels=None,
    ):
        # ========================================================
        # contextual Encoding
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]
        lm = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
        )
        pooled_output = lm[1]

        logits = self.classifier(pooled_output)

        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs



class AffinityDiffProposedModel(nn.Module):
    def __init__(
        self,
        language_model,
        config,
        max_prem_length=512,
        max_hypo_length=512,
        device="cpu",
        d_embed=768,
        sep_token_id = 102,
        num_relations=105,
        add_interact_graphs=True,
        graph_n_bases=-1,
        graph_dim=16,
        gcn_dep=0.0,
        gcn_layers=2,
        activation=F.relu,
        n_graph_attn_composition_layers=0,
    ):
        super().__init__()
        self.language_model = language_model
        self.sep_token_id = sep_token_id
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.max_prem_length = max_prem_length
        self.max_hypo_length = max_hypo_length
        self.device = device
        self.d_embed = d_embed

        ################## link_phrase ########################
        self.affinity_attn = MultiHeadAttn(
            self.hidden_size, num_heads=1, dropout=config.hidden_dropout_prob
        )
        
        self.diff_attn = DiffAttn(
            self.hidden_size, num_heads=1, dropout=config.hidden_dropout_prob, 
        )
        
        self.special_proj = nn.Linear(5*self.hidden_size, self.hidden_size)
        self.special_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # # gate_function
        # self.sigmoid = nn.Sigmoid()
        # self._lambda = nn.Parameter(torch.tensor(0.0)) # Initialize to 0.0 (sigmoid(0) = 0.5)
        ############################################################

        ## token2word
        self.token2word = Token2Word(dropout=config.hidden_dropout_prob)

        # relational layer
        self.graph_dim = graph_dim
        self.emb_proj = nn.Linear(self.hidden_size, self.graph_dim)
        self.num_relations = num_relations
        self.n_graph_attn_composition_layers = n_graph_attn_composition_layers
        self.activation = activation

        def get_gnn_instance(n_layers):
            return RGCN(
                h_dim=self.graph_dim,
                num_relations=self.num_relations,
                num_hidden_layers=n_layers,
                dropout=gcn_dep,
                activation=self.activation,
                num_bases=graph_n_bases,
                eps=self.config.layer_norm_eps,
            )

        self.rgcn = get_gnn_instance(gcn_layers)

        self.add_interact_graphs = add_interact_graphs
        if self.add_interact_graphs:
            # interact_graphs()
            if self.n_graph_attn_composition_layers > 0:
                self.composition_rgcn = get_gnn_instance(
                    self.n_graph_attn_composition_layers
                )
            self.attn_biaffine = BilinearMatrixAttention(
                self.graph_dim, self.graph_dim, use_input_biases=True
            )
            self.attn_proj = nn.Linear(4 * self.graph_dim, self.graph_dim)

        self.graph_last_proj = nn.Linear(self.graph_dim * 2, self.d_embed)

        self.graph_output_proj = nn.Linear(self.graph_dim * 4, self.d_embed)
        self.graph_output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(2 * self.hidden_size + self.d_embed, self.num_labels)
        # self.classifier = nn.Linear(self.hidden_size + self.d_embed, self.num_labels)
        self.reset_parameters

    def reset_parameters(self):
        self.special_proj.reset_parameters()
        self.emb_proj.reset_parameters()
        self.attn_proj.reset_parameters()
        self.graph_last_proj.reset_parameters()
        self.graph_output_proj.reset_parameters()
        self.classifier.reset_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        prem_word_idxs=None,  # (batch, max_pair_sen, 2)  
        hypo_word_idxs=None,  
        real_prem_len=None,  # word-based
        real_hypo_len=None,

        prem_graphs=None,
        hypo_graphs=None,
        labels=None,
    ):
        # ========================================================
        # Premise-Hypothesis Pair Encoder
        # ========================================================
        # discriminator_hidden_states: [batch, max_seq_len, hidden_size]

        batch_special_position = torch.tensor(
            [[0] + list(filter(lambda x: input_ids[i][x] == self.sep_token_id, range(len(input_ids[i])))) for i in
             range(0, len(input_ids))]) #, device=self.device)  # sep token: 102 => 3 or 4
        
        lm = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
        )
        discriminator_hidden_states = lm[0]
        pooled_output = lm[1]
        self.batch_size = discriminator_hidden_states.size(0)

        # ========================================================
        # Syntactic Feature Enhancing Module
        # ========================================================

        prem_max_token_len = torch.max(batch_special_position[:, 1] - 1, dim=0)[0].item()
        hypo_max_token_len = torch.max(batch_special_position[:, -1] - batch_special_position[:, -2] - 1, dim=0)[0].item()
        
        prem_tokens = torch.zeros(self.batch_size, prem_max_token_len, discriminator_hidden_states.size(1)).to(self.device)
        hypo_tokens = torch.zeros(self.batch_size, hypo_max_token_len, discriminator_hidden_states.size(1)).to(self.device)
        
        #"""
        prem_tokens[torch.arange(self.batch_size).unsqueeze(1), torch.arange(prem_max_token_len).unsqueeze(0), torch.ones(1, 1, dtype=torch.long) + batch_special_position[:, 0].unsqueeze(1) + torch.arange(prem_max_token_len).unsqueeze(0)] = 1
        prem_tokens[torch.arange(self.batch_size).unsqueeze(1), torch.arange(prem_max_token_len).unsqueeze(0), batch_special_position[:, 1].unsqueeze(1) + torch.arange(prem_max_token_len).unsqueeze(0)] = 0
        hypo_tokens[torch.arange(self.batch_size).unsqueeze(1), torch.arange(hypo_max_token_len).unsqueeze(0), torch.ones(1, 1, dtype=torch.long) + batch_special_position[:, -2].unsqueeze(1) + torch.arange(hypo_max_token_len).unsqueeze(0)] = 1
        hypo_tokens[torch.arange(self.batch_size).unsqueeze(1), torch.arange(hypo_max_token_len).unsqueeze(0), batch_special_position[:, -1].unsqueeze(1) + torch.arange(hypo_max_token_len).unsqueeze(0)] = 0
        """
        for b_idx in range(self.batch_size):
            prem_tokens[b_idx, :, torch.tensor(list(range(1+batch_special_position[b_idx, 0].item(),batch_special_position[b_idx, 1].item()))).unsqueeze(0)] = 1
            hypo_tokens[b_idx, :, torch.tensor(list(range(1+batch_special_position[b_idx, -2].item(),batch_special_position[b_idx, -1].item()))).unsqueeze(0)] = 1
            
            prem_start_pos = batch_special_position[b_idx, 0].item() + 1
            prem_end_pos = batch_special_position[b_idx, 1].item()
            hypo_start_pos = batch_special_position[b_idx, -2].item() + 1
            hypo_end_pos = batch_special_position[b_idx, -1].item()

            prem_positions = torch.tensor(list(range(prem_start_pos, prem_end_pos)))
            if len(prem_positions) > prem_max_token_len:
                prem_positions = prem_positions[:prem_max_token_len]            
            hypo_positions = torch.tensor(list(range(hypo_start_pos, hypo_end_pos)))
            if len(hypo_positions) > hypo_max_token_len:
                hypo_positions = hypo_positions[:hypo_max_token_len]

            prem_tokens[b_idx, torch.arange(len(prem_positions)), prem_positions] = 1
            hypo_tokens[b_idx, torch.arange(len(hypo_positions)), hypo_positions] = 1
        
        """           
        prem_tokens = torch.matmul(prem_tokens, discriminator_hidden_states)
        hypo_tokens = torch.matmul(hypo_tokens, discriminator_hidden_states)
        prem_token_masks = torch.arange(prem_max_token_len)[None, :] < (batch_special_position[:, 1, None] - batch_special_position[:, 0, None] - 1)
        hypo_token_masks = torch.arange(hypo_max_token_len)[None, :] < (batch_special_position[:, -1, None] - batch_special_position[:, -2, None] - 1)
        prem_token_masks = prem_token_masks.to(self.device)
        hypo_token_masks = hypo_token_masks.to(self.device)
        
        
        affinity_attention_output, affinity_score = self.affinity_attn(
            hypo_tokens, prem_tokens, prem_tokens, last_layer=False
        )
        diff_attention_output, diff_score = self.diff_attn(
            hypo_tokens, prem_tokens, prem_tokens, 
            M=hypo_token_masks.unsqueeze(-1)*prem_token_masks.unsqueeze(1), last_layer=False
        )
        
        A_mean = self.pool_graph(affinity_attention_output, hypo_token_masks, is_mean_pool=True)
        D_mean = self.pool_graph(diff_attention_output, hypo_token_masks, is_mean_pool=True)
        link_phrase = self.special_dropout(self.special_proj(
            # torch.cat([A_mean, D_mean, torch.abs(A_mean - D_mean)], dim=-1) 
            # torch.cat([A_mean, D_mean, torch.abs(A_mean - D_mean), A_mean*D_mean], dim=-1) 
            torch.cat([A_mean, D_mean, A_mean + D_mean, A_mean - D_mean, torch.abs(A_mean - D_mean)], dim=-1) 
            ))  

        link_phrase = link_phrase.unsqueeze(1)
        link_phrase_origin = link_phrase.squeeze(1).clone()
        
        # ========================================================
        # change token2word for Expanding Syntactic Features
        ## change token2word of premise and hypothesis embedding
        # ========================================================
        prem_word_embedding, prem_word_masks = self.token2word(
            hidden_states=discriminator_hidden_states,
            word_idxs=prem_word_idxs,
            max_word_len=real_prem_len,
        )  # w/ cls (first position)
        hypo_word_embedding, hypo_word_masks = self.token2word(
            hidden_states=discriminator_hidden_states,
            word_idxs=hypo_word_idxs,
            max_word_len=real_hypo_len,
        )  #  w/ sep (first position, (real_hypo_len-1)-th position)

        # # separate premise and hypothesis w/o special_tokens
        prem_word_embedding = prem_word_embedding[:, 1:, :]  # remove cls
        hypo_word_embedding = hypo_word_embedding[:, 1:, :]  # remove first sep
        hypo_word_embedding = torch.cat(
            [
                torch.cat(
                    (
                        hypo_word_embed[: hypo_len.item() - 2, :],
                        hypo_word_embed[hypo_len.item() - 1 :, :],
                    ),
                    dim=0,
                ).unsqueeze(0)
                for hypo_len, hypo_word_embed in zip(real_hypo_len, hypo_word_embedding)
            ],
            dim=0,
        )

        hypo_word_embedding = torch.cat((link_phrase, hypo_word_embedding), dim=1)

        prem_word_masks = prem_word_masks[:, 1:]  # remove cls
        hypo_word_masks = hypo_word_masks[:, 1:]  # remove sep + add mask of label_embedding
        # hypo_word_masks = hypo_word_masks[:, 2:]  # remove sep

        #'''
        # ========================================================
        # Generating Syntax-based Representation with RGCN
        # ========================================================
        # prem
        prem_word_embedding = self.flatten_node_embeddings(
            prem_word_embedding, prem_word_masks
        )
        prem_word_embedding = self.activation(self.emb_proj(prem_word_embedding))
        prem_graphs = dgl.batch(prem_graphs)
        if len(prem_word_embedding) != len(prem_graphs.ndata["id"]):
            print(len(prem_word_embedding))
            print(prem_graphs.ndata)
        prem_word_embedding = self.rgcn(prem_graphs, prem_word_embedding)
        prem_word_embedding = self.unflatten_node_embeddings(
            prem_word_embedding, prem_word_masks
        )
        # hypo
        hypo_word_embedding = self.flatten_node_embeddings(
            hypo_word_embedding, hypo_word_masks
        )
        hypo_word_embedding = self.activation(self.emb_proj(hypo_word_embedding))
        hypo_graphs = dgl.batch(hypo_graphs)
        if len(hypo_word_embedding) != len(hypo_graphs.ndata["id"]):
            print(len(hypo_word_embedding))
            print(hypo_graphs.ndata)
        hypo_word_embedding = self.rgcn(hypo_graphs, hypo_word_embedding)
        hypo_word_embedding = self.unflatten_node_embeddings(
            hypo_word_embedding, hypo_word_masks
        )

        if self.add_interact_graphs:
            prem_word_embedding, hypo_word_embedding = self.interact_graphs(
                prem_graphs,
                hypo_graphs,
                prem_word_embedding,
                hypo_word_embedding,
                prem_word_masks,
                hypo_word_masks,
            )

        prem_rgcn_output = self.pool_graph(prem_word_embedding, prem_word_masks)
        hypo_rgcn_output = self.pool_graph(hypo_word_embedding, hypo_word_masks)

        rgcn_output = self.graph_output_dropout(
            (
                self.graph_last_proj(
                    torch.cat((prem_rgcn_output, hypo_rgcn_output), dim=-1)
                )
            )
        )

        logits = self.classifier(torch.cat((pooled_output, link_phrase_origin, rgcn_output), dim=-1))
        # logits = self.classifier(torch.cat((pooled_output, link_phrase, rgcn_output), dim=-1))
        # logits = self.classifier(torch.cat((pooled_output, rgcn_output), dim=-1))

        outputs = (logits, affinity_score, diff_score, link_phrase_origin, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits,

    def pool_graph(self, node_embs, node_emb_mask, is_mean_pool=False):
        """
        Parameters:
            node_embs: (bsz, n_nodes, graph_dim)
            node_emb_mask: (bsz, n_nodes)
        Returns:
            (bsz, graph_dim (*2))
        """
        node_emb_mask = node_emb_mask.unsqueeze(-1)
        if is_mean_pool:
            output = masked_mean(node_embs, node_emb_mask, 1)
        else:
            output = masked_max(node_embs, node_emb_mask, 1)
        output = torch.where(node_emb_mask.any(1), output, torch.zeros_like(output))

        return output

    def interect_vector(
        self,
        attn,
        node_embs_a,
        node_embs_b,
        node_emb_mask_a,
        node_emb_mask_b,
    ):
        """
        Parameters:
            node_embs_{a,b}: (bsz, n_nodes_{a,b}, graph_dim)
            node_emb_mask_{a,b}: (bsz, n_nodes_{a,b})
        """
        # attn: (bsz, n_nodes_a, n_nodes_b)

        normalized_attn_a = masked_softmax(
            attn, node_emb_mask_a.unsqueeze(2), dim=1
        )  # (bsz, n_nodes_a, n_nodes_b)
        attended_a = normalized_attn_a.transpose(1, 2).bmm(
            node_embs_a
        )  # (bsz, n_nodes_b, graph_dim)
        new_node_embs_b = torch.cat(
            [
                node_embs_b,
                attended_a,
                node_embs_b - attended_a,
                node_embs_b * attended_a,
            ],
            dim=-1,
        )  # (bsz, n_nodes_b, graph_dim * 4)

        normalized_attn_b = masked_softmax(
            attn, node_emb_mask_b.unsqueeze(1), dim=2
        )  # (bsz, n_nodes_a, n_nodes_b)
        attended_b = normalized_attn_b.bmm(node_embs_b)  # (bsz, n_nodes_a, graph_dim)
        new_node_embs_a = torch.cat(
            [
                node_embs_a,
                attended_b,
                node_embs_a - attended_b,
                node_embs_a * attended_b,
            ],
            dim=-1,
        )  # (bsz, n_nodes_a, graph_dim * 4)
        return new_node_embs_a, new_node_embs_b
    
    def interact_graphs(
        self,
        graph_a,
        graph_b,
        node_embs_a,
        node_embs_b,
        node_emb_mask_a,
        node_emb_mask_b,
    ):
        """
        Parameters:
            node_embs_{a,b}: (bsz, n_nodes_{a,b}, graph_dim)
            node_emb_mask_{a,b}: (bsz, n_nodes_{a,b})
        """
        orig_node_embs_a, orig_node_embs_b = node_embs_a, node_embs_b
        attn = attn = self.attn_biaffine(node_embs_a, node_embs_b)
        new_node_embs_a, new_node_embs_b = self.interect_vector(attn, node_embs_a, node_embs_b, node_emb_mask_a, node_emb_mask_b)
        new_node_embs_b = self.activation(self.attn_proj(new_node_embs_b))  # (bsz, n_nodes_a, graph_dim)
        new_node_embs_a = self.activation(self.attn_proj(new_node_embs_a))  # (bsz, n_nodes_b, graph_dim)

        node_embs_a = self.flatten_node_embeddings(new_node_embs_a, node_emb_mask_a)
        node_embs_b = self.flatten_node_embeddings(new_node_embs_b, node_emb_mask_b)

        if self.n_graph_attn_composition_layers > 0:
            node_embs_a = self.composition_rgcn(graph_a, node_embs_a)
            node_embs_b = self.composition_rgcn(graph_b, node_embs_b)

        node_embs_a = self.unflatten_node_embeddings(node_embs_a, node_emb_mask_a)
        node_embs_b = self.unflatten_node_embeddings(node_embs_b, node_emb_mask_b)

        # If the other graph is empty, we don't do any attention at all and use the original embedding
        node_embs_a = torch.where(
            node_emb_mask_b.any(1, keepdim=True).unsqueeze(-1),
            node_embs_a,
            orig_node_embs_a,
        )
        node_embs_b = torch.where(
            node_emb_mask_a.any(1, keepdim=True).unsqueeze(-1),
            node_embs_b,
            orig_node_embs_b,
        )

        return node_embs_a, node_embs_b

    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = (
            node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        )
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        # return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = (
            node_embeddings_mask.clone().detach().type(torch.BoolTensor).tolist()
        )
        # mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0],
            node_embeddings_mask.shape[1],
            node_embeddings.shape[-1],
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings


class MultiHeadAttn(nn.Module):
    def __init__(self, num_units, num_heads=1, dropout=0, gpu=True, causality=False):
        """Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        """
        super(MultiHeadAttn, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.causality = causality
        self.Q_proj = nn.Sequential(
            nn.Linear(self.num_units, self.num_units), nn.ReLU()
        )
        self.K_proj = nn.Sequential(
            nn.Linear(self.num_units, self.num_units), nn.ReLU()
        )
        self.V_proj = nn.Sequential(
            nn.Linear(self.num_units, self.num_units), nn.ReLU()
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # get dim to concat
        concat_dim = len(Q.shape) - 1

        if concat_dim == 1:
            Q = Q.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=1)
            concat_dim = 2

        # Split and concat
        Q_ = torch.cat(
            torch.chunk(Q, self.num_heads, dim=concat_dim), dim=0
        )  # (h*N, T_q, C/h)
        K_ = torch.cat(
            torch.chunk(K, self.num_heads, dim=concat_dim), dim=0
        )  # (h*N, T_q, C/h)
        V_ = torch.cat(
            torch.chunk(V, self.num_heads, dim=concat_dim), dim=0
        )  # (h*N, T_q, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if not last_layer:
            attn_score = outputs.clone()
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
            # attn_score = outputs.clone()

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(
            1, 1, keys.size()[1]
        )  # (h*N, T_q, T_k)
        query_masks = query_masks.reshape(
            [outputs.shape[0], outputs.shape[1], outputs.shape[2]]
        )

        outputs = outputs * query_masks

        # Dropouts
        outputs = self.dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer:
            return outputs
        attn_score = attn_score * query_masks

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(
            torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim
        )  # (N, T_q, C)

        # Residual connection
        # outputs += queries

        return outputs, attn_score


class DiffAttn(nn.Module):
    def __init__(self, num_units, num_heads=1, dropout=0, gpu=True, causality=False, is_dist_method=None):
        """Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        """
        super(DiffAttn, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.causality = causality
        self.is_dist_method = is_dist_method
        
        self.Q_proj = nn.Sequential(
            nn.Linear(self.num_units, self.num_units), nn.ReLU()
        )
        self.K_proj = nn.Sequential(
            nn.Linear(self.num_units, self.num_units), nn.ReLU()
        )
        self.V_proj = nn.Sequential(
            nn.Linear(self.num_units, self.num_units), nn.ReLU()
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, M=None, last_layer=False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # get dim to concat
        concat_dim = len(Q.shape) - 1

        if concat_dim == 1:
            Q = Q.unsqueeze(dim=1)
            queries = queries.unsqueeze(dim=1)
            concat_dim = 2

        # Split and concat
        Q_ = torch.cat(
            torch.chunk(Q, self.num_heads, dim=concat_dim), dim=0
        )  # (h*N, T_q, C/h)
        K_ = torch.cat(
            torch.chunk(K, self.num_heads, dim=concat_dim), dim=0
        )  # (h*N, T_k, C/h)
        V_ = torch.cat(
            torch.chunk(V, self.num_heads, dim=concat_dim), dim=0
        )  # (h*N, T_v, C/h)

        # Difference
        if self.is_dist_method == "euclidean":
            diff = Q_.unsqueeze(2) - K_.unsqueeze(1) # (h*N, T_q, T_k, C)
            dist = torch.norm(diff, p=2, dim=-1)
            outputs = - dist
            # Scale
            outputs = outputs / (K_.size()[-1] ** 0.5)
        elif self.is_dist_method == "cosine":
            cosine_similarity = F.cosine_similarity(Q_.unsqueeze(2), K_.unsqueeze(1), dim=-1)
            outputs = 0.5 * cosine_similarity + 0.5  # 0과 1 사이로 scale
        else:
            diff = Q_.unsqueeze(2) - K_.unsqueeze(1) # (h*N, T_q, T_k, C)
            outputs = torch.sum(diff, dim=-1)  # (h*N, T_q, T_k)
            outputs += M
            # Scale
            outputs = outputs / (K_.size()[-1] ** 0.5)
        
        # Activation
        if not last_layer:
            attn_score = outputs.clone()
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
            # attn_score = outputs.clone()

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(
            1, 1, keys.size()[1]
        )  # (h*N, T_q, T_k)
        query_masks = query_masks.reshape(
            [outputs.shape[0], outputs.shape[1], outputs.shape[2]]
        )
        outputs = outputs * query_masks

        # Dropouts
        outputs = self.dropout(outputs)  # (h*N, T_q, T_k)

        if last_layer:
            return outputs
        attn_score = attn_score * query_masks

        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(
            torch.chunk(outputs, self.num_heads, dim=0), dim=concat_dim
        )  # (N, T_q, C)

        # Residual connection
        # outputs += queries

        return outputs, attn_score


class Token2Word(nn.Module):
    def __init__(self, dropout):
        super(Token2Word, self).__init__()
        self.dropout = nn.Dropout(dropout) 
        self.reset_parameters

    def reset_parameters(self):
        self.dropout.reset_parameters()

    def forward(self, hidden_states, word_idxs, max_word_len):
        output = self.mean_pooling(word_idxs, max_word_len, hidden_states.size(1))
        word_idxs = output[0]
        word_masks = output[1]

        # (batch, max_sen, seq_len) @ (batch, seq_len, hidden) = (batch, max_sen, hidden)
        word_embedding = torch.matmul(word_idxs, hidden_states)
        # word_embedding = self.dropout(torch.matmul(word_idxs, hidden_states))

        return word_embedding, word_masks

    def mean_pooling(self, word_idxs_feature, max_idx_feature, seq_len):
        # seq_len = hidden_states.size[1]
        mean_pooling = []
        word_masks = []
        """
        word_idxs_feature: (batch, max_pair_sen, 2)
        max_idx_feature: (batch)

        output: 
        mean_pooling: (batch, max_pair_sen, seq_len)
        word_masks: (batch, max_pair_sen)
        """
        for word_idxs, max_idx in zip(word_idxs_feature, max_idx_feature):
            not_word_list = []
            word_masks.append(
                [1] * max_idx.item() + [0] * (word_idxs.size(0) - max_idx.item())
            )
            for k, s_e_idx in enumerate(word_idxs):
                prem_not_word_idxs = [0] * seq_len
                if k < max_idx.item():
                    start_idx = s_e_idx[0].item()
                    end_idx = s_e_idx[1].item()
                    if (end_idx - start_idx + 1) > 0:
                        for j in range(start_idx, end_idx + 1):
                            prem_not_word_idxs[j] = 1 / (end_idx - start_idx + 1)
                # if k< 2: print(prem_not_word_idxs)
                not_word_list.append(prem_not_word_idxs)
            mean_pooling.append(not_word_list)
        mean_pooling = torch.tensor(mean_pooling, dtype=torch.float).to("cuda")
        return (mean_pooling, torch.tensor(word_masks, dtype=torch.long).to("cuda"))


class BilinearMatrixAttention(nn.Module):
    def __init__(
        self,
        matrix_1_dim,
        matrix_2_dim,
        activation=None,
        use_input_biases=False,
        label_dim=1,
    ):
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if label_dim == 1:
            self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = Parameter(
                torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim)
            )
        self._bias = Parameter(torch.Tensor(1))

        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1, matrix_2):
        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        output = final.squeeze(1) + self._bias

        return output


class GCN(nn.Module):
    def __init__(self, gcn_layers, 
                 in_features, out_features, edge_types, gcn_dep, gcn_use_bn, use_highway, mutual_link,
                 rgcn=None, composition_rgcn=None,
                 device="cpu"):
        super(GCN, self).__init__()
        self.device = device
        self.mutual_link = mutual_link
        self.gcn_layers = gcn_layers
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = edge_types
        self.gcn_dep = gcn_dep
        self.gcn_use_bn = gcn_use_bn
        self.use_highway = use_highway
        
        self.rgcn = rgcn
        self.emb_proj = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()  # activation function
        self.composition_rgcn=composition_rgcn

        self.attn_biaffine = BilinearMatrixAttention(out_features, out_features, use_input_biases=True)
        self.attn_proj = nn.Linear(4 * out_features, out_features)

        # GCN
        self.gcns = nn.ModuleList()
        for i in range(gcn_layers):
            gcn = GraphConvolution(in_features=in_features if i != 0 else out_features,
                                   out_features=out_features,
                                   edge_types=edge_types,
                                   dropout=gcn_dep if i != gcn_layers - 1 else None,
                                   use_bn=gcn_use_bn,
                                   device=device)
            self.gcns.append(gcn)

        # Highway
        if use_highway:
            if in_features == out_features:
                self.hws = nn.ModuleList()
                for i in range(gcn_layers):
                    hw = HighWay(size=out_features, dropout_ratio=gcn_dep)
                    self.hws.append(hw)
            else:
                print("When using highway, the input feature size should be equivalent to the output feature size. "
                      "The highway structure is abandoned.")

        self.W = nn.ParameterList()

        w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('tanh'))
        self.W.append(w)

        for i in range(gcn_layers):
            w = nn.Parameter(torch.Tensor(out_features, out_features))
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('tanh'))
            self.W.append(w)


    def forward(self, seq_features1, mask1, graph1, seq_features2, mask2, graph2, adj):

        max_seq1_len = seq_features1.size(1)
        seq_features = torch.cat((seq_features1, seq_features2), 1)

        if self.rgcn:
            prem_emb = seq_features1.clone().detach()
            hypo_emb = seq_features2.clone().detach()

        mask = torch.matmul(torch.unsqueeze(mask1, 2), torch.unsqueeze(mask2, 1))

        for i in range(self.gcn_layers):
            seq_features1 = seq_features[:, :max_seq1_len, :]
            seq_features2 = seq_features[:, max_seq1_len:, :]

            C = F.tanh(torch.matmul(torch.matmul(seq_features1, self.W[i]), torch.transpose(seq_features2, 1, 2)))

            if self.mutual_link == "co_attn":
                C1 = masked_softmax(C, mask)
                C2 = masked_softmax(torch.transpose(C, 1, 2), torch.transpose(mask, 1, 2))

                adj_mask = torch.zeros(adj.size(0), adj.size(1) - 1, adj.size(2), adj.size(3)).to(self.device)  # edge_types == adj.size(1)
                C1 = torch.cat((torch.zeros(C1.size(0), C1.size(1), adj.size(3) - C1.size(2)).to(self.device), C1), 2)
                C2 = torch.cat((C2, torch.zeros(C2.size(0), C2.size(1), adj.size(3) - C2.size(2)).to(self.device)), 2)

                adj_mask = torch.cat((adj_mask, torch.unsqueeze(torch.cat((C1, C2), 1), 1)), 1)
                adj_mask = adj_mask + adj
            else:
                adj_mask = adj

            seq_features = torch.cat((seq_features1, seq_features2), 1)

            if self.rgcn:

                # flatten_node_embeddings
                # (batch, max_prem_word_len, hidden_size) -> (sum(max_prem_word_len for each batch), hidden_size)
                prem_emb = self.flatten_node_embeddings(prem_emb, mask1)
                hypo_emb = self.flatten_node_embeddings(hypo_emb, mask2)

                # (batch * max_prem_word_len, hidden_size) -> (batch * max_prem_word_len, graph_dim=out_features)
                prem_emb = self.activation(self.emb_proj(prem_emb))
                hypo_emb = self.activation(self.emb_proj(hypo_emb))

                # (batch * max_prem_word_len, graph_dim=out_features)
                if len(prem_emb) != len(graph1.ndata["id"]):
                    print(len(prem_emb))
                    print(graph1.ndata)

                if len(hypo_emb) != len(graph2.ndata["id"]):
                    print(len(hypo_emb))
                    print(graph2.ndata)

                prem_emb = self.rgcn.each_layer_forward(graph1, prem_emb, i)
                hypo_emb = self.rgcn.each_layer_forward(graph2, hypo_emb, i)

                # (batch * max_prem_word_len, out_features) -> (batch, max_prem_word_len, out_features)
                prem_emb = self.unflatten_node_embeddings(prem_emb, mask1)
                hypo_emb = self.unflatten_node_embeddings(hypo_emb, mask2)

                # interact_graphs
                # (batch, max_prem_word_len, out_features)
                prem_emb, hypo_emb = self.interact_graphs(graph1, graph2, prem_emb, hypo_emb, mask1, mask2, i)

                rgcn_ith_output = torch.cat((prem_emb, hypo_emb), dim=1)

            if self.use_highway:
                seq_features = self.gcns[i](seq_features, adj_mask, rgcn_ith_output=rgcn_ith_output if self.rgcn else None) + self.hws[i](seq_features)  # (batch_size, seq_len, d')
            else:
                seq_features = self.gcns[i](seq_features, adj_mask, rgcn_ith_output=rgcn_ith_output if self.rgcn else None)

        seq_features1 = seq_features[:, :max_seq1_len, :]
        seq_features2 = seq_features[:, max_seq1_len:, :]

        return seq_features1, seq_features2

    def interact_graphs(self, graph_a, graph_b, node_embs_a, node_embs_b, node_emb_mask_a, node_emb_mask_b, i):
        """
        Parameters:
            node_embs_{a,b}: (bsz, n_nodes_{a,b}, graph_dim)
            node_emb_mask_{a,b}: (bsz, n_nodes_{a,b})
        """
        bsz = node_embs_a.size(0)
        orig_node_embs_a, orig_node_embs_b = node_embs_a, node_embs_b

        # attn: (bsz, n_nodes_a, n_nodes_b)
        attn = self.attn_biaffine(node_embs_a, node_embs_b)

        normalized_attn_a = masked_softmax(attn, node_emb_mask_a.unsqueeze(2), dim=1)  # (bsz, n_nodes_a, n_nodes_b)
        attended_a = normalized_attn_a.transpose(1, 2).bmm(node_embs_a)  # (bsz, n_nodes_b, graph_dim)
        new_node_embs_b = torch.cat([node_embs_b, attended_a, node_embs_b - attended_a, node_embs_b * attended_a],
                                    dim=-1)  # (bsz, n_nodes_b, graph_dim * 4)
        new_node_embs_b = self.activation(self.attn_proj(new_node_embs_b))  # (bsz, n_nodes_b, graph_dim)

        normalized_attn_b = masked_softmax(attn, node_emb_mask_b.unsqueeze(1), dim=2)  # (bsz, n_nodes_a, n_nodes_b)
        attended_b = normalized_attn_b.bmm(node_embs_b)  # (bsz, n_nodes_a, graph_dim)
        new_node_embs_a = torch.cat([node_embs_a, attended_b, node_embs_a - attended_b, node_embs_a * attended_b],
                                    dim=-1)  # (bsz, n_nodes_a, graph_dim * 4)
        new_node_embs_a = self.activation(self.attn_proj(new_node_embs_a))  # (bsz, n_nodes_a, graph_dim)

        node_embs_a = self.flatten_node_embeddings(new_node_embs_a, node_emb_mask_a)
        node_embs_b = self.flatten_node_embeddings(new_node_embs_b, node_emb_mask_b)

        if self.composition_rgcn:
            node_embs_a = self.composition_rgcn(graph_a, node_embs_a, i)
            node_embs_b = self.composition_rgcn(graph_b, node_embs_b, i)

        node_embs_a = self.unflatten_node_embeddings(node_embs_a, node_emb_mask_a)
        node_embs_b = self.unflatten_node_embeddings(node_embs_b, node_emb_mask_b)

        # If the other graph is empty, we don't do any attention at all and use the original embedding
        node_embs_a = torch.where(node_emb_mask_b.any(1, keepdim=True).unsqueeze(-1), node_embs_a, orig_node_embs_a)
        node_embs_b = torch.where(node_emb_mask_a.any(1, keepdim=True).unsqueeze(-1), node_embs_b, orig_node_embs_b)

        return node_embs_a, node_embs_b

    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        return node_embeddings[mask_bool_list, :]
        #return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        mask_bool_list = torch.tensor(node_embeddings_mask, dtype=torch.bool).tolist()
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[mask_bool_list, :] = node_embeddings
        # output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings




class RGCN(nn.Module):
    def __init__(
        self,
        h_dim,
        num_relations,
        num_hidden_layers=1,
        dropout=0,
        activation=F.relu,
        num_bases=-1,
        eps=1e-8,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.activation = activation
        self.num_bases = None if num_bases < 0 else num_bases
        self.layers = nn.ModuleList(
            [self.create_graph_layer() for _ in range(num_hidden_layers)]
        )
        self.ffn = nn.Linear(self.h_dim, self.h_dim)
        self.norm_layer = nn.LayerNorm(self.h_dim, eps=eps)

    def each_layer_forward(self, graph, h, i):
        if h != graph.num_nodes():
            raise ValueError("Node embedding initialization shape mismatch")
        h = self.ffn(self.forward_graph_layer(self.layers[i], graph, h)) + h
        h = self.norm_layer(h)
        return h
    
    def forward(self, graph, initial_embeddings):
        if len(initial_embeddings) != graph.num_nodes():
            raise ValueError("Node embedding initialization shape mismatch")
        h = initial_embeddings
        for layer in self.layers:
            # h = self.forward_graph_layer(layer, graph, h)
            h = self.ffn(self.forward_graph_layer(layer, graph, h)) + initial_embeddings
            h = self.norm_layer(h)
        return h

    def create_graph_layer(self):
        return RelGraphConv(
            self.h_dim,
            self.h_dim,
            self.num_relations,
            "basis",
            self.num_bases,
            activation=self.activation,
            self_loop=True,
            dropout=self.dropout,
        )
        # return GraphConv(self.h_dim, self.h_dim)
        # return GATConv(self.h_dim, self.h_dim, num_heads=1)

    def forward_graph_layer(self, layer, graph, h):
        # print(graph.edata['special_weight'])
        return layer(
            graph,
            h,
            graph.edata["type"] if "type" in graph.edata else h.new_empty(0),
            graph.edata["norm"] if "norm" in graph.edata else h.new_empty(0),
            # graph.edata['special_weight'] if 'special_weight' in graph.edata else h.new_empty(0),
        )


