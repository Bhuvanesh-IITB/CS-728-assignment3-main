'''
Part 2: are we lost in the middle?

Goal:
- visualize the attention from the query to gold document based on the distance between them
- use attention as a metric to rank documents for a query
'''

import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)

    # TODO 1: implement to get final query to doc attention stored in doc_scores

    query_start, query_end = query_span
    num_docs   = len(doc_spans)
    num_layers = len(attentions)

    
    doc_scores_cpu = torch.zeros(num_docs)  

    for layer_attn in attentions:
        
        avg_heads = layer_attn.mean(dim=1).squeeze(0).cpu().float()

        for doc_idx, (doc_start, doc_end) in enumerate(doc_spans):
            attn_block = avg_heads[query_start:query_end, doc_start:doc_end]
            doc_scores_cpu[doc_idx] += attn_block.mean().item()

        del avg_heads

    
    doc_scores_cpu /= num_layers

    
    doc_scores = doc_scores_cpu.to(attentions[0].device)

    return doc_scores


def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    # TODO 2: visualize graph
    """
    input -> result: list of dicts with keys:
        - gold_position
        - gold_score
        - gold_rank

    GOAL: Using the results data, generate a visualization that shows how
    attention to the gold tool varies with its position in the prompt.
    """

    positions = [r["gold_position"] for r in result]
    scores    = [r["gold_score"]    for r in result]

    
    pos_to_scores = {}
    for pos, sc in zip(positions, scores):
        if pos not in pos_to_scores:
            pos_to_scores[pos] = []
        pos_to_scores[pos].append(sc)

    sorted_positions = sorted(pos_to_scores.keys())
    avg_scores       = [np.mean(pos_to_scores[p]) for p in sorted_positions]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(sorted_positions, avg_scores,
             marker='o', linewidth=1.5, markersize=4, color='steelblue')
    plt.xlabel("Position of Gold Tool in Prompt (0 = first)", fontsize=12)
    plt.ylabel("Average Attention Score from Query Tokens",   fontsize=12)
    plt.title("Lost-in-the-Middle: Attention to Correct Tool vs Its Position",
              fontsize=13)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Plot saved to {save_path}")


def get_query_span(prompt, tokenizer, item_spans, putils):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.

    Prompt structure:
        [user_header][item_instruction][tool_0]...[tool_N]
        [separator][add_text1][separator][Query: ...]\nCorrect tool_id:[asst_header]

    Query tokens sit after add_text1+separator and before the assistant suffix.
    """
    input_ids    = tokenizer(prompt, add_special_tokens=False).input_ids
    total_tokens = len(input_ids)

    suffix_len    = putils.prompt_suffix_length
    add_text1_len = putils.add_text1_length
    separator_len = len(tokenizer(
        putils.prompt_seperator, add_special_tokens=False).input_ids)

    after_last_tool = item_spans[-1][1]
    query_start     = after_last_tool + separator_len + add_text1_len + separator_len
    query_end       = total_tokens - suffix_len

    return (query_start, query_end)


parser = argparse.ArgumentParser()
parser.add_argument('--seed',      type=int,  default=64)
parser.add_argument('--model',     type=str,  default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int,  default=20)
parser.add_argument("--debug",     action="store_true", help="Enable debug mode")
args = parser.parse_args()

if __name__ == '__main__':
    seed_all(seed=args.seed)

    model_name = args.model
    device     = "cuda:0"

    tokenizer, model = load_model_tokenizer(
        model_name=model_name, device=device, dtype=torch.float16)

    num_heads            = model.config.num_attention_heads
    num_layers           = model.config.num_hidden_layers
    d                    = getattr(model.config, "head_dim",
                                   model.config.hidden_size // model.config.num_attention_heads)
    num_key_value_groups = num_heads // model.config.num_key_value_heads
    softmax_scaling      = d ** -0.5

    train_queries, test_queries, tools = get_queries_and_items()

    print("---- debug print start ----")
    print(f"seed: {args.seed}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data        = []
    avg_latency    = []
    count          = 0
    start_time     = time.time()
    results        = []

    recall_at_1_total = 0
    recall_at_5_total = 0
    num_queries       = len(test_queries)

    for qix in tqdm(range(num_queries)):
        sample         = test_queries[qix]
        qid            = sample["qid"]
        question       = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        num_dbs       = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer    =tokenizer,
            doc_ids      =shuffled_keys,
            dict_all_docs=tools,
        )

        item_spans     = putils.doc_spans
        doc_lengths    = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v: k for k, v in map_docname_id.items()}

        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        gold_tool_id  = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------" * 5)
            print(prompt)
            print("-------" * 5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------" * 5)

        with torch.no_grad():
            attentions = model(**inputs).attentions

        '''
        attentions - tuple of length = # layers
        attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
        '''

        
        del inputs
        torch.cuda.empty_cache()

        query_span = get_query_span(prompt, tokenizer, item_spans, putils)

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # Free attentions from GPU after scoring is done
        del attentions
        torch.cuda.empty_cache()
        gc.collect()

        # TODO: find gold_rank - rank of gold tool in doc_scores
        ranked_indices = torch.argsort(doc_scores, descending=True).cpu().tolist()
        gold_rank      = ranked_indices.index(gold_tool_id) + 1  

        # TODO: find gold_score - score of gold tool
        gold_score = doc_scores[gold_tool_id].item()

        results.append({
            "qid":           qid,
            "gold_position": gold_tool_id,
            "gold_score":    gold_score,
            "gold_rank":     gold_rank
        })

        # TODO: calculate recall@1, recall@5 metric and print at end of loop
        recall_at_1_total += 1 if gold_rank <= 1 else 0
        recall_at_5_total += 1 if gold_rank <= 5 else 0

        if (qix + 1) % 50 == 0:
            r1 = recall_at_1_total / (qix + 1)
            r5 = recall_at_5_total / (qix + 1)
            print(f"[{qix+1}/{num_queries}]  Recall@1: {r1:.4f}  |  Recall@5: {r5:.4f}")

    # Final metrics
    final_r1 = recall_at_1_total / num_queries
    final_r5 = recall_at_5_total / num_queries

    print("\n" + "=" * 50)
    print("PART 2 RESULTS")
    print("=" * 50)
    print(f"Recall@1 : {final_r1:.4f}")
    print(f"Recall@5 : {final_r5:.4f}")
    print("=" * 50)

    analyze_gold_attention(results)
