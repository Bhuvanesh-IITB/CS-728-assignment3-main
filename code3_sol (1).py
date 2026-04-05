import torch
import gc
from tqdm import tqdm
from utils import PromptUtils
import random 

# Number of tools to include per prompt during head selection.
# Reduced from 100 to avoid CUDA OOM on Colab (15GB GPU).
# Gold tool is always included, rest are randomly sampled distractors.
TOOLS_PER_PROMPT = 20

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)

    for qix in tqdm(range(len(train_queries))):

        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # Use a small random subset of tools to avoid OOM on Colab.
        # Always include the gold tool so we can score heads against it.
        all_tool_ids = list(tools.keys())
        distractors = [t for t in all_tool_ids if t != gold_tool_name]
        random.shuffle(distractors)
        tool_ids = [gold_tool_name] + distractors[:TOOLS_PER_PROMPT - 1]
        random.shuffle(tool_ids)  # shuffle so gold is not always at position 0

        putils = PromptUtils(
        tokenizer=tokenizer, 
        doc_ids=tool_ids, 
        dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions 

        # Add your head scoring logic after this line

        # Identify gold tool position in the shuffled subset
        gold_tool_id = map_docname_id[gold_tool_name]

        # Identify query span using the same approach as Part 2
        total_tokens = input_ids.shape[0]
        suffix_len = putils.prompt_suffix_length
        add_text1_len = putils.add_text1_length
        separator_len = len(tokenizer(putils.prompt_seperator, add_special_tokens=False).input_ids)

        after_last_tool = item_spans[-1][1]
        query_start = after_last_tool + separator_len + add_text1_len + separator_len
        query_end = total_tokens - suffix_len

        query_start = min(query_start, total_tokens - 1)
        query_end = max(query_end, query_start + 1)

        # For each (layer, head), compute attention from query tokens -> each tool's tokens.
        # Score a head by whether it ranks the gold tool highest (rank-based scoring).
        # We award 1 point to a head if it places the gold tool at rank 1.
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: [1, num_heads, N, N]
            attn = layer_attn.squeeze(0).cpu().float()  # move to CPU to save GPU RAM

            # Compute per-head mean attention score for every doc
            per_head_doc_scores = torch.zeros(num_heads, len(item_spans))
            for doc_idx, (doc_start, doc_end) in enumerate(item_spans):
                # block: [num_heads, q_len, d_len]
                block = attn[:, query_start:query_end, doc_start:doc_end]
                per_head_doc_scores[:, doc_idx] = block.mean(dim=(-2, -1))

            # Award 1 point to heads that rank the gold tool highest
            gold_scores = per_head_doc_scores[:, gold_tool_id]   # [num_heads]
            max_scores  = per_head_doc_scores.max(dim=1).values   # [num_heads]
            gold_is_top = (gold_scores >= max_scores).float()     # [num_heads]

            head_scores[layer_idx] += gold_is_top.to(device)

            del attn, per_head_doc_scores

        # Free GPU memory after each query
        del attentions, inputs
        torch.cuda.empty_cache()
        gc.collect()

    # TODO: select top heads
    selected_heads = []

    # Flatten head_scores [num_layers, num_heads] and pick top max_heads
    flat_scores = head_scores.view(-1)
    top_indices = torch.topk(flat_scores, k=max_heads).indices

    for idx in top_indices:
        layer_idx = (idx // num_heads).item()
        head_idx  = (idx % num_heads).item()
        selected_heads.append((layer_idx, head_idx))

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads
