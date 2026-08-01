import torch

def get_top_examples_with_marked_token(top_values, top_doc_ids, top_positions, all_str_tokens, feature_idx, top_k=4):
    """Same as before, but marks the exact activating token with [[...]]
    instead of just showing surrounding context — tells us whether the
    feature fired on the concept itself vs. adjacent punctuation/tokens."""
    results = []
    for k in range(top_k):
        val = top_values[k, feature_idx].item()
        doc_idx = top_doc_ids[k, feature_idx].item()
        pos = top_positions[k, feature_idx].item()
        if doc_idx < 0:
            continue

        str_tokens = all_str_tokens[doc_idx]
        start_ctx = max(0, pos - 5)
        end_ctx = min(len(str_tokens), pos + 6)

        left = "".join(str_tokens[start_ctx:pos])
        center = str_tokens[pos]
        right = "".join(str_tokens[pos + 1:end_ctx])
        context = f"{left}[[{center}]]{right}"

        results.append((val, context))

    results.sort(key=lambda x: x[0], reverse=True)
    return results

if __name__ == "__main__":
    cached = torch.load("results/steering_candidate_screen.pt")

    top_values = cached["top_values"]
    top_doc_ids = cached["top_doc_ids"]
    top_positions = cached["top_positions"]
    all_str_tokens = cached["all_str_tokens"]

    shortlist = [1492, 754, 3574, 6861, 3030, 2553, 727, 1046]

    for feature_idx in shortlist:
        print(f"\n--- Feature #{feature_idx} ---")
        examples = get_top_examples_with_marked_token(top_values, top_doc_ids, top_positions, all_str_tokens, feature_idx, top_k=4)
        for val, context in examples:
            print(f"  {val:.2f} | {context}")