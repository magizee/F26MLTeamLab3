"""
Lab 3, Part 2: Attention

Last week your GPT-2 ran end to end, but attention was a placeholder that returned zeros,
so every token was blind to the others. This week you'll write attention, plug it in,
and then look inside the trained model to find heads that do specific jobs.
Part 1's functions are provided at the top so everyone starts from working code.

Shapes in code: in lecture each token was a column (d × n). In code we follow GPT-2 and
PyTorch: each token is a row, so the input matrix is (n × d). Same math, just transposed.
Every function says the shapes it expects.

In code each QUERY is a ROW of the attention weights, so softmax goes across each row
(lecture had queries as columns and softmaxed down each column: same thing, transposed).

Functions you'll write, in order (each one has inputs, outputs and an example in its docstring):
    [ ] 1. causal_mask(n)                               which (query, key) pairs are in the future
    [ ] 2. attention(Q, K, V, mask)                     softmax(Q K^T / sqrt(d_k)) V
    [ ] 3. split_heads(x, n_heads)                      (n, d) -> (n_heads, n, d_head)
    [ ] 4. merge_heads(x)                               (n_heads, n, d_head) -> (n, d)
    [ ] 5. multi_head_attention(x, block, n_heads)      GPT-2's full attention sub-layer
    [ ] 6. head_scores(cache, offset)                   find heads that look back `offset` tokens

Already written for you in this file:  all of Part 1's functions (embed ... generate),
    attention_block() (now calls YOUR multi_head_attention), forward_with_cache(),
    plot_attention(), top_heads().
From lab_utils.py (same file as Part 1):  enc (tokenizer), EOS, N_HEADS (= 12), params (GPT-2's weights),
    layer_norm(), gelu(). The test file also uses lab_utils' check(), run_tests(), show_top_k()
    and check_against_gpt2().

How to work:
    1. Put this file and test_part2.py in the same folder as lab_utils.py (from Part 1).
    2. Fill in every "YOUR CODE HERE" in this file, in order.
    3. Check your work:   python test_part2.py         (or e.g.  python test_part2.py mask)
"""
import numpy as np
from lab_utils import enc, EOS, N_HEADS, params, layer_norm, gelu

# ============================================================================
# Part 1 functions (provided, already working)
# ============================================================================

def embed(ids, W_E):
    """
    Look up the embedding vector for each token ID.

    Inputs:
        ids : list of n ints       the token IDs, e.g. [464, 1468, 3797]
        W_E : array (V, d)         embedding matrix; row i is the embedding of token i
    Output:
        array (n, d)               row k is W_E[ids[k]]

    Example:
        W_E = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        embed([2, 0], W_E)   ->   [[6, 7, 8],
                                   [0, 1, 2]]

    Hint: NumPy can index an array with a list of row numbers.
    """
    return W_E[ids]


def add_positions(X, W_pos):
    """
    Add each token's positional embedding: the token at position t gets W_pos[t] added.

    Inputs:
        X     : array (n, d)       token embeddings (output of embed)
        W_pos : array (n_ctx, d)   positional embeddings; row t is for position t (n_ctx = 1024 for GPT-2)
    Output:
        array (n, d)               row t is X[t] + W_pos[t]   (only the first n rows of W_pos are used)

    Example:
        X     = np.array([[1, 1],
                          [1, 1]])
        W_pos = np.array([[0, 1],
                          [2, 3],
                          [4, 5]])
        add_positions(X, W_pos)   ->   [[1, 2],
                                        [3, 4]]
    """
    n = X.shape[0]
    return X + W_pos[:n]


def mlp(x, W_up, b_up, W_down, b_down):
    """
    The MLP (feed-forward layer): expand to 4d, apply gelu, shrink back to d.
    Each token's row is processed on its own.

    Inputs:
        x      : array (n, d)      the (layer-normed) residual stream
        W_up   : array (d, 4d)     b_up   : array (4d,)
        W_down : array (4d, d)     b_down : array (d,)
    Output:
        array (n, d)               gelu(x @ W_up + b_up) @ W_down + b_down     (this is ΔE_MLP)

    Example (d = 1, so 4d would be 4, but any hidden size works):
        x      = np.array([[1.0]])
        W_up   = np.array([[2.0, -1.0]]);   b_up   = np.array([0.0, 0.0])
        W_down = np.array([[1.0], [1.0]]);  b_down = np.array([0.5])
        hidden = gelu([[2.0, -1.0]]) = [[1.9546, -0.1588]]
        mlp(x, W_up, b_up, W_down, b_down)   ->   [[2.2958]]        (1.9546 - 0.1588 + 0.5)

    Use the provided gelu() from lab_utils.
    """
    hidden = gelu(x @ W_up + b_up)      # (n, 4d)
    return hidden @ W_down + b_down     # (n, d)


def transformer_block(x, block):
    """
    One transformer block, applied to the residual stream.

    Inputs:
        x     : array (n, d)       the residual stream going into this block
        block : dict               this block's weights. Keys:
                                   "ln1_g", "ln1_b"   layer norm before attention
                                   "ln2_g", "ln2_b"   layer norm before the MLP
                                   "W_up", "b_up", "W_down", "b_down"   MLP weights
                                   (plus attention weights, which attention_block uses)
    Output:
        array (n, d)               the residual stream after this block (same shape as x)

    Steps (GPT-2 applies layer_norm BEFORE each sub-layer, and each sub-layer ADDS to x):
        1. x = x + attention_block(layer_norm(x, block["ln1_g"], block["ln1_b"]), block)
        2. x = x + mlp(layer_norm(x, block["ln2_g"], block["ln2_b"]),
                       block["W_up"], block["b_up"], block["W_down"], block["b_down"])
        3. return x

    Sanity check: if attention and the MLP both returned all zeros, the output would equal x.
    """
    x = x + attention_block(layer_norm(x, block["ln1_g"], block["ln1_b"]), block)
    x = x + mlp(layer_norm(x, block["ln2_g"], block["ln2_b"]),
                block["W_up"], block["b_up"], block["W_down"], block["b_down"])
    return x


def unembed(x, W_E):
    """
    Turn each vector back into a score (logit) for every token in the vocabulary.
    GPT-2 reuses the embedding matrix for this ("weight tying"): the unembedding matrix is W_E transposed.

    Inputs:
        x   : array (n, d)         the final residual stream
        W_E : array (V, d)         the embedding matrix
    Output:
        array (n, V)               out[i, j] = dot product of x[i] with W_E[j]
                                   = how well token j fits after position i

    Example:
        x   = np.array([[1, 2, 3]])
        W_E = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 1]])
        unembed(x, W_E)   ->   [[1, 2, 6]]
    """
    return x @ W_E.T


def forward(ids, params):
    """
    The whole model: token IDs in, logits out.

    Inputs:
        ids    : list of n ints    token IDs (from enc.encode)
        params : dict              all of GPT-2's weights (already loaded for you):
                                   params["W_E"], params["W_pos"], params["blocks"] (list of 12 dicts),
                                   params["lnf_g"], params["lnf_b"] (final layer norm)
    Output:
        array (n, V)               row i = logits for the token that comes AFTER ids[i]
                                   (the last row is the prediction for the next token)

    Steps:
        1. x = embed(ids, params["W_E"])
        2. x = add_positions(x, params["W_pos"])
        3. for each block in params["blocks"]:  x = transformer_block(x, block)
        4. x = layer_norm(x, params["lnf_g"], params["lnf_b"])
        5. return unembed(x, params["W_E"])

    Example:
        forward(enc.encode("The old cat"), params).shape   ->   (3, 50257)
    """
    x = embed(ids, params["W_E"])
    x = add_positions(x, params["W_pos"])
    for block in params["blocks"]:
        x = transformer_block(x, block)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return unembed(x, params["W_E"])


def softmax(z, T=1.0):
    """
    Softmax with temperature, applied along the LAST axis:   p_i = exp(z_i / T) / sum_j exp(z_j / T)

    Inputs:
        z : array (k,) or (m, k)   logits; for a 2D array, each ROW gets its own softmax
        T : float > 0              temperature (higher = flatter, lower = sharper)
    Output:
        array, same shape as z     probabilities; each row sums to 1

    Examples:
        softmax(np.array([4.0, 2.0, 1.0]), T=2)     ->   [0.629, 0.231, 0.140]
        softmax(np.array([4.0, 2.0, 1.0]), T=0.5)   ->   [0.980, 0.018, 0.002]
        softmax(np.array([[1.0, 1.0],
                          [0.0, np.log(3)]]))       ->   [[0.50, 0.50],
                                                          [0.25, 0.75]]
        softmax(np.array([1000.0, 1001.0]))         ->   [0.269, 0.731]   (no overflow / nan)

    Hints: convert first with np.asarray(z, dtype=np.float64), divide by T, and subtract
           z.max(axis=-1, keepdims=True) before np.exp so big logits don't overflow.
    """
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)    # same answer, but no overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sample(probs, greedy=False, rng=None):
    """
    Pick the next token ID from a probability distribution.

    Inputs:
        probs  : array (V,)        probabilities over the vocabulary (sums to 1)
        greedy : bool              True  -> always return the most likely token
                                   False -> draw a random token, each with probability probs[i]
        rng    : np.random.Generator or None   (already handled for you on the line below)
    Output:
        int                        the chosen token ID

    Examples:
        sample(np.array([0.1, 0.6, 0.3]), greedy=True)   ->   1
        sample(np.array([0.1, 0.6, 0.3]))                ->   1 about 60% of the time,
                                                              2 about 30%, 0 about 10%

    Hints: np.argmax; rng.choice(len(probs), p=probs). Return a plain int, e.g. int(...).
    """
    rng = rng or np.random.default_rng()
    if greedy:
        return int(np.argmax(probs))
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(prompt, max_new_tokens=20, T=1.0, greedy=False):
    """
    Generate text one token at a time.

    Inputs:
        prompt         : str       the starting text
        max_new_tokens : int       the most tokens to add
        T              : float     temperature passed to softmax
        greedy         : bool      passed to sample
    Output:
        str                        the prompt followed by the generated text

    Steps:
        1. ids = enc.encode(prompt)
        2. repeat max_new_tokens times:
             logits  = forward(ids, params)
             probs   = softmax(logits[-1], T)          (only the LAST row predicts the next token)
             next_id = sample(probs, greedy)
             if next_id == EOS: stop
             ids.append(next_id)
        3. return enc.decode(ids)

    Example:
        generate("The old cat sat on the", max_new_tokens=3, greedy=True)
            ->   "The old cat sat on the" + 3 more tokens   (the exact words depend on the model)
    """
    ids = enc.encode(prompt)
    for _ in range(max_new_tokens):
        logits = forward(ids[-params["W_pos"].shape[0]:], params)   # keep within the context window
        probs = softmax(logits[-1], T)
        next_id = sample(probs, greedy=greedy)
        if next_id == EOS:
            break
        ids.append(next_id)
    return enc.decode(ids)


# Every layer's attention weights from the most recent forward_with_cache() call.
ATTN_CACHE = []
_recording = [False]    # True only while forward_with_cache() is running


def attention_block(x, block):
    """
    The attention sub-layer that transformer_block() calls. In Part 1 this was a placeholder
    returning zeros; now it calls YOUR multi_head_attention() below.

    Inputs:
        x     : array (n, d)       the layer-normed residual stream
        block : dict               this block's weights (multi_head_attention uses W_qkv, b_qkv, W_O, b_O)
    Output:
        array (n, d)               ΔE_attn, which transformer_block adds to the residual stream

    While forward_with_cache() is running, it also saves this layer's attention weights.
    """
    out, weights = multi_head_attention(x, block)
    if _recording[0]:
        ATTN_CACHE.append(weights)
    return out


def forward_with_cache(ids):
    """
    Runs the whole model AND collects every layer's attention weights, so you can look inside.

    Inputs:
        ids : list of n ints       token IDs (from enc.encode)
    Output (a tuple):
        logits : array (n, V)      same as forward(ids, params)
        cache  : list with one array per layer (12 for GPT-2), each (n_heads, n, n)
                 cache[layer][head, i, j] = how much token i attends to token j in that head
    """
    ATTN_CACHE.clear()
    _recording[0] = True
    try:
        logits = forward(ids, params)
    finally:
        _recording[0] = False
    return logits, list(ATTN_CACHE)

# ============================================================================
# 1. The causal mask
# ============================================================================

def causal_mask(n):
    """
    Which (query, key) pairs are NOT allowed: a token may only look at itself and earlier tokens.

    Inputs:
        n : int                    number of tokens
    Output:
        bool array (n, n)          mask[i, j] = True  if j > i  (key j is in query i's future -> block it)
                                   mask[i, j] = False if j <= i (allowed)

    Example:
        causal_mask(3)   ->   [[False,  True,  True],
                               [False, False,  True],
                               [False, False, False]]

    Hint: np.triu(..., k=1)
    """
    # YOUR CODE HERE
    return np.triu(np.ones((n, n), dtype=bool), k=1)

# ============================================================================
# 2. Scaled dot-product attention:   softmax(Q K^T / sqrt(d_k)) V
# ============================================================================

def attention(Q, K, V, mask=True):
    """
    Scaled dot-product attention:   softmax(Q Kᵀ / sqrt(d_k)) V
    Row i of Q is token i's query, row j of K is token j's key.

    Inputs:
        Q : array (n, d_k)         queries   } also accept a leading heads dimension:
        K : array (n, d_k)         keys      }   (n_heads, n, d_k), so all heads run at once
        V : array (n, d_v)         values    }
        mask : bool                if True, apply the causal mask (no looking at future tokens)
    Output (a tuple):
        output  : array (n, d_v)   row i = weighted average of the value rows = ΔE for token i
        weights : array (n, n)     weights[i, j] = how much query i attends to key j; each ROW sums to 1
        (with a heads dimension: output (n_heads, n, d_v), weights (n_heads, n, n))

    Steps:
        1. scores = Q @ Kᵀ / sqrt(d_k)          use K.swapaxes(-1, -2) instead of K.T so heads work
        2. if mask: scores = np.where(causal_mask(n), -np.inf, scores)
        3. weights = softmax(scores)            softmax over the last axis = across each query's row
        4. output = weights @ V

    Example (d_k = 2):
        Q = K = V = np.array([[1.0, 0.0],
                              [0.0, 1.0]])
        scores  = [[0.707, 0.000],      ->  masked:  [[0.707,  -inf],
                   [0.000, 0.707]]                    [0.000, 0.707]]
        output, weights = attention(Q, K, V)
        weights   ->   [[1.000, 0.000],
                        [0.330, 0.670]]
        output    ->   [[1.000, 0.000],
                        [0.330, 0.670]]
    """
    # YOUR CODE HERE
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(Q.shape[-1])
    if mask:
        scores = np.where(causal_mask(Q.shape[-2]), -np.inf, scores)
    weights = softmax(scores)
    return weights @ V, weights

# ============================================================================
# 3. Multiple heads  (GPT-2: 12 heads, each gets a 64-dim slice of Q, K, V)
# ============================================================================

def split_heads(x, n_heads):
    """
    Cut each token's vector into n_heads equal slices, one slice per head.

    Inputs:
        x       : array (n, d)
        n_heads : int              must divide d evenly; d_head = d // n_heads
    Output:
        array (n_heads, n, d_head) out[h] = columns h*d_head to (h+1)*d_head of x

    Example (n = 2 tokens, d = 4, 2 heads):
        x = np.array([[0, 1, 2, 3],
                      [4, 5, 6, 7]])
        split_heads(x, 2)   ->   [[[0, 1],        head 0
                                   [4, 5]],
                                  [[2, 3],        head 1
                                   [6, 7]]]

    Hint: reshape to (n, n_heads, d_head), then transpose(1, 0, 2).
    """
    # YOUR CODE HERE
    n, d = x.shape
    return x.reshape(n, n_heads, d // n_heads).transpose(1, 0, 2)


def merge_heads(x):
    """
    The inverse of split_heads: put the heads' slices back side by side.

    Inputs:
        x : array (n_heads, n, d_head)
    Output:
        array (n, n_heads * d_head)

    Example:
        merge_heads(np.array([[[0, 1], [4, 5]],
                              [[2, 3], [6, 7]]]))   ->   [[0, 1, 2, 3],
                                                          [4, 5, 6, 7]]
        merge_heads(split_heads(x, n_heads)) should give back x exactly.
    """
    # YOUR CODE HERE
    n_heads, n, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n, n_heads * d_head)


def multi_head_attention(x, block, n_heads=N_HEADS):
    """
    GPT-2's full attention sub-layer: 12 heads in parallel, then an output projection.

    Inputs:
        x       : array (n, d)     the (layer-normed) residual stream
        block   : dict             uses block["W_qkv"] (d, 3d), block["b_qkv"] (3d,),
                                        block["W_O"] (d, d),    block["b_O"] (d,)
        n_heads : int              12 for GPT-2
    Output (a tuple):
        output  : array (n, d)     ΔE_attn, the update added to the residual stream
        weights : array (n_heads, n, n)   every head's attention weights

    Steps:
        1. qkv = x @ block["W_qkv"] + block["b_qkv"]       (n, 3d): Q, K, V side by side
        2. Q, K, V = np.split(qkv, 3, axis=-1)            each (n, d)
        3. split each into heads                           each (n_heads, n, d_head)
        4. out, weights = attention(Q, K, V, mask=True)    all heads at once
        5. out = merge_heads(out) @ block["W_O"] + block["b_O"]
        6. return out, weights

    Example (GPT-2, 6 tokens):
        x.shape = (6, 768)   ->   output.shape = (6, 768),   weights.shape = (12, 6, 6)
    """
    # YOUR CODE HERE
    qkv = x @ block["W_qkv"] + block["b_qkv"]
    Q, K, V = np.split(qkv, 3, axis=-1)
    Q = split_heads(Q, n_heads)
    K = split_heads(K, n_heads)
    V = split_heads(V, n_heads)
    out, weights = attention(Q, K, V, mask=True)
    out = merge_heads(out) @ block["W_O"] + block["b_O"]
    return out, weights

# ============================================================================
# 4. Finding heads that do specific jobs
# ============================================================================

def head_scores(cache, offset):
    """
    Measures, for every head in every layer, how much it looks exactly `offset` tokens back.
    A score near 1 means that head's job is "look back `offset` tokens"; near 0 means it never does.

    Inputs:
        cache  : list of n_layers arrays, each (n_heads, n, n)   attention weights, from forward_with_cache()
                 cache[layer][head, i, j] = how much query token i attends to key token j
        offset : int               how far back to look (1 = the previous token, 0 = the token itself)
    Output:
        array (n_layers, n_heads)  scores[layer, head] = the AVERAGE of cache[layer][head, i, i - offset]
                                   over every query position i from offset to n-1

    Example (one layer, two heads, 4 tokens):
        head 0 puts all its weight on the previous token, head 1 puts all its weight on itself
        head_scores(cache, offset=1)   ->   [[1.0, 0.0]]
        head_scores(cache, offset=0)   ->   [[0.25, 1.0]]   (head 0 only looks at itself at position 0)

    Hint: for one layer's w (n_heads, n, n):  i = np.arange(offset, n);  w[:, i, i - offset]
          picks those entries for every head at once, shape (n_heads, n - offset). Average over the last axis.
    """
    # YOUR CODE HERE
    scores = []
    for w in cache:
        i = np.arange(offset, w.shape[-1])
        scores.append(w[:, i, i - offset].mean(axis=-1))
    return np.asarray(scores)


# Provided helpers for section 4
def plot_attention(prompt, layer, head):
    """
    Draws one head's attention weights as a heatmap (needs matplotlib).

    Inputs:
        prompt : str               the text to run through the model
        layer  : int               which block, 0-11
        head   : int               which head in that block, 0-11
    Output:
        None (shows a plot). Row = the token doing the attending (query),
        column = the token being attended to (key). Darker = more attention.

    Example:
        plot_attention("The old cat sat on the mat.", layer=4, head=11)
    """
    import matplotlib.pyplot as plt
    ids = enc.encode(prompt)
    toks = [enc.decode([i]) for i in ids]
    _, cache = forward_with_cache(ids)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cache[layer][head], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(toks)), toks, rotation=90)
    ax.set_yticks(range(len(toks)), toks)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending)")
    ax.set_title(f"Layer {layer}, head {head}")
    plt.tight_layout()
    plt.show()


def top_heads(scores, k=5):
    """
    Prints the k highest-scoring (layer, head) pairs from head_scores().

    Inputs:
        scores : array (n_layers, n_heads)   output of head_scores()
        k      : int                         how many to print
    Output:
        None (prints lines like "layer  4, head 11: 0.85")
    """
    flat = np.argsort(scores.ravel())[::-1][:k]
    for layer, head in zip(*np.unravel_index(flat, scores.shape)):
        print(f"  layer {layer:2d}, head {head:2d}: {scores[layer, head]:.2f}")

if __name__ == "__main__":
    print("To check your functions, run:  python test_part2.py")
    # Once attention works, uncomment to see a heatmap of one head:
    plot_attention("The old cat sat on the mat because it was tired.", layer=4, head=11)
