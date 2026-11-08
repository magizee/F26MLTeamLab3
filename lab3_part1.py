"""
Lab 3, Part 1: Build GPT-2 (everything except attention)

This week you'll write the forward pass of a real LLM, GPT-2 small, in plain NumPy:
tokens -> embeddings -> transformer blocks -> logits -> softmax -> next token.

Attention is switched off (it adds zero). Your model will run end to end, but it can't
connect words to each other yet. Next week you'll write attention.

Shapes in code: in lecture each token was a column (d × n). In code we follow GPT-2 and
PyTorch: each token is a row, so the input matrix is (n × d). Same math, just transposed.
Every function says the shapes it expects.

GPT-2 small: d = 768, 12 blocks, 12 attention heads, vocabulary V = 50,257, context window 1,024 tokens.

Functions you'll write, in order (each one has inputs, outputs and an example in its docstring):
    [ ] 1. embed(ids, W_E)                              token IDs -> embedding rows
    [ ] 2. add_positions(X, W_pos)                      add positional embeddings
    [ ] 3. mlp(x, W_up, b_up, W_down, b_down)           the feed-forward layer
    [ ] 4. transformer_block(x, block)                  one block on the residual stream
    [ ] 5. unembed(x, W_E)                              vectors -> logits
    [ ] 6. forward(ids, params)                         the whole model: IDs -> logits
    [ ] 7. softmax(z, T)                                logits -> probabilities, with temperature
    [ ] 8. sample(probs, greedy)                        probabilities -> one token ID
    [ ] 9. generate(prompt, max_new_tokens, T, greedy)  the token-by-token loop

Already written for you:  explore_tokens(), attention_block() (a placeholder that returns zeros),
and from lab_utils: layer_norm(), gelu(), enc (the tokenizer), params (GPT-2's weights), EOS.

How to work:
    1. Fill in every "YOUR CODE HERE" in this file.
    2. Check your work:   python test_part1.py         (or e.g.  python test_part1.py softmax)
    3. Running this file itself (python lab3_part1.py) shows how GPT-2 splits text into tokens.
"""
import numpy as np
from lab_utils import enc, EOS, N_HEADS, params, layer_norm, gelu

# ============================================================================
# 1. Tokens  (nothing to write: run this file and look at how text gets split)
# ============================================================================

def explore_tokens():
    for text in ["The old cat sat on the mat.", "cat", " cat", "unbelievably", "strawberry", " Vayun"]:
        ids = enc.encode(text)
        print(f"{text!r:32} -> {ids}  {[enc.decode([i]) for i in ids]}")

# ============================================================================
# 2. Embeddings
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
    # YOUR CODE HERE
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
    # YOUR CODE HERE
    r = X.shape[0]
    W_pos = W_pos[:r]
    return X + W_pos
# ============================================================================
# 3. The MLP
# ============================================================================

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
    # YOUR CODE HERE
    return gelu(x @ W_up + b_up) @W_down + b_down

# ============================================================================
# 4. A transformer block
# ============================================================================

# Attention is a PLACEHOLDER until Part 2. Don't change this.
def attention_block(x, block):
    """PLACEHOLDER until Part 2: attention adds nothing (ΔE_attn = 0)."""
    return np.zeros_like(x)


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
    # YOUR CODE HERE
    x= x + attention_block(layer_norm(x, block["ln1_g"], block["ln1_b"]), block)
    x = x + mlp(layer_norm(x, block["ln2_g"], block["ln2_b"]), block["W_up"], block["b_up"], block["W_down"], block["b_down"])
    return x
# ============================================================================
# 5. Unembedding
# ============================================================================

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
    # YOUR CODE HERE
    return x @ W_E.T


# ============================================================================
# 6. The full forward pass
# ============================================================================

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
    # YOUR CODE HERE
    x = embed(ids, params["W_E"])
    x = add_positions(x, params["W_pos"])
    for block in params["blocks"]:  
        x = transformer_block(x, block)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return unembed(x, params["W_E"])

# ============================================================================
# 7. Softmax with temperature:   p_i = exp(z_i / T) / sum_j exp(z_j / T)
# ============================================================================

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
    # YOUR CODE HERE
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)

# ============================================================================
# 8. Picking the next token
# ============================================================================

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
    # YOUR CODE HERE
    if greedy:
        return int(np.argmax(probs))
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
    # YOUR CODE HERE
    ids = enc.encode(prompt)
    for i in range(max_new_tokens):
        logits = forward(ids, params)
        probs = softmax(logits[-1], T)
        next_id = sample(probs, greedy)
        if next_id == EOS:
            break
        ids.append(next_id)
    return enc.decode(ids)


if __name__ == "__main__":
    explore_tokens()
    print("\nTo check your functions, run:  python test_part1.py")
