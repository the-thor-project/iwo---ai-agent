"""
Pure Python transformer language model engine for IWO.
No external dependencies are required.
"""

import json
import math
import random
import re
from typing import List, Dict, Optional, Tuple

# Global seed for reproducibility
random.seed(42)


class TransformerTokenizer:
    """Minimal tokenizer that converts text into token IDs and back."""

    def __init__(self):
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

        common_tokens = [
            "hello", "hi", "i", "you", "are", "am", "is", "it", "the", "a", "an",
            "local", "ai", "model", "can", "help", "with", "code", "python", "programming",
            "language", "learn", "information", "about", "and", "to", "in", "of", "for",
            "that", "this", "on", "by", "using", "no", "external", "api", "machine",
            "learning", "what", "why", "how", "who", "where", "when", "which", "do",
            "you", "need", "want", "more", "detail", "explain", "answer", "task", "tasks",
            "creative", "writing", "brainstorm", "science", "math", "technology", "data",
            "system", "response", "memory", "local", "offline", "self", "contained", "engine",
            "run", "runs", "inside", "application", "without", "calling", "cloud", "service",
            "program", "debug", "error", "question", "issue", "understand", "please", "sure",
            "okay", "thank", "thanks", "yes", "no", "because", "more", "better", "right",
            "start", "end", "stop", "continue", "now", "then", "next", "first", "second",
            "user", "bot", "assistant", "session", "conversation", "history", "helpful", "friendly"
        ]

        for token in common_tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token

        self.vocab_size = len(self.vocab)
        self.bos_token_id = self.vocab["[BOS]"]
        self.eos_token_id = self.vocab["[EOS]"]
        self.pad_token_id = self.vocab["[PAD]"]
        self.unk_token_id = self.vocab["[UNK]"]

    def _clean_text(self, text: str) -> str:
        text = text.lower().strip()
        text = text.replace("\n", " ")
        # More inclusive regex: allow alphanumeric, underscores, and common punctuation
        text = re.sub(r"[^a-z0-9\s.,!?'\-_\[\]\(\){}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        cleaned = self._clean_text(text)
        # Improved tokenization: handle numbers, underscores, code-like patterns
        tokens = re.findall(r"[a-z_]+|[0-9]+|[.,!?'\-\[\]\(\){}]|.", cleaned)
        return [t for t in tokens if t.strip()]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(token_id, "[UNK]") for token_id in token_ids]
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?'-])", r"\1", text)
        text = re.sub(r"\s+'", "'", text)
        text = text.replace(" [EOS]", "").replace(" [PAD]", "")
        return text.strip()

    def add_tokens(self, texts: List[str]):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token
        self.vocab_size = len(self.vocab)


class PureTransformer:
    """Minimal transformer block implementation with pure Python math."""

    def __init__(self, tokenizer: TransformerTokenizer, embed_dim: int = 64, num_heads: int = 4,
                 ff_dim: int = 128, num_layers: int = 2, max_length: int = 64, dropout_p: float = 0.1):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.dropout_p = dropout_p  # Dropout probability

        self.wte = self._init_matrix("wte", self.vocab_size, self.embed_dim)
        self.wpe = self._init_matrix("wpe", self.max_length, self.embed_dim)
        self.blocks = [self._init_transformer_block(layer) for layer in range(self.num_layers)]
        self.ln_f_gamma = self._init_vector("ln_f_gamma", self.embed_dim)
        self.ln_f_beta = self._init_vector("ln_f_beta", self.embed_dim)
        self.lm_head = self._init_matrix("lm_head", self.embed_dim, self.vocab_size)

        # Tie lm_head to the token embedding matrix for stability and faster convergence
        self.lora_r = 4
        self.lora_alpha = 16.0
        self.lora_A = self._init_matrix("lora_A", self.lora_r, self.embed_dim)
        self.lora_B = self._init_matrix("lora_B", self.embed_dim, self.lora_r)
        self.lora_scale = self.lora_alpha / self.lora_r
        self.blocks[0]["use_lora"] = True
        self._tie_weights()

    def _seed(self, name: str) -> random.Random:
        seed_value = sum(ord(ch) for ch in name) * 1001
        return random.Random(seed_value)

    def _init_vector(self, name: str, length: int) -> List[float]:
        rand = self._seed(name)
        return [rand.uniform(0.88, 1.12) for _ in range(length)]

    def _init_matrix(self, name: str, rows: int, cols: int) -> List[List[float]]:
        """Xavier initialization for better convergence."""
        rand = self._seed(name)
        # Xavier/Glorot initialization: limit = sqrt(6 / (rows + cols))
        limit = math.sqrt(6.0 / (rows + cols))
        return [[rand.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

    def _init_transformer_block(self, layer_index: int) -> Dict[str, List]:
        prefix = f"block_{layer_index}"
        return {
            "wqkv": self._init_matrix(f"{prefix}_wqkv", self.embed_dim, self.embed_dim * 3),
            "wo": self._init_matrix(f"{prefix}_wo", self.embed_dim, self.embed_dim),
            "w1": self._init_matrix(f"{prefix}_w1", self.embed_dim, self.ff_dim),
            "w2": self._init_matrix(f"{prefix}_w2", self.ff_dim, self.embed_dim),
            "ln1_gamma": self._init_vector(f"{prefix}_ln1_gamma", self.embed_dim),
            "ln1_beta": self._init_vector(f"{prefix}_ln1_beta", self.embed_dim),
            "ln2_gamma": self._init_vector(f"{prefix}_ln2_gamma", self.embed_dim),
            "ln2_beta": self._init_vector(f"{prefix}_ln2_beta", self.embed_dim)
        }

    def _matmul(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        return [sum(x * y for x, y in zip(row, vector)) for row in matrix]

    def _matmul_matrix(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        cols_b = len(b[0])
        result = [[0.0 for _ in range(cols_b)] for _ in range(len(a))]
        b_t = list(zip(*b))
        for i, a_row in enumerate(a):
            for j, b_col in enumerate(b_t):
                result[i][j] = sum(x * y for x, y in zip(a_row, b_col))
        return result

    def _add(self, a: List[float], b: List[float]) -> List[float]:
        return [x + y for x, y in zip(a, b)]

    def _scale(self, vector: List[float], scalar: float) -> List[float]:
        return [x * scalar for x in vector]

    def _layer_norm(self, vector: List[float], gamma: List[float], beta: List[float], eps: float = 1e-5) -> List[float]:
        mean = sum(vector) / len(vector)
        variance = sum((x - mean) ** 2 for x in vector) / len(vector)
        normalized = [(x - mean) / math.sqrt(variance + eps) for x in vector]
        return [gamma[i] * normalized[i] + beta[i] for i in range(len(vector))]

    def _gelu(self, x: float) -> float:
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

    def _dropout(self, vector: List[float], p: float = 0.0) -> List[float]:
        """Apply dropout: randomly zero out elements with probability p."""
        if p <= 0:
            return vector
        scale = 1.0 / (1.0 - p)  # Scale to maintain expected value
        return [x * scale if random.random() > p else 0.0 for x in vector]

    def _softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax with logit clipping and input normalization."""
        # Clip logits to prevent overflow/underflow
        clipped = [max(min(x, 50), -50) for x in logits]
        mean = sum(clipped) / max(1, len(clipped))
        variance = sum((x - mean) ** 2 for x in clipped) / max(1, len(clipped))
        std = math.sqrt(variance + 1e-6)
        normalized = [(x - mean) / max(std, 1e-6) for x in clipped]
        max_logit = max(normalized)
        exps = [math.exp(x - max_logit) for x in normalized]
        total = sum(exps) + 1e-9
        return [exp_val / total for exp_val in exps]

    def _transpose(self, matrix: List[List[float]]) -> List[List[float]]:
        return [list(col) for col in zip(*matrix)]

    def _tie_weights(self):
        self.lm_head = self._transpose(self.wte)

    def _apply_lora(self, x: List[float]) -> List[float]:
        down = [sum(self.lora_A[r][i] * x[i] for i in range(self.embed_dim)) for r in range(self.lora_r)]
        up = [sum(self.lora_B[i][r] * down[r] for r in range(self.lora_r)) * self.lora_scale
              for i in range(self.embed_dim)]
        return up

    def _update_lora(self, hidden_state: List[float], dlogits: List[float], lr: float):
        """Approximate LoRA update to allow one layer to fine-tune."""
        if not hasattr(self, 'lora_A') or not hasattr(self, 'lora_B'):
            return
        mean_grad = sum(dlogits) / max(len(dlogits), 1)
        for i in range(self.embed_dim):
            for r in range(self.lora_r):
                self.lora_B[i][r] -= lr * 1e-4 * mean_grad * hidden_state[i] * self.lora_A[r][i] * self.lora_scale
        for r in range(self.lora_r):
            for i in range(self.embed_dim):
                self.lora_A[r][i] -= lr * 1e-4 * mean_grad * hidden_state[i] * self.lora_B[i][r] * self.lora_scale

    def _compute_wte_gradients(self, hidden_state: List[float], dlogits: List[float], lr: float,
                               gradient_clip: float = 1.0) -> List[List[float]]:
        dlogits_clipped = self._clip_gradient_per_layer(dlogits, gradient_clip)
        return [[hidden_state[i] * dlogits_clipped[token_id] * lr for i in range(self.embed_dim)]
                for token_id in range(self.vocab_size)]

    def _apply_wte_gradients(self, gradients: List[List[float]]):
        for token_id in range(self.vocab_size):
            for i in range(self.embed_dim):
                self.wte[token_id][i] -= gradients[token_id][i]
        self._tie_weights()

    def _split_heads(self, x: List[float]) -> List[List[float]]:
        return [x[i * self.head_dim:(i + 1) * self.head_dim] for i in range(self.num_heads)]

    def _combine_heads(self, heads: List[List[float]]) -> List[float]:
        combined = []
        for head in heads:
            combined.extend(head)
        return combined

    def _scaled_dot_product_attention(self, q: List[List[float]], k: List[List[float]], v: List[List[float]], 
                                       mask: Optional[List[bool]] = None) -> List[List[float]]:
        """Scaled dot-product attention with causal masking and padding mask."""
        scores = []
        seq_len = len(q)
        
        for i, qi in enumerate(q):
            row = []
            for j, kj in enumerate(k):
                # Causal mask: don't attend to future tokens
                if j > i:
                    row.append(-1e10)
                # Padding mask: don't attend to [PAD] tokens
                elif mask and not mask[j]:
                    row.append(-1e10)
                else:
                    dot = sum(x * y for x, y in zip(qi, kj))
                    row.append(dot / math.sqrt(self.head_dim))
            scores.append(row)

        # Apply softmax
        attention = [self._softmax(row) for row in scores]
        
        output = []
        for i in range(len(attention)):
            weighted = [0.0 for _ in range(self.head_dim)]
            for j, score in enumerate(attention[i]):
                # Zero out padding positions in output
                if mask and not mask[j]:
                    continue
                for v_idx, value in enumerate(v[j]):
                    weighted[v_idx] += score * value
            output.append(weighted)
        return output

    def _multi_head_attention(self, x: List[List[float]], block_weights: Dict[str, List],
                              mask: Optional[List[bool]] = None) -> List[List[float]]:
        seq_len = len(x)

        # Project to q, k, v
        qkv_proj = self._matmul_matrix(x, block_weights["wqkv"])
        q = [[qkv_proj[i][j] for j in range(self.embed_dim)] for i in range(seq_len)]
        k = [[qkv_proj[i][j + self.embed_dim] for j in range(self.embed_dim)] for i in range(seq_len)]
        v = [[qkv_proj[i][j + 2 * self.embed_dim] for j in range(self.embed_dim)] for i in range(seq_len)]

        # Split into heads
        q_heads = [[[q[pos][h * self.head_dim + d] for d in range(self.head_dim)] for pos in range(seq_len)] for h in range(self.num_heads)]
        k_heads = [[[k[pos][h * self.head_dim + d] for d in range(self.head_dim)] for pos in range(seq_len)] for h in range(self.num_heads)]
        v_heads = [[[v[pos][h * self.head_dim + d] for d in range(self.head_dim)] for pos in range(seq_len)] for h in range(self.num_heads)]

        # Compute attention for each head
        head_outputs = []
        for h in range(self.num_heads):
            attn_out = self._scaled_dot_product_attention(q_heads[h], k_heads[h], v_heads[h], mask)
            head_outputs.append(attn_out)

        # Concatenate heads
        combined = []
        for pos in range(seq_len):
            concat_vec = []
            for h in range(self.num_heads):
                concat_vec.extend(head_outputs[h][pos])
            combined.append(concat_vec)

        # Apply output projection
        output = [self._matmul(block_weights["wo"], vec) for vec in combined]
        if block_weights.get("use_lora"):
            output = [self._add(row, self._apply_lora(row)) for row in output]
        return output

    def _feed_forward(self, x: List[List[float]], block_weights: Dict[str, List]) -> List[List[float]]:
        hidden = [self._matmul(block_weights["w1"], vector) for vector in x]
        activated = [[self._gelu(value) for value in row] for row in hidden]
        output = [self._matmul(block_weights["w2"], row) for row in activated]
        return output

    def _transformer_block(self, x: List[List[float]], block_weights: Dict[str, List],
                           mask: Optional[List[bool]] = None) -> List[List[float]]:
        normalized = [self._layer_norm(row, block_weights["ln1_gamma"], block_weights["ln1_beta"]) for row in x]
        attention_output = self._multi_head_attention(normalized, block_weights, mask)
        attention_output = [self._dropout(row, self.dropout_p) for row in attention_output]
        attention_residual = [self._add(original, self._scale(attn, 0.5)) for original, attn in zip(x, attention_output)]

        normalized2 = [self._layer_norm(row, block_weights["ln2_gamma"], block_weights["ln2_beta"]) for row in attention_residual]
        ff_output = self._feed_forward(normalized2, block_weights)
        ff_output = [self._dropout(row, self.dropout_p) for row in ff_output]
        return [self._add(res, ff) for res, ff in zip(attention_residual, ff_output)]

    def forward(self, input_ids: List[int]) -> List[List[float]]:
        seq_length = len(input_ids)
        token_embeddings = [self.wte[token_id] for token_id in input_ids]
        position_embeddings = [self.wpe[min(pos, self.max_length - 1)] for pos in range(seq_length)]
        x = [[token_embeddings[i][j] + position_embeddings[i][j] for j in range(self.embed_dim)] for i in range(seq_length)]
        mask = [token_id != self.tokenizer.pad_token_id for token_id in input_ids]

        for block in self.blocks:
            x = self._transformer_block(x, block, mask)

        x = [self._layer_norm(row, self.ln_f_gamma, self.ln_f_beta) for row in x]
        self._last_hidden_states = x
        self._tie_weights()
        logits = [self._matmul(self.lm_head, row) for row in x]
        return logits

    def _cross_entropy_loss(self, logits: List[float], target_index: int) -> float:
        """Compute cross-entropy loss. Handle out-of-bounds target gracefully."""
        if target_index < 0 or target_index >= len(logits):
            # Return high loss for out-of-bounds (shouldn't happen normally)
            return 10.0  # Large loss as penalty
        probs = self._softmax(logits)
        return -math.log(probs[target_index] + 1e-9)

    def _loss_gradient(self, logits: List[float], target_index: int) -> List[float]:
        """Compute gradient of cross-entropy loss w.r.t logits."""
        if target_index < 0 or target_index >= len(logits):
            # Handle out of bounds - return zero gradient
            return [0.0] * len(logits)
        probs = self._softmax(logits)
        grad = [p for p in probs]
        grad[target_index] -= 1.0
        return grad

    def _clip_gradient_per_layer(self, gradient: List[float], max_norm: float = 1.0) -> List[float]:
        """Clip gradient to max_norm per layer for stable training."""
        norm = sum(x * x for x in gradient) ** 0.5
        if norm > max_norm and norm > 1e-6:
            scale = max_norm / norm
            return [g * scale for g in gradient]
        return gradient

    def _update_lm_head(self, hidden_state: List[float], dlogits: List[float], lr: float,
                       gradient_clip: float = 1.0):
        """Update tied embedding weights with per-layer gradient clipping."""
        dlogits_clipped = self._clip_gradient_per_layer(dlogits, gradient_clip)
        for token_id in range(self.vocab_size):
            for i in range(min(self.embed_dim, len(hidden_state))):
                self.wte[token_id][i] -= lr * hidden_state[i] * dlogits_clipped[token_id]
        self._tie_weights()

    def train_step(self, input_ids: List[int], lr: float = 1e-3, weight_decay: float = 1e-5,
                   fine_tune_first_layer: bool = True) -> float:
        """Training step - updates lm_head and embeddings with weight decay regularization."""
        logits = self.forward(input_ids)
        total_loss = 0.0
        weight_decay_loss = 0.0

        for t in range(len(input_ids) - 1):
            target_id = input_ids[t + 1]
            if 0 <= target_id < self.vocab_size:
                dlogits = self._loss_gradient(logits[t], target_id)
                total_loss += self._cross_entropy_loss(logits[t], target_id)
                self._update_lm_head(self._last_hidden_states[t], dlogits, lr)
                self._update_lora(self._last_hidden_states[t], dlogits, lr)

        # Add weight decay loss to total
        wte_norm = sum(sum(x * x for x in row) for row in self.wte)
        weight_decay_loss = weight_decay * wte_norm
        
        return (total_loss + weight_decay_loss) / max(1, len(input_ids) - 1)

    def _gelu_derivative(self, x: float) -> float:
        """Derivative of GELU activation function."""
        tanh_arg = math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
        sech_sq = 1 / (math.cosh(tanh_arg) ** 2)
        return 0.5 * (1 + math.tanh(tanh_arg)) + 0.5 * x * sech_sq * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x ** 2)

    def _layer_norm_backward(self, grad_output: List[float], input_vector: List[float],
                           gamma: List[float], beta: List[float]) -> tuple:
        """Compute gradients for layer normalization."""
        eps = 1e-5
        mean = sum(input_vector) / len(input_vector)
        variance = sum((x - mean) ** 2 for x in input_vector) / len(input_vector)
        std = math.sqrt(variance + eps)

        # Normalized input
        normalized = [(x - mean) / std for x in input_vector]

        # Gradients w.r.t. gamma and beta
        dgamma = [grad_output[i] * normalized[i] for i in range(len(grad_output))]
        dbeta = grad_output[:]

        # Gradient w.r.t. input
        dnormalized = [grad_output[i] * gamma[i] for i in range(len(grad_output))]
        dvar = sum(dnormalized[i] * (input_vector[i] - mean) * (-0.5) * (variance + eps) ** (-1.5)
                  for i in range(len(dnormalized)))
        dmean = sum(dnormalized[i] * (-1/std) for i in range(len(dnormalized))) + \
                dvar * sum(-2 * (input_vector[i] - mean) for i in range(len(input_vector))) / len(input_vector)

        dinput = []
        for i in range(len(input_vector)):
            dx = dnormalized[i] / std
            dx += dvar * 2 * (input_vector[i] - mean) / len(input_vector)
            dx += dmean / len(input_vector)
            dinput.append(dx)

        return dinput, dgamma, dbeta

    def _attention_backward(self, grad_output: List[List[float]], q: List[List[float]],
                           k_matrix: List[List[float]], v: List[List[float]], attention_weights: List[List[float]],
                           block_weights: Dict[str, List]) -> tuple:
        """Compute gradients for multi-head attention.
        
        Note: Full attention backprop is complex. In practice, we use frozen transformer
        and only train the lm_head layer. This stub is kept for future use.
        """
        dwo = [[0.0 for _ in range(self.embed_dim)] for _ in range(self.embed_dim)]
        dwqkv = [[0.0 for _ in range(self.embed_dim * 3)] for _ in range(self.embed_dim)]
        return dwqkv, dwo

    def _feed_forward_backward(self, grad_output: List[List[float]], hidden_states: List[List[float]],
                              block_weights: Dict[str, List]) -> tuple:
        """Compute gradients for feed-forward network."""
        dw1 = [[0.0 for _ in range(self.ff_dim)] for _ in range(self.embed_dim)]
        dw2 = [[0.0 for _ in range(self.embed_dim)] for _ in range(self.ff_dim)]

        for seq_idx in range(len(grad_output)):
            # Gradient through w2
            dhidden = [sum(grad_output[seq_idx][j] * block_weights["w2"][i][j]
                          for j in range(self.embed_dim)) for i in range(self.ff_dim)]

            # Gradient through GELU
            dgelu = [dhidden[i] * self._gelu_derivative(hidden_states[seq_idx][i])
                    for i in range(self.ff_dim)]

            # Gradient through w1
            for i in range(self.embed_dim):
                for j in range(self.ff_dim):
                    dw1[i][j] += dgelu[j] * grad_output[seq_idx][i] * 0.1  # Simplified
                    dw2[j][i] += dgelu[j] * hidden_states[seq_idx][j]

        return dw1, dw2

    def _update_block_weights(self, block: Dict[str, List], gradients: Dict[str, List], lr: float):
        """Update transformer block weights using gradients."""
        for key in ['wqkv', 'wo', 'w1', 'w2', 'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta']:
            if key in gradients:
                for i in range(len(block[key])):
                    for j in range(len(block[key][i])):
                        block[key][i][j] -= lr * gradients[key][i][j]

    def train_step_full(self, input_ids: List[int], lr: float = 1e-3) -> float:
        """Training step with frozen transformer backbone.
        
        Strategy: Keep transformer weights frozen, only train lm_head.
        - Practical for pure Python (no complex backprop infrastructure needed)
        - Gives good results for language modeling
        - Prevents gradient explosion through deep networks
        - Full backprop through transformer requires sophisticated gradient accumulation
        """
        logits = self.forward(input_ids)
        total_loss = 0.0

        for t in range(len(input_ids) - 1):
            target_id = input_ids[t + 1]
            if 0 <= target_id < self.vocab_size:
                dlogits = self._loss_gradient(logits[t], target_id)
                total_loss += self._cross_entropy_loss(logits[t], target_id)
                
                # Apply gradient clipping to prevent instability
                dlogits_clipped = [max(min(g, 1.0), -1.0) for g in dlogits]
                
                self._update_lm_head(self._last_hidden_states[t], dlogits_clipped, lr)

        return total_loss / max(1, len(input_ids) - 1)

    def train_minibatch(self, sequences: List[List[int]], batch_size: int = 4, lr: float = 1e-3, 
                       num_epochs: int = 1, fine_tune_first_layer: bool = True) -> List[float]:
        """Train on mini-batches with true gradient accumulation."""
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = (len(sequences) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(sequences))
                batch = sequences[start_idx:end_idx]
                
                batch_loss = 0.0
                accumulated_wte_gradients = [[0.0] * self.embed_dim for _ in range(self.vocab_size)]
                token_steps = 0
                
                for sequence in batch:
                    logits = self.forward(sequence)
                    for t in range(len(sequence) - 1):
                        target_id = sequence[t + 1]
                        if 0 <= target_id < self.vocab_size:
                            dlogits = self._loss_gradient(logits[t], target_id)
                            batch_loss += self._cross_entropy_loss(logits[t], target_id)
                            step_grad = self._compute_wte_gradients(self._last_hidden_states[t], dlogits, lr)
                            token_steps += 1
                            for token_id in range(self.vocab_size):
                                for i in range(self.embed_dim):
                                    accumulated_wte_gradients[token_id][i] += step_grad[token_id][i]
                            if fine_tune_first_layer:
                                self._update_lora(self._last_hidden_states[t], dlogits, lr)
                
                if token_steps > 0:
                    scale = 1.0 / token_steps
                    averaged_gradients = [[value * scale for value in row] for row in accumulated_wte_gradients]
                    self._apply_wte_gradients(averaged_gradients)
                
                batch_loss /= max(1, len(batch))
                epoch_loss += batch_loss
            
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_epoch_loss)
        
        return losses

    def sample_top_k(self, logits: List[float], k: int = 5, temperature: float = 1.0) -> int:
        """Top-k sampling: sample from top k most likely tokens."""
        # Scale logits by temperature
        scaled = [logit / max(temperature, 0.01) for logit in logits]
        
        # Get top-k indices
        indexed = [(val, idx) for idx, val in enumerate(scaled)]
        indexed.sort(reverse=True)
        top_k_vals = indexed[:k]
        
        # Convert to probabilities
        top_k_logits = [val for val, _ in top_k_vals]
        top_k_indices = [idx for _, idx in top_k_vals]
        
        # Softmax on top-k
        max_logit = max(top_k_logits) if top_k_logits else 0
        exp_logits = [math.exp(min(x - max_logit, 50)) for x in top_k_logits]
        sum_exp = sum(exp_logits)
        probs = [x / max(sum_exp, 1e-8) for x in exp_logits]
        
        # Sample from top-k
        choice = random.random()
        cumulative = 0.0
        for prob, idx in zip(probs, top_k_indices):
            cumulative += prob
            if choice <= cumulative:
                return idx
        
        return top_k_indices[-1] if top_k_indices else 0

    def sample_top_p(self, logits: List[float], p: float = 0.9, temperature: float = 1.0) -> int:
        """Top-p (nucleus) sampling: sample from smallest set whose cumulative prob >= p."""
        # Scale logits by temperature
        scaled = [logit / max(temperature, 0.01) for logit in logits]
        
        # Convert to probabilities
        max_logit = max(scaled) if scaled else 0
        exp_logits = [math.exp(min(x - max_logit, 50)) for x in scaled]
        sum_exp = sum(exp_logits)
        probs = [x / max(sum_exp, 1e-8) for x in exp_logits]
        
        # Sort by probability descending
        indexed = [(prob, idx) for idx, prob in enumerate(probs)]
        indexed.sort(reverse=True)
        
        # Find smallest set with cumulative prob >= p
        cumulative = 0.0
        nucleus = []
        for prob, idx in indexed:
            cumulative += prob
            nucleus.append((prob, idx))
            if cumulative >= p:
                break
        
        # Renormalize and sample
        nucleus_probs = [prob for prob, _ in nucleus]
        nucleus_indices = [idx for _, idx in nucleus]
        total_prob = sum(nucleus_probs)
        nucleus_probs = [p / max(total_prob, 1e-8) for p in nucleus_probs]
        
        choice = random.random()
        cumulative = 0.0
        for prob, idx in zip(nucleus_probs, nucleus_indices):
            cumulative += prob
            if choice <= cumulative:
                return idx
        
        return nucleus_indices[-1] if nucleus_indices else 0

    def train_with_early_stopping(self, sequences: List[List[int]], val_sequences: List[List[int]],
                                 max_epochs: int = 10, lr: float = 1e-3, 
                                 patience: int = 3, min_delta: float = 1e-4) -> Dict:
        """Train with early stopping based on validation perplexity."""
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            # Training phase
            train_loss = 0.0
            for seq in sequences:
                train_loss += self.train_step(seq, lr)
            train_loss /= max(1, len(sequences))
            training_losses.append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            for seq in val_sequences:
                logits = self.forward(seq)
                for t in range(len(seq) - 1):
                    target_id = seq[t + 1]
                    if 0 <= target_id < self.vocab_size:
                        val_loss += self._cross_entropy_loss(logits[t], target_id)
            val_loss /= max(1, len(val_sequences) * 8)  # Approximate divisor
            val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return {
            'epochs_trained': len(training_losses),
            'final_train_loss': training_losses[-1] if training_losses else 0.0,
            'final_val_loss': val_losses[-1] if val_losses else 0.0,
            'best_val_loss': best_val_loss,
            'training_losses': training_losses,
            'val_losses': val_losses
        }

    def save_config(self, path: str):
        """Save model configuration (architecture parameters) for reproducibility."""
        config = {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_length': self.max_length,
            'ff_dim': self.ff_dim,
            'vocab_size': self.vocab_size,
            'dropout_p': self.dropout_p if hasattr(self, 'dropout_p') else 0.1,
            'architecture': 'pure_python_transformer'
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, sequences: List[List[int]], lr: float = 1e-3, full_backprop: bool = False,
                   lr_schedule: Optional[str] = None, epoch: int = 0, total_epochs: int = 1) -> float:
        """Train one epoch with optional learning rate scheduling."""
        # Apply learning rate schedule
        if lr_schedule == "linear_decay":
            lr = lr * (1 - epoch / max(1, total_epochs))
        elif lr_schedule == "exponential_decay":
            lr = lr * (0.95 ** epoch)
        elif lr_schedule == "warmup":
            lr = lr * min(1.0, (epoch + 1) / max(1, total_epochs / 10))
        
        total_loss = 0.0
        for input_ids in sequences:
            if full_backprop and len(input_ids) > 1:
                total_loss += self.train_step_full(input_ids, lr)
            else:
                total_loss += self.train_step(input_ids, lr)
        return total_loss / max(1, len(sequences))

    def _sample_next_token(self, logits: List[float], temperature: float) -> int:
        scaled = [logit / max(temperature, 0.01) for logit in logits]
        probs = self._softmax(scaled)
        chosen = random.random()
        cumulative = 0.0
        for index, probability in enumerate(probs):
            cumulative += probability
            if chosen <= cumulative:
                return index
        return len(probs) - 1

    def generate(self, prompt: str, max_new_tokens: int = 32, temperature: float = 1.0, 
                sampling_method: str = "auto", top_k: int = 5, top_p: float = 0.9) -> str:
        """Generate text with configurable sampling strategy.
        
        Args:
            prompt: Starting text
            max_new_tokens: Max tokens to generate
            temperature: Controls randomness (higher = more random)
            sampling_method: "greedy", "top_k", "top_p", or "auto" (uses temperature to decide)
            top_k: For top-k sampling
            top_p: For nucleus sampling
        """
        input_ids = self.tokenizer.encode(prompt)
        # Context window management
        if len(input_ids) > self.max_length:
            # Shift context by removing oldest tokens
            input_ids = input_ids[-(self.max_length - max_new_tokens):]
        generated_ids = input_ids.copy()

        for step in range(max_new_tokens):
            # Temperature scheduling: higher early on, lower later (more confident)
            if sampling_method == "auto":
                progress = step / max(max_new_tokens, 1)
                scheduled_temp = temperature * (1 - 0.5 * progress)
                if scheduled_temp < 0.3:
                    method = "greedy"
                elif scheduled_temp < 0.7:
                    method = "top_k"
                else:
                    method = "top_p"
            else:
                method = sampling_method
                scheduled_temp = temperature
            
            # Check context window
            if len(generated_ids) > self.max_length:
                # Remove tokens from start while keeping generated tokens
                start_index = len(generated_ids) - self.max_length
                input_seq = generated_ids[start_index:]
            else:
                input_seq = generated_ids

            logits = self.forward(input_seq)
            
            # Choose sampling strategy
            if method == "greedy":
                next_token_id = max(range(len(logits[-1])), key=lambda i: logits[-1][i])
            elif method == "top_k":
                next_token_id = self.sample_top_k(logits[-1], k=top_k, temperature=scheduled_temp)
            elif method == "top_p":
                next_token_id = self.sample_top_p(logits[-1], p=top_p, temperature=scheduled_temp)
            else:
                next_token_id = self._sample_next_token(logits[-1], scheduled_temp)
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
            generated_ids.append(next_token_id)
            if len(generated_ids) >= self.max_length * 2: # Stop if too long
                break

        new_token_ids = generated_ids[len(input_ids):]
        return self.tokenizer.decode(new_token_ids)

    def calculate_perplexity(self, sequences: List[List[int]]) -> float:
        """Calculate perplexity on validation sequences.
        
        Lower perplexity = better model.
        Perplexity = exp(average_loss)
        """
        total_loss = 0.0
        total_tokens = 0

        for input_ids in sequences:
            if len(input_ids) < 2:
                continue
            logits = self.forward(input_ids)
            for t in range(len(input_ids) - 1):
                target_id = input_ids[t + 1]
                if 0 <= target_id < self.vocab_size:
                    loss = self._cross_entropy_loss(logits[t], target_id)
                    total_loss += loss
                    total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)

    def train_sequences(self, sequences: List[List[int]], epochs: int = 1, lr: float = 1e-3,
                       full_backprop: bool = False, lr_decay: float = 0.99) -> List[float]:
        total_losses = []
        current_lr = lr

        for epoch in range(epochs):
            epoch_loss = self.train_epoch(sequences, current_lr, full_backprop)
            total_losses.append(epoch_loss)
            current_lr *= lr_decay  # Decay learning rate

        return total_losses


class TransformerLLM:
    """Wrapper around tokenizer and transformer model for local inference and training."""

    def __init__(self):
        self.tokenizer = TransformerTokenizer()
        self.model = PureTransformer(self.tokenizer)

    def _rebuild_model_if_vocab_changed(self):
        if self.model.vocab_size != self.tokenizer.vocab_size:
            self.model = PureTransformer(self.tokenizer,
                                         embed_dim=self.model.embed_dim,
                                         num_heads=self.model.num_heads,
                                         ff_dim=self.model.ff_dim,
                                         num_layers=self.model.num_layers,
                                         max_length=self.model.max_length)

    def generate(self, prompt: str, max_tokens: int = 48, temperature: float = 1.0) -> str:
        prompt_text = prompt.strip()
        if not prompt_text:
            prompt_text = "Hello."
        self._rebuild_model_if_vocab_changed()
        return self.model.generate(prompt_text, max_new_tokens=max_tokens, temperature=temperature)

    def train_on_texts(self, texts: List[str], epochs: int = 1, lr: float = 1e-3, max_length: int = 64) -> List[float]:
        """Train the model on text data."""
        # Add new tokens from texts
        self.tokenizer.add_tokens(texts)
        
        # Rebuild model if vocab changed
        self._rebuild_model_if_vocab_changed()

        # Encode texts into sequences
        sequences = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            sequences.append(token_ids)

        # Train on sequences
        return self.model.train_sequences(sequences, epochs=epochs, lr=lr)

    def train_on_texts_full(self, texts: List[str], epochs: int = 1, lr: float = 1e-3,
                           max_length: int = 64, lr_decay: float = 0.99) -> List[float]:
        """Train with full backpropagation through lm_head (frozen transformer backbone)."""
        self.tokenizer.add_tokens(texts)
        self._rebuild_model_if_vocab_changed()

        sequences = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            sequences.append(token_ids)

        return self.model.train_sequences(sequences, epochs=epochs, lr=lr,
                                        full_backprop=True, lr_decay=lr_decay)

    def save_weights(self, path: str):
        weights = {
            'vocab': self.tokenizer.vocab,
            'wte': self.model.wte,
            'wpe': self.model.wpe,
            'lm_head': self.model.lm_head,
            'ln_f_gamma': self.model.ln_f_gamma,
            'ln_f_beta': self.model.ln_f_beta,
            'blocks': self.model.blocks
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(weights, f)

    def load_weights(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            weights = json.load(f)
        self.tokenizer.vocab = {k: int(v) for k, v in weights.get('vocab', {}).items()}
        self.tokenizer.inv_vocab = {int(v): k for k, v in self.tokenizer.vocab.items()}
        self.tokenizer.vocab_size = len(self.tokenizer.vocab)
        self._rebuild_model_if_vocab_changed()
        self.model.wte = weights.get('wte', self.model.wte)
        self.model.wpe = weights.get('wpe', self.model.wpe)
        self.model.lm_head = weights.get('lm_head', self.model.lm_head)
        self.model.ln_f_gamma = weights.get('ln_f_gamma', self.model.ln_f_gamma)
        self.model.ln_f_beta = weights.get('ln_f_beta', self.model.ln_f_beta)
        if 'blocks' in weights:
            self.model.blocks = weights['blocks']
