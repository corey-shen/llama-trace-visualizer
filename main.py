import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float16)
class RecordData:
    def __init__(self, num_engines=4):
        self.operations = []
        self.operation_counter = 0
        self.num_engines = num_engines
        self.engine_end_times = [0] * num_engines   # Tracking when each engine is free

    def add_operation(self, op_type, input_shape, output_shape, dependencies=None):
        dependencies = dependencies or []

        earliest_start = 0
        if dependencies:    # Calculates earliest possible start time based on when all dependencies finish
            earliest_start = max([self.operations[dep_id]["end_time"] for dep_id in dependencies])

        best_engine = 0     # Find engine with earliest availability
        for i in range(self.num_engines):
            if self.engine_end_times[i] < self.engine_end_times[best_engine]:
                best_engine = i

        start_time = max(earliest_start, self.engine_end_times[best_engine])
        duration = 1.0  # Arbitrary hard coded unit duration
        end_time = start_time + duration

        self.engine_end_times[best_engine] = end_time

        operation_dict = {
            "id": self.operation_counter,
            "type": op_type,
            "inputs": input_shape,
            "outputs": output_shape,
            "dependencies": dependencies,
            "engine": best_engine,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        }
        self.operations.append(operation_dict)
        self.operation_counter += 1
        return self.operation_counter - 1

    def get_operation(self):
        return self.operations


tracker = RecordData()
op_id = tracker.add_operation("layer_norm", [[1, 512, 768]], [[1, 512, 768]], [])

print(tracker.get_operation())


def trace_llama_block(model_id="meta-llama/Llama-3.2-1B", past_seq_len=64): # Sets the number of past tokens to 64 (used in attention caching)
    tracker = RecordData()
    op_ids = {}     # Map each operation to unique IDs

    config = model.config
    hidden_size = config.hidden_size             # Dimension of token embeddings + hidden states
    num_heads = config.num_attention_heads      # Number of attention heads in multi-head self-attention
    head_dim = hidden_size // num_heads         # Dimension of each attention head

    # Dummy inputs to simulate the forward pass through the block
    current_token = torch.zeros(1, 1, hidden_size)  # [batch_size = 1, seq_len = 1, hidden_size]
    past_keys = torch.zeros(1, past_seq_len, hidden_size)   # (N, D) | 1 -> batch dimension
    past_values = torch.zeros(1, past_seq_len, hidden_size)

# === Input LayerNorm ===
    op_ids["input_layernorm"] = tracker.add_operation(
        "Input Layer Norm",
        [list(current_token.shape)],
        [[1, 1, hidden_size]],
        []
    )

# === Self Attention ===
    op_ids["q_proj"] = tracker.add_operation(
        "Query Projection",
        [list(current_token.shape)],
        [[1, 1, hidden_size]],
        [op_ids["input_layernorm"]]
    )

    op_ids["k_proj"] = tracker.add_operation(
        "Key Project",
        [list(past_keys.shape)],
        [[1, past_seq_len, hidden_size]],
        [op_ids["input_layernorm"]]
    )

    op_ids["v_proj"] = tracker.add_operation(
        "Value Projection",
        [list(past_values.shape)],
        [[1, past_seq_len, hidden_size]],
        [op_ids["input_layernorm"]]
    )

# === QK Dot Product ===
    op_ids["QK_dot_product"] = tracker.add_operation(
        "QK Dot-Product",
        [[1, 1, hidden_size], [1, past_seq_len, hidden_size]],
        [[1, 1, past_seq_len]],
        [op_ids["q_proj"], op_ids["k_proj"]]
    )

# === Softmax ===
    op_ids["softmax_max"] = tracker.add_operation(
        "Softmax (Max Value)",
        [[1, 1, past_seq_len]],
        [[1, 1, past_seq_len]],
        [op_ids["QK_dot_product"]]
    )

    op_ids["softmax_exp"] = tracker.add_operation(
        "Softmax (e^x)",
        [[1, 1,past_seq_len]],
        [[1, 1,past_seq_len]],
        [op_ids["softmax_max"]]
    )

    op_ids["softmax_normalization"] = tracker.add_operation(
        "Softmax Normalization",
        [[1, 1, past_seq_len]],
        [[1, 1, past_seq_len]],
        [op_ids["softmax_exp"]]
    )

# === Attention Output ===
    op_ids["attention_output"] = tracker.add_operation(
        "Attention Output",
        [[1, 1, past_seq_len], [1, past_seq_len, hidden_size]], # Input for attention output operation contains attention weights and value vectors
        [[1, 1, hidden_size]],   # singleton dimension, does not affect matrix multiplication
        [op_ids["softmax_normalization"], op_ids["v_proj"]]
    )

# === Output Projection ===
    op_ids["output_proj"] = tracker.add_operation(
        "Output Projection",
        [[1, 1, hidden_size]],
        [[1, 1, hidden_size]],
        [op_ids["attention_output"]]
    )

# === Post Attention LayerNorm ===
    op_ids["post_attention_layernorm"] = tracker.add_operation(
        "Post Attention Layernorm",
        [[1, 1, hidden_size]],
        [[1, 1, hidden_size]],
        [op_ids["output_proj"]]
    )

# === MLP (Feed-Forward)
    op_ids["gate_proj"] = tracker.add_operation(
        "Gate Projection",
        [[1, 1, hidden_size]],
        [[1, 1, hidden_size * 4]],  # 2048 -> 8192
        [op_ids["post_attention_layernorm"]]
    )

    op_ids["up_proj"] = tracker.add_operation(
        "Up Projection",
        [[1, 1, hidden_size]],
        [[1, 1, hidden_size * 4]],
        [op_ids["post_attention_layernorm"]]
    )

    op_ids["siLU_activation"] = tracker.add_operation(
        "siLU Activation",
        [[1, 1, hidden_size * 4]],
        [[1, 1, hidden_size * 4]],
        [op_ids["gate_proj"]]
    )

    op_ids["gate_multiplication"] = tracker.add_operation(
        "Gate Multiplication",
        [[1, 1, hidden_size * 4], [1, 1, hidden_size * 4]],  # SiLU Activation output * Up Projection output
        [[1, 1, hidden_size * 4]],
        [op_ids["siLU_activation"], op_ids["up_proj"]]
    )

    op_ids["down_proj"] = tracker.add_operation(
        "Down Projection",
        [[1, 1, hidden_size * 4]],
        [[1, 1, hidden_size]],      # Projects back to hidden_size
        [op_ids["gate_multiplication"]]
    )

# === KV Cache ===
    op_ids["key_update"] = tracker.add_operation(   # Concatenates existing key cache (past_keys) with the new projected key that is one token longer
        "key Update (KV Cache)",
        [[1, past_seq_len, hidden_size], [1, 1, hidden_size]],      # Existing cache + new key
        [[1, past_seq_len + 1, hidden_size]],      # Updated cache with +1 token
        [op_ids["k_proj"]]      # Depends on key projection
    )

    op_ids["value_update"] = tracker.add_operation(
        "Value Update (KV Cache)",
        [[1, past_seq_len, hidden_size], [1, 1, hidden_size]],      # Existing cache + new value
        [[1, past_seq_len + 1, hidden_size]],       # Update cache with one more token
        [op_ids["v_proj"]]      # Depends on value projection
    )

    return tracker


def generate_perfetto_trace(operation):      # Convert operations to Perfetto trace format JSON
    trace = {
        "traceEvents": []
    }

    for op in operation:
        event = {
            "name": op["type"],     # operation type
            "cat": "compute",       # Category
            "ph": "X",           # Phase | X == complete events with start time and duration
            "ts": op["start_time"] * 1000,       # Timestamp | Convert to microseconds
            "dur": op["duration"] * 1000,        # Duration
            "pid": 1,        # process ID | Since we are modeling a single hardware system, we use 1 for all events
            "tid": op["engine"] + 1,   # Thread ID | Used to represent different compute engines
            "args": {               # Allows you to attach arbitrary metadata to each event
                "inputs": str(op["inputs"]),
                "outputs": str(op["outputs"]),
                "dependencies": str(op["dependencies"])
            }
        }
        trace["traceEvents"].append(event)
    return trace


if __name__ == "__main__":
    # Trace one LLaMA block
    tracker = trace_llama_block()
    operations = tracker.get_operation()

    # Print operation trace
    print("LLaMA Block Operation Trace:")
    print(f"{'ID':<5} {'Type':<25} {'Engine':<8} {'Start':<7} {'End':<7} {'Input Shapes':<45} {'Output Shapes':<45} {'Dependencies':<15}")
    for op in operations:
        print(
            f"{op['id']:<5} {op['type']:<25} {op['engine']:<8} {op['start_time']:<7.2f} {op['end_time']:<7.2f} "
            f"{str(op['inputs']):<45} {str(op['outputs']):<45} {str(op['dependencies']):<15}")

    print(f"\nTotal operations: {len(operations)}")
