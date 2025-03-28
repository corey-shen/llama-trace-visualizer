# LLaMA 3.2 Compute Chain Visualization
This project visualizes the computational chain of a single transformer decoder block in the LLaMA 3.2 model, focusing specifically on the autoregressive decoding operations. It breaks down the complex computational graph into individual operations, schedules them across four compute engines, and generates an interactive visualization using Perfetto.
## Overview
The code traces and visualizes:

- Self-attention mechanism (Q/K/V projections, dot products, softmax, attention outputs)
- Feed-forward networks (gate projections, activations, up/down projections)
- Layer normalization operations
- KV-cache updates

Each operation is scheduled to run on one of four compute engines while respecting dependencies between operations.
## Requirements
- Python 3.8+
- PyTorch
- Transformers library from Hugging Face

Install dependencies with:
`pip install -r requirements.txt`

## How to Run

1. Clone this repository
2. Install dependencies
3. Run the main script:

`python main.py`

The script will:

1. Load the LLaMA 3.2 1B model from Hugging Face
2. Trace one transformer decoder block
3. Schedule operations across four compute engines
4. Generate a Perfetto-compatible JSON trace file

## Visualization
To view the generated trace:

1. Open https://ui.perfetto.dev/ in your browser
2. Click "Open trace file"
3. Select the `llama_trace.json` file generated by the script
4. Explore the visualization showing operations distributed across four compute engines

The visualization shows:
- Each operation as a color block
- Time flowing from left to right
- Four parallel tracks representing the compute engines
- Timing relationships and dependencies

## Understanding the Visualization
When you click on any operation in the Perfetto UI, you can see detailed metadata including:

- Input tensor shapes
- Output tensor shapes
- Dependencies on other operations

This helps in understanding the data flow through the transformer block.
## How It Works
The code:

1. Creates a computational graph tracker that records each operation's:

    - Type (e.g., matrix multiplication, layer norm)
    - Input and output tensor shapes
    - Dependencies on previous operations

2. Schedules operations across four compute engines:

    - Respects dependencies (operations can't start until dependencies complete)
    - Assigns each operation to the earliest available engine
    - Tracks timing information
3. Generates a Perfetto-compatible trace file to visualize the execution
## Project Structure

`main.py` - The main script that implements the tracing, scheduling, and visualization

`llama_trace.json` - The generated Perfetto trace file

`requirements.txt` - Required Python packages

## Customization
You can modify parameters like:

- The model size (default is 1B parameters)
- The number of compute engines (default is 4)
- The sequence length for past tokens in KV cache

## Notes on Model Loading
This project requires downloading the LLaMA 3.2 1B model from Hugging Face. If you haven't downloaded it previously, the script will attempt to download it automatically (requires Hugging Face authentication if it's a gated model).