# AirLLM Test

A simple test project for benchmarking LLM inference performance using [airllm](https://github.com/lyogavin/airllm) with MLX (Apple Silicon optimized).

## Overview

This project demonstrates how to use airllm to run large language models on Apple Silicon with MLX, and includes performance benchmarking metrics such as:
- **Time to first token** - Measures the latency before the first token is generated
- **Tokens per second** - Calculates the generation throughput

## Model

Currently configured to use:
- **Model**: `mlx-community/Meta-Llama-3.1-405B-Instruct-8bit`
- **Compression**: None (8-bit quantization)

## Requirements

- Python 3.8+
- Apple Silicon Mac (M1/M2/M3) for MLX support
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/demersaj/airLLM-test.git
   cd air-llm-test
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load the Llama 3.1 405B Instruct model (8-bit quantized)
2. Tokenize the input text
3. Generate tokens and measure performance metrics
4. Display the generation output and performance statistics

## Performance Metrics

The script measures and reports:
- **Time to first token**: Latency in milliseconds before the first token is generated
- **Total generation time**: Total time taken for the full generation
- **Generated tokens**: Number of tokens generated
- **Tokens per second**: Throughput metric

## Configuration

You can modify the following in `main.py`:
- **Model**: Change the `from_pretrained()` model path
- **Compression**: Adjust quantization (currently set to `'None'`)
- **Input text**: Modify the `input_text` variable
- **Max tokens**: Change `max_new_tokens` parameter
- **Max length**: Adjust `MAX_LENGTH` for input tokenization

## Dependencies

- `mlx-lm`: MLX language model library for Apple Silicon
- `airllm`: AirLLM inference framework
- `torch`: PyTorch (dependency of airllm)

## Notes

- The model will be downloaded on first run (can be several GB)
- Performance will vary based on your Mac's hardware
- MLX is optimized for Apple Silicon and will not work on Intel Macs or other platforms

## License

[Add your license here]
