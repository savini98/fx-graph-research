# fx-graph-research

This experiment runs as follows

1. Runs a warmup run
2. Profiler run (with CUDA Graph enabled)
3. 30 Inference runs to measure average throughput

## Setup Environment
- Need conda
```bash
bash setup_env.sh
```

## Load Models
```bash
bash load_models.sh
```

## Run experiment scripts

Each run will execute the model with and without graph breaks. Outputs include:
- Log file (in `traces/`)
- Throughput CSV file
- PyTorch profiler JSON dump

### Example: Run all experiment scripts

```bash
# Run Phi-4-mini-instruct
bash run_phi_4_mini_instruct.sh

# Run Blenderbot-400M-distill
bash run_blenderbot-400M-distill.sh

# Run Flan-T5-Large
bash run_flan-t5-large.sh

# Run Qwen-Audio-Chat
bash run_qwen_audio_chat.sh

# Run tiny-random-PegasusForCausalLM
bash run_tiny-random-PegasusForCausalLM.sh
```