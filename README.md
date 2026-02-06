# Fine-Tuning DeepSeek-R1 on Medical Reasoning Dataset

This notebook walks through the process of fine-tuning DeepSeek's R1 distilled model (8B parameters) on a medical reasoning dataset. The goal is to improve the model's ability to handle medical questions with proper chain-of-thought reasoning.

## What's This About?

We're taking the `DeepSeek-R1-Distill-Llama-8B` model and training it on medical Q&A data that includes detailed reasoning steps. The dataset has questions, complex chain-of-thought explanations, and responses - perfect for teaching the model to think through medical problems systematically.

## Setup Requirements

You'll need:
- A GPU (the notebook uses a Tesla T4 on Colab)
- Hugging Face account and token
- Weights & Biases account (optional, for tracking training)
- About 500 medical examples from the dataset (we use a subset to keep training manageable)

## Installation

The notebook starts by installing Unsloth, which is a library that makes LLM fine-tuning faster and more memory-efficient:

```bash
pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## What Happens Step by Step

### 1. Load the Base Model

We load DeepSeek-R1-Distill-Llama-8B in 4-bit quantization to save memory. The model is set up with:
- Max sequence length: 2048 tokens
- 4-bit quantization enabled
- Linear RoPE scaling

### 2. Dataset Preparation

The dataset comes from `FreedomIntelligence/medical-o1-reasoning-SFT`. Each example has:
- **Question**: A medical query or case
- **Complex_CoT**: Chain-of-thought reasoning
- **Response**: The final answer

We only use 500 examples from the training split to keep things fast and manageable.

### 3. Prompt Engineering

The training uses a specific prompt format that encourages step-by-step reasoning:

```
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts...

### Instruction:
You are a medical expert...

### Query:
{question}

### Chain of Thought:
{reasoning_steps}

### Answer:
{response}
```

### 4. LoRA Fine-Tuning

Instead of training the entire 8B parameter model, we use LoRA (Low-Rank Adaptation) which is way more efficient. The configuration:

- Rank (r): 16
- Alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- No dropout
- Gradient checkpointing enabled

### 5. Training Configuration

The actual training uses these settings:
- Batch size: 1 per device
- Gradient accumulation: 4 steps
- Learning rate and other hyperparameters are set in TrainingArguments

Training progress is logged to Weights & Biases if you have it configured.

### 6. Testing the Model

After training, you can test the model on new medical questions. The notebook includes an example about urinary incontinence to see how the model performs with its new medical reasoning abilities.

## Key Files and Outputs

- The fine-tuned LoRA weights can be saved and shared
- Training metrics are tracked in W&B
- You can load the model later for inference or further fine-tuning

## Notes and Tips

- The notebook is set up for Google Colab, so some cells handle Colab-specific stuff (like getting secrets from userdata)
- If you run into memory issues, try reducing the sequence length or batch size
- The 4-bit quantization makes this runnable on free Colab GPUs
- You can adjust the number of training examples by changing `split = "train[:500]"` to use more or fewer samples

## Dataset Citation

The medical dataset used here is from FreedomIntelligence's medical-o1-reasoning-SFT collection, which provides high-quality medical reasoning examples with detailed chain-of-thought annotations.

## Why This Approach?

Chain-of-thought training helps the model not just memorize answers, but actually work through medical problems step by step. This is especially important in healthcare applications where you want to see the reasoning, not just the conclusion.

The combination of DeepSeek's strong base model, medical-specific training data, and efficient LoRA fine-tuning makes this a practical way to build a medical reasoning assistant without needing massive compute resources.
