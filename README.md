# Online-DPO-R1: Unlocking Effective Reasoning Without the PPO Overhead

<div align="center">
  <a href="https://efficient-unicorn-451.notion.site/Online-DPO-R1-Unlocking-Effective-Reasoning-Without-the-PPO-Overhead-1908b9a70e7b80c3bc83f4cf04b2f175">
    <img src="https://www.notion.so/front-static/favicon.ico" alt="Notion Icon">
  </a>
  <br>
  <a href="https://www.notion.so/">Notion Page</a>
</div>


This is the repository for running the Iterative DPO with rule-based rewards. In every iteration, we sample responses from the model and label the rewards using the rule-based method. We then construct the preference pair based on the reward scores for DPO training.

## Introduction

Inspired by the success of Deepseek-R1-Zero and several replications of PPO training which achieve superior performance on mathematical reasoning and demonstrate the “Aha moment” with Supervised Fine-tuning, we are curious about alternative algorithms in RL in this scenario. In this project, we implement rule-based RL from Qwen2.5-MATH-7B-base using iterative DPO and rejection sampling (RAFT), which are efficient and easy to implement. We train the models using the prompt set from the MATH training set and Numina-Math, and evaluate the models on AIME24, AMC23, MATH500, Minerva Math, and OlympiadBench. After several iterations, our models achieve an overall accuracy of 49.7% for DPO after SFT warm-up, 47.0% for DPO starting from the Base Model, and 44.4% for RAFT, compared to 33.9% for the Base Model.

## Requirements

We have two separate environments for running the Iterative DPO.

### Generation
```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# The following code is tested for CUDA12.0-12.2, and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.48.1
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!

pip install antlr4-python3-runtime==4.7.2
pip install sympy==1.12
pip install latex2sympy2==1.9.1
pip install word2number==1.1
```

### Training
```sh
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0
pip install huggingface-hub==0.24.7

pip install transformers==4.42.2
pip install peft==0.7.1  #We do not use peft, but some versions would cause errors.
pip install deepspeed==0.15.4
pip install trl==0.9.6
pip install wandb
```

## Running Online-DPO with numina prompt set:

```
bash run_iter_dpo.sh
```
## Evaluation

We provide the evaluation scripts for all the benchmarks we use, including **AIME24**, **AMC23**, **MATH500**, **OlympiadBench**, and **Minerva_Math**. Please go to ```eval_math``` folder for the detailed instructions.
