# 基于ChatGLM的text embedding实现

**text embedding based on ChatGLM:** This project aims to prefix-tuning ChatGLM-6B with respect to the text-embedding task, so that a single model can be used to complete the text-embedding task and question-answer task at the same time, which reduce the overhead of additional deployment of the embedding model. You can easily use it in langchain and implement a fully localized-deploymented trivia model

## 介绍

本项目尝试在text embedding任务上对ChatGLM-6B进行prefix-tuning微调，从而使用单个模型即可完成text embedding和问答任务，减少额外部署embedding模型的开销，可以轻松接入langchain实现一个全本地化部署的知识问答模型

微调方法：[mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)和ChatGLM-6B的[P-tuning v2](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md)

数据集：[medi-data](https://github.com/HKUNLP/instructor-embedding#train-instructor)

text embedding的实现方式：在prefix-tuning的基础上，在需要embedding的文本最后加上\[MASK\]标记，取transformer模块在\[MASK\]标记对应处的输出作为文本的text embedding

受限于个人的硬件条件，本人只使用了medi-data中约8000个英文样本对模型进行了微调，微调后在MTEB的Arguana任务上的效果比glove.6B.300d稍好，后续可以考虑使用更多数据或中文语料进行进一步的微调

## 使用

### 模型训练

```bash
cd example_with_embeddings
```

tokenization

```bash
python preprocess_and_tokenize.py \
--train_file medi-data/medi-data.json \
--overwrite_cache
--model_name_or_path "THUDM/chatglm-6b-int4" \
--max_source_length 512 \
--per_device_train_batch_size 8
```

训练

```bash
python finetune.py \ 
--dataset_path medi-data/processed \
--model_name_or_path "THUDM/chatglm-6b-int4" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_train_epochs 1 \
--max_steps 1000 \
--save_steps 50 \
--save_total_limit 2 \
--learning_rate 1e-4 \
--remove_unused_columns false \
--logging_steps 1 \
--output_dir output
```

### 模型评估

```bash
python mteb_benchmark.py
```