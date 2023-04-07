本Repository尝试使用ChatGLM-6B的Transformer模块直接进行Text embedding

参考[mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)和ChatGLM-6B的[P-tuning v2](https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/README.md)方法，尝试在embedding任务上对模型进行微调

数据集：[medi-data](https://github.com/HKUNLP/instructor-embedding#train-instructor)

方法：在需要embedding的文本最后加上prompt和\[MASK\]标记，取\[MASK\]标记对应的输出作为文本的embedding

进行微调前，模型在MTEB的ArguAna、CQADupstackWebmasterRetrieval、CQADupstackAndroidRetrieval任务上的效果和BERT-base-uncased差不多