This folder contains the Multitask Embeddings Data with Instructions (MEDI) for the paper: One Embedder, Any Task: Instruction-Finetuned Text Embeddings.

It contains the follow file:
- medi-data.json
	# Training Examples: 1,435,000
- README.txt

The MEDI data consists of a collection of 330 datasets from Super-NI(Super-NaturalInstructions), sentence-transformer embedding training data, and KILT, spanning a wide range of domains and tasks.

If you use the dataset, please cite the following papers including Su et al., 2022, Wang et al., 2022, Petroni et al., 2021 and sentence transformer embedding training data at https://huggingface.co/datasets/sentence-transformers/embedding-training-data.

@inproceedings{INSTRUCTOR,
  title={One Embedder, Any Task: Instruction-Finetuned Text Embeddings},
  author={Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, Tao Yu},
  url={https://arxiv.org/abs/2212.09741},
  year={2022},
}

@inproceedings{wang2022super,
  title={Super-naturalinstructions: generalization via declarative instructions on 1600+ tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and Mirzaei, Amirreza and Arunkumar, Anjana and Ashok, Arjun and Dhanasekaran, Arut Selvan and Naik, Atharva and Stap, David and others},
  year={2022},
  organization={EMNLP}
}

@article{petroni2020kilt,
  title={KILT: a benchmark for knowledge intensive language tasks},
  author={Petroni, Fabio and Piktus, Aleksandra and Fan, Angela and Lewis, Patrick and Yazdani, Majid and De Cao, Nicola and Thorne, James and Jernite, Yacine and Karpukhin, Vladimir and Maillard, Jean and others},
  journal={arXiv preprint arXiv:2009.02252},
  year={2020}
}


