from typing import List, Dict

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, top_k_top_p_filtering
import torch

from mteb import MTEB,DRESModel

from models import embedding_model,tokenizer

def get_embeddings(tokenizer, query):
    tokenized = tokenizer([q+'[MASK]' for q in query], return_tensors="pt", padding=True,truncation=True,max_length=1024)['input_ids'].to(embedding_model.device)
    with torch.no_grad():
        outputs = embedding_model(tokenized)
    return outputs.tolist()
    #     outputs = model.transformer(tokenized)
    # return outputs.last_hidden_state[-1, :, :].tolist()

class MyModel(DRESModel):

    def encode(self,sentences,**kwargs):
        batch_size = 4
        dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
        embeddings = []
        for batch in tqdm(dataloader):
            embeddings.extend(get_embeddings(tokenizer, batch))
        return embeddings

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.encode(queries,**kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.encode(sentences, **kwargs)


get_embeddings(tokenizer, ["I am a sentence"])

evalModel = MyModel(None)
# evaluation = MTEB(task_types=['Retrieval'])
evaluation = MTEB(tasks=['ArguAna'])
evaluation.run(evalModel,output_folder=f'results_finetuned')