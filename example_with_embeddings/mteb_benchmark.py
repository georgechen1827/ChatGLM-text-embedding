from typing import List, Dict

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, top_k_top_p_filtering
import torch

from mteb import MTEB,DRESModel

MODEL = "THUDM/chatglm-6b-int4"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,truncation_side="left")#,cls_token="<cls>",sep_token="<sep>")
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).half().cuda()
model = model.eval()

embedding = model.transformer.word_embeddings
embedding_dense = torch.nn.Linear(embedding.num_embeddings, embedding.embedding_dim, bias=False)
embedding_dense.weight = torch.nn.Parameter(embedding.weight.T)
embedding_dense = embedding_dense.cuda()

'''
'{}\n这段话的主题是[MASK]'
"[Round 0]\n问：{}\n答：这段话的检索主题是[MASK]"
'''
corpus_prompt = '{}\n这段话的主题是[MASK]'
query_prompt = '{}\n这段话的主题是[MASK]'
def get_embeddings(tokenizer, query, prompt=corpus_prompt):
    tokenized = tokenizer([prompt.format(q) for q in query], return_tensors="pt", padding=True,truncation=True,max_length=1024)['input_ids'].to(model.device)
    with torch.no_grad():
        outputs = model.transformer(tokenized).last_hidden_state[-1]
        outputs = model.lm_head(outputs)
        outputs[:, model.config.pad_token_id] = float("-inf")
        outputs = top_k_top_p_filtering(outputs, top_k=50, top_p=0.7)
        outputs = torch.softmax(outputs, dim=-1)
        # outputs = torch.softmax(outputs, dim=-1)
        #     scores = scores / temperature
        # # Top-p/top-k filtering
        # # Sample
        # probs = F.softmax(next_token_logscores, dim=-1)
        outputs = embedding_dense(outputs)
    return outputs.tolist()
    #     outputs = model.transformer(tokenized)
    # return outputs.last_hidden_state[-1, :, :].tolist()

class MyModel(DRESModel):

    def encode(self,sentences,prompt=corpus_prompt,**kwargs):
        batch_size = 2
        dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
        embeddings = []
        for batch in tqdm(dataloader):
            embeddings.extend(get_embeddings(tokenizer, batch,prompt))
        return embeddings

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.encode(queries, query_prompt,**kwargs)

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
        return self.encode(sentences, corpus_prompt, **kwargs)


get_embeddings(tokenizer, ["I am a sentence"])

evalModel = MyModel(None)
# evaluation = MTEB(task_types=['Retrieval'])
evaluation = MTEB(tasks=['ArguAna'])
evaluation.run(evalModel,output_folder=f'results_gmask')