from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

from mteb import MTEB

MODEL = "../model-int4" #"THUDM/chatglm-6b-int4"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,truncation_side="left")#,cls_token="<cls>",sep_token="<sep>")
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).half().cuda()
model = model.eval()

def get_embeddings(tokenizer, query):
    tokenized = tokenizer([q+'\n这段话的主题是[MASK]' for q in query], return_tensors="pt", padding=True,truncation=True,max_length=2048)['input_ids'].to(model.device)
    with torch.no_grad():
        outputs = model.transformer(tokenized)
    return outputs.last_hidden_state[-1, :, :].tolist()

class MyModel():
    def __init__(self, get_embeddings=get_embeddings):
        self.get_embeddings = get_embeddings

    def encode(self,sentences,batch_size=32,**kwargs):
        batch_size = 2
        dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
        embeddings = []
        for batch in tqdm(dataloader):
            embeddings.extend(self.get_embeddings(tokenizer, batch))
        return embeddings


get_embeddings(tokenizer, ["I am a sentence"])

evalModel = MyModel()
evaluation = MTEB(tasks=['CQADupstackAndroidRetrieval'])
evaluation.run(evalModel,output_folder=f'results')