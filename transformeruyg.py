from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")


model.save_pretrained('./distilbart-cnn-12-6')
tokenizer.save_pretrained('./distilbart-cnn-12-6')

from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = model.to(torch_device)


def get_summary(t,tokenizer_summary,model_summary):
  txt = t['text']
  minl = t['min_length'] #75
  maxl = t['max_length'] #150
  inputs = tokenizer_summary([txt], max_length=1024,truncation=True, return_tensors='pt').to(torch_device)
  summary_ids = model_summary.generate(inputs['input_ids'], num_beams=3,num_return_sequences=1,no_repeat_ngram_size=2, min_length = minl,max_length=maxl, early_stopping=True)
  dec = [tokenizer_summary.decode(ids,skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in summary_ids]
  output = dec[0].strip()
  return {'summary':output}

text = """
Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.   A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency.
"""

payload = {'text':text,'min_length':75,'max_length':150}
summ = get_summary(payload,tokenizer,model)
summary = summ['summary']

print (summary)