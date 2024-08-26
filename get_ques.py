import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

trained_model_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'
trained_tokenizer_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'

class QuestionGeneration:
    def __init__(self, model_dir=None):
        self.model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, answer: str, context: str):
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        question = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return {'question': question, 'answer': answer, 'context': context}

def getQuestions(dick):
  ques_lis = []
  def get_ques(context):
    max_len = 0
    for k in context:
      if len(k)>max_len:
        answer = k
        max_len = len(k)
    context = ' '.join(context).strip("\n").strip("\t").strip("\n\n")
    if len(context)>512:
      context = context[:512]
    QG = QuestionGeneration()
    qa = QG.generate(answer, context)
    ques = qa['question']
    if ques[-1] == '?':
      ques = ques[:-1]
    if len(ques)<80:
      return ques
    return False


  for i in range(len(dick)):
    num = len(dick[i])
    iterat = 0
    for j in range(num):
      context = dick[i][j]
      q = get_ques(context)
      if q:
        iterat +=1
        ques_lis.append(q)
        if iterat == 2:
          break
    return ques_lis