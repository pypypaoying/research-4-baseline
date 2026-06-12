import os

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train',
        prompt_batch_size = 32
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        self.prompt_batch_size = max(1, int(os.environ.get("T3TIME_PROMPT_BATCH_SIZE", prompt_batch_size)))
        
        model_name = os.environ.get("T3TIME_GPT2_MODEL_PATH", model_name)
        local_only = os.path.isdir(model_name) or os.environ.get("T3TIME_GPT2_LOCAL_ONLY") == "1"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name, local_files_only=local_only).to(self.device)
        self.model.eval()

    def _prepare_prompt_text(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        # Last token
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"
        
        # Date
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else: # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        return in_prompt

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        in_prompt = self._prepare_prompt_text(input_template, in_data, in_data_mark, i, j)
        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
            input_templates = {
                'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
                'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
                'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends",
                'Traffic': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'Exchange': "From [t1] to [t2], the values were value1, ..., valuen every day. The total trend value was Trends",
            }

            input_template = input_templates.get(self.data_path, input_templates['FRED'])
            
            prompts = []
            prompt_index = []
            for i in range(len(in_data)):
                for j in range(in_data.shape[2]):
                    prompts.append(self._prepare_prompt_text(input_template, in_data, in_data_mark, i, j))
                    prompt_index.append((i, j))

            last_token_emb = torch.empty(
                (len(in_data), self.d_model, in_data.shape[2]),
                dtype=torch.float32,
                device=self.device,
            )

            for start in range(0, len(prompts), self.prompt_batch_size):
                end = min(start + self.prompt_batch_size, len(prompts))
                batch = self.tokenizer(
                    prompts[start:end],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    prompt_embeddings = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state
                last_positions = batch["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(prompt_embeddings.size(0), device=self.device)
                batch_last_token_emb = prompt_embeddings[rows, last_positions, :]
                for row, (i, j) in enumerate(prompt_index[start:end]):
                    last_token_emb[i, :, j] = batch_last_token_emb[row]

            return last_token_emb
