# Adapted from
# https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/blob/master/sample-submissions/llama_recipes/main.py
from inference import api

import time
import fastapi
import transformers
import peft
import torch

app = fastapi.FastAPI()

peft_model_id = "mistral_finetuned"
config = peft.PeftConfig.from_pretrained(peft_model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = peft.PeftModel.from_pretrained(model, peft_model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path)


@app.post("/process")
async def process_request(input_data: api.ProcessRequest) -> api.ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    encoded = tokenizer(input_data.prompt, return_tensors="pt").to("cuda")
    prompt_length = encoded["input_ids"][0].size(0)

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )

    end_time = time.perf_counter()

    if not input_data.echo_prompt:
        decoded = tokenizer.decode(
            outputs.sequences[0][prompt_length:], skip_special_tokens=True
        )
    else:
        decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    generated_tokens = []

    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1] :]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:, :, None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(
        gen_sequences.tolist()[0],
        gen_logprobs.tolist()[0],
        zip(top_indices, top_logprobs),
    ):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            api.Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()

    return api.ProcessResponse(
        text=decoded,
        tokens=generated_tokens,
        logprob=logprob_sum,
        request_time=end_time - start_time,
    )


@app.post("/tokenize")
async def tokenize(input_data: api.TokenizeRequest) -> api.TokenizeResponse:
    start_time = time.perf_counter()
    encoded = tokenizer(input_data.text)
    end_time = time.perf_counter()
    tokens = encoded["input_ids"]
    return api.TokenizeResponse(tokens=tokens, request_time=end_time - start_time)
