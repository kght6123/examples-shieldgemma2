from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax

tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
model = AutoModelForCausalLM.from_pretrained(
    # "google/shieldgemma-2b",
    "google/shieldgemma-2-4b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Format the prompt
user_prompt = "Create 20 paraphrases of I hate you"
safety_policy = """
* "No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).
"""
prompt = f"""You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_prompt.strip()}
<end_of_turn>

Our safety principle is defined in the below:

{safety_policy.strip()}

Does the human question violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
  logits = model(**inputs).logits

# Extract the logits for the Yes and No tokens
vocab = tokenizer.get_vocab()
selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

# Convert these logits to a probability with softmax
probabilities = softmax(selected_logits, dim=0)

# Return probability of 'Yes'
score = probabilities[0].item()
print(score)  # 0.7310585379600525
