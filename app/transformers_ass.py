from transformers import BertTokenizer, BertModel
from bertviz import head_view

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True,
    output_hidden_states=True
)

sentence = "Transformers are amazing!"
inputs = tokenizer(sentence, return_tensors="pt")

print("Input IDs:", inputs["input_ids"])
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

outputs = model(**inputs)

hidden_states = outputs.hidden_states
attentions = outputs.attentions

print("\n--- Hidden States ---")
print("Number of layers:", len(hidden_states))  
print("Shape of embeddings layer:", hidden_states[0].shape)  
print("Shape of last hidden layer:", hidden_states[-1].shape)  

print("\n--- Attentions ---")
print("Number of layers:", len(attentions))  
print("Shape of first layer attention:", attentions[0].shape)  
print("Shape of last layer attention:", attentions[-1].shape)  


tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
head_view(attentions, tokens)  # Opens interactive attention view
