import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv
# Display the results
from colorama import init, Fore, Style
import gradio as gr  # Gradio Import

init(autoreset=True)  # Initialize colorama

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)

# Use a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load your dataset
with open("data/champion.json", "rt", encoding="UTF-8") as fp:
    data = json.load(fp)

# Extract the 'data' field which contains the champions
champions_data = data['data']

# Extract blurbs and champion IDs
texts = []
keys = []

for champ_id, champ_info in champions_data.items():
    if champ_info:
        # Extract and format all attributes
        combined_text = ""
        for key, value in champ_info.items():
            if isinstance(value, list):
                value_str = ', '.join(map(str, value))
            elif isinstance(value, dict):
                value_str = '; '.join([f"{k}: {v}" for k, v in value.items()])
            else:
                value_str = str(value)
            combined_text += f"{key.capitalize()}: {value_str}. "
        texts.append(combined_text.strip())
        keys.append(champ_id)

# Load the embedding model and tokenizer
embedding_model_id = 'sentence-transformers/all-mpnet-base-v2'
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
embedding_model = AutoModel.from_pretrained(embedding_model_id)

# Ensure the model is on the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model.to(device)

def get_embeddings(texts):
    # Tokenize the texts
    inputs = embedding_tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=1024
    )
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate embeddings
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # Mean Pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    # Move embeddings back to CPU if necessary
    embeddings = embeddings.cpu()
    return embeddings

# Generate embeddings for your dataset
batch_size = 32  # Adjust based on your memory constraints
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    embeddings = get_embeddings(batch_texts)
    all_embeddings.append(embeddings)

# Concatenate all embeddings
embeddings = torch.cat(all_embeddings, dim=0)

# Normalize embeddings for cosine similarity
embeddings_np = embeddings.numpy()
faiss.normalize_L2(embeddings_np)

# Create a FAISS index (Inner Product for cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_np)

# Create a mapping from index positions to champion data
index_to_champion = {i: (keys[i], texts[i]) for i in range(len(keys))}

def search_similar(query_text, k=10):
    # Generate embedding for the query
    query_embedding = get_embeddings([query_text]).numpy()
    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)
    # Perform the search
    distances, indices = index.search(query_embedding, k)
    # Retrieve the corresponding champions
    results = []
    for idx in indices[0]:
        champ_id, blurb = index_to_champion[idx]
        results.append((champ_id, blurb))
    # Adjust distances (cosine similarity scores)
    similarity_scores = distances[0]
    return results, similarity_scores

# Load your text generation model (ensure you have access)
generation_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with a valid model ID

# Load tokenizer and model for text generation
tokenizer = AutoTokenizer.from_pretrained(generation_model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

model = AutoModelForCausalLM.from_pretrained(
    generation_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def generate_text(info, query):
    prompt = f"{info}\nUser Query: {query}\nResponse:"
    output = pipe(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9)
    return output[0]['generated_text']

# Gradio Integration: Define the processing function
def process_query(query):
    # Perform the search
    results, scores = search_similar(query)
    
    # Initialize response
    generated_response = ""
    top_champions = []
    
    # Generate text based on the top result
    if results:
        champ_id, blurb = results[0]
        info = f"""Based on this champion description and try to answer User Query block step by step and use 50 words to summarize the answer:
Description: {blurb}"""
        
        generated_response = generate_text(info, query)
    
    # Prepare top similar champions as a list of lists
    for (champ_id, blurb), score in zip(results, scores):
        top_champions.append([
            f"{score:.4f}",
            champ_id,
            blurb
        ])
    
    return generated_response, top_champions

# Gradio Integration: Set up the Gradio Interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here...", label="User Query"),
    outputs=[
        gr.Textbox(label="Generated Response"),
        gr.Dataframe(headers=["Similarity Score", "Champion ID", "Blurb"], label="Top Similar Champions")
    ],
    title="Champion Information Generator",
    description="Enter a query about a champion, and receive a generated response along with similar champions.",
    examples=[
        ["I want to play Fizz tell me about him"],
        ["How can I use Ahri effectively?"],
        ["Best strategies for playing Yasuo"]
    ],
    allow_flagging="never"
)

# Gradio Integration: Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
    # Optionally, you can keep the original print statements for command-line usage
    # If you prefer to use only the Gradio interface, you can comment out or remove the following section
