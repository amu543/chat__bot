import json
import ollama
from ollama import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os
from chromadb.utils import embedding_functions


chat_bot = ollama.Client(host="http://localhost:11434")
# 1. Initialize the local Vector Database (ChromaDB)
client = chromadb.PersistentClient()
remote_client = Client(host=f'http://localhost:11434')
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text", url="http://localhost:11434/api/embeddings"
)

counter=0
if os.path.exists("counter.txt"):
    with open("counter.txt", "r") as f:
        counter = int(f.read())
        
collection = client.get_or_create_collection(
    name="simple_tryout",
    embedding_function=ollama_ef,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=['.','\n']
)

with open('articles.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < counter:
            print(f"skipping line {i}")
            continue

        print(f"adding line {i}")
        article = json.loads(line)
        content = article["content"]
        chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]
        
        if not content:
            continue
        
        for j, chunk in enumerate(chunks):
            collection.add(
                ids=[f"article_{i}chunk{j}"],
                #embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"title": article["title"]}]
            )
        counter+=1
        

print("Database built successfully!")

'''
# 3. Test Retrieval
#query = "When will the result of March 5 election be out?"
query = "Who was installed as prime minister?"
#query = "How long ago were the maps of Burma made that were extraordinarily accurate?"

results = collection.query(
    query_texts=[query],
    n_results=1
)

print(f"\nQuestion: {query}")
print(f"Retrieved Context: {results['documents'][0][0]}")
# After stationing vehicle on ground, leave money on gate.
# Pay for dinner using company app.
'''

with open("counter.txt", "w") as f:
    f.write(str(counter))

while True:
    user_input = input("how may i assist you? \nQuestion:")
    query_embed = remote_client.embed(model='nomic-embed-text', input=f"search_query: {user_input}")['embeddings'][0]

    results = collection.query(query_texts=[user_input], n_results=2)
    retrieved_docs = results['documents'][0]
    context = "\n\n".join(retrieved_docs)


    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

    Context: {context}

    Question: {user_input}

    Answer:"""



    response = chat_bot.generate(
        model="qwen3:4b-instruct-2507-q4_K_M", 
        prompt=prompt,
        options={
            "temperature": 0.1
        }
    )

    answer = response['response']

    print(f'Answer: {answer}')

    user_continue = input("do you want to continue? (y/n): ")
    if user_continue.lower() != 'y':
        break