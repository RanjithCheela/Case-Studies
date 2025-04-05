# Step 1: Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```
# Step 2: Prepare API Keys
Paste your OpenAI API key into ```api_keys/OpenAI_API_Key.txt``` file. The key can be generate from [here](https://platform.openai.com/api-keys).

Paste your Gemini API key into ```api_keys/Gemini_API_Key.txt``` file. The key can be generate from [here](https://aistudio.google.com/apikey).

# Step 3: Embed and Store Documents
Run the ```Embedder.py``` script to process and store documents in ChromaDB:
```bash
python Embedder.py
```

The following output should appear, indicate the Vector Store have successfully created:

```bash
Chroma had successfully created Vector database from 983 chunks and stored at /chroma_db/.
```

# Step 4: Query Documents
Run the ```Retriever.py``` script to interact with the system:
Run the ```Embedder.py``` script to process and store documents in ChromaDB:
```bash
python Retriever.py
```

Enter your query when prompted, or type ```exit``` to quit.