from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from llm import LLM
import requests
import nltk
import re

# Begin

class MEM:
    def __init__(self):
        # -Load
        self.llm = LLM()
        nltk.download('punkt')
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 

    # -Functions
    # --Func to calculate cosine similarity
    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude_vec2 = sum(b ** 2 for b in vec2) ** 0.5
        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0
        return dot_product / (magnitude_vec1 * magnitude_vec2)
    
    # --Func to generate embedding from text
    def generate_embeddings_json(self, sentence):
        sentences = sent_tokenize(sentence)
        sentence_embeddings = self.model.encode(sentences)
        if sentences:
            chunks_data = {
                "data": sentences[0], 
                "embedding": sentence_embeddings[0].tolist()
            }
        else:
            chunks_data = {"data": "", "embedding": []}
        return chunks_data

    # --Func to clean data
    def clean_text(self, sentences):
        try:
            cleaned_sentences = str()
            for sentence in sentences:
                cleaned = re.sub(r'\[.*?\]', '', sentence)
                cleaned = cleaned.replace('\n', ' ')
                cleaned = cleaned.strip()
                if cleaned:
                    cleaned_sentences += ". " + cleaned
            return cleaned_sentences
        except Exception as e:
            print("Error :", e)

    # --Func to fetch nearest neighbour from DB
    def get_nearest_embeddings(self, json_data, tree_name, n=5):
        data = json_data['data']
        e = json_data['embedding']
        nearest_neighbors_payload = {
            "data": data,
            "embedding": e,
        }
        neighbor_response = requests.post(f'http://127.0.0.1:8080/nearesttop?n={n}&tree_name={tree_name}', json=nearest_neighbors_payload)
        if neighbor_response.status_code == 200:
            nearest_neighbors = neighbor_response.json()
            print(f'Found {len(nearest_neighbors)} nearest neighbors:')
            return nearest_neighbors
        else:
            print(f"Error: Received status code {neighbor_response.status_code}")
            return []
    
    # --Func to store vector in vector store
    def store_vector(self, tree_name, data):
        json_data = self.generate_embeddings_json(data)
        data = json_data['data']
        e = json_data['embedding']
        insert_payload = {
            "data": data,
            "embedding": e,
        }
        insert_response = requests.post(f'http://127.0.0.1:8080/insert?tree_name={tree_name}', json=insert_payload)
        if insert_response.status_code == 200:
            print(f'{insert_response.text}')
            return insert_response
        else:
            print(insert_response.status_code)

    # --Func to chunk paragraph
    def chunk_text(self, paragraph, max_length=2000, min_chunk_size=100):
        # ---Func to flush current chunk
        def flush_current_chunk():
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk.clear()

        sentences = nltk.tokenize.sent_tokenize(paragraph)
        chunks = []
        current_chunk = []
        current_length = 0
        code_buffer = []
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        for i, sentence in enumerate(sentences):
            if "```" in sentence:
                if code_buffer:
                    code_buffer.append(sentence)
                    if current_length + sum(len(s) for s in code_buffer) > max_length:
                        flush_current_chunk()
                    current_chunk.extend(code_buffer)
                    current_length += sum(len(s) for s in code_buffer)
                    code_buffer = []
                else:
                    flush_current_chunk()
                    code_buffer.append(sentence)
                    current_length = sum(len(s) for s in current_chunk)
            else:
                sentence_length = len(sentence)
                
                if i > 0:
                    similarity = self.cosine_similarity(
                        sentence_embeddings[i].cpu().numpy(), 
                        sentence_embeddings[i - 1].cpu().numpy()
                    )
                else:
                    similarity = 1
                if (current_length + sentence_length > max_length and current_length >= min_chunk_size) or similarity < 0.5:
                    flush_current_chunk()
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        
        flush_current_chunk()
        print(f"Created {len(chunks)} semantic chunks.")
        return chunks

    
    # --Func to Pipeline
    def pipeline(self, tree_name, message):
        embeded_json = self.generate_embeddings_json(message)
        requested_json = self.get_nearest_embeddings(embeded_json, tree_name, 10)
        if not requested_json:
            text = " "
        else:
            nearest_embedding = [item['data'] for item in requested_json]
            text = self.clean_text(nearest_embedding)
        print(text)
        prompt = (f"you are a chatbot your job is to answer any query: {message} . this is history of conversation collected using vector search histry: {text}, be creative and try to answer with what you have dont ask many question, histry provided may be out of context answer based on the question asked. ignore context that dont have meaning")
        response = self.llm.model(prompt)
        sentences = self.chunk_text(response)
        for sentence in sentences:
            print('\n',sentence)
            self.store_vector(tree_name, sentence)

        return response

# -Main
if __name__ == "__main__":
    processor = MEM()
    tree_name = 'llm'
    while True:
        message = input("Enter the message you want to process or type 'exit': ")
        if message == "exit":
            exit()
        print(f'\nAnswer by LLM: {processor.pipeline(tree_name, message)}')
