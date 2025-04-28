from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# #one time download for tokenizer
# nltk.download('punkt_tab')

def novel_preprocessing (fileaddress) :
    '''
    pypdf to read and extract the text
    using NLTK to split the text into sentences
    :param fileaddress:
    :return data:
    '''
    reader = PdfReader(fileaddress)
    print(reader.get_num_pages())
    start_page = 10
    end_page = 30
    data = []
    #worddata = []

    for i in range(start_page, end_page + 1) :
        page = reader.pages[i]
        # result += ("Page %d \n" % i)
        text = page.extract_text(0)
        if text :
            sentences = sent_tokenize(text)
            #words = word_tokenize(text)
            for sentence in sentences:
                data.append((i + 1, sentence.strip()))
            '''for word in words:
                worddata.append ((i + 1, word))'''
        # result += ("\n")
    #print(result)
    return data

def transformer (text, model) :
    '''
    Using SBERT to calculate the embeddings for each sentences
    all-MiniLM-L6-v2 have 384 vector for each sentences
    :param text:
    :param model:
    :return:
    '''
    sentences = [item[1] for item in text]
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True, batch_size=32, device="cpu")
    return embeddings

def indexing (embed) :
    '''
    indexing the embeddings with hnswlib
    practically making graph from the embedded sentences to improve search time
    hnsw has been set to 50 nodes only
    :param embed:
    :return:
    '''
    dim = embed.shape[1]
    num_elements = embed.shape[0]
    index = hnswlib.Index(space='l2', dim=dim)  # L2 distance for similarity
    index.init_index(max_elements=num_elements, ef_construction=100, M=16)
    index.add_items(embed, np.arange(num_elements))
    index.set_ef(50)
    return index

def search_query(query, model, index, data, top_k=5):
    '''
    searching the index based on query
    query will be embedded by sbert
    and compared with the index
    and return top_k value
    :param query:
    :param model:
    :param index:
    :param data:
    :param top_k:
    :return:
    '''
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    labels, distances = index.knn_query(query_embedding, k=top_k)

    results = [(data[i][0], data[i][1], d) for i, d in zip (labels[0], distances[0])]
    return results

def main():
    pdf_path = "The Angel Next Door Spoils Me Rotten - Volume 05.pdf"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sample = "he wanted to take the initiative"
    extracted_text = novel_preprocessing(pdf_path)
    embeddings = transformer(extracted_text, model)
    index = indexing(embeddings)
    results = search_query(sample, model, index, extracted_text)
    for page, sentence, score in results:
        print(f"Score - {score} - Page {page}: {sentence}")
    print()

if __name__ == "__main__":
    main()