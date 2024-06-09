import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import faiss
import re
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from pypai.io import TableReader
from pypai.utils import env_utils
import jieba
import pickle
import os


model_path = "/ossfs/workspace/Users/wuyan/Downloads/text2vec_sentence"

class DocsLoader():
    @classmethod
    def txt_loader(cls, filepath):
        """
        加载 txt 数据
        :param filepath:
        :return:
        """
        loader = TextLoader(filepath, encoding='utf8')
        docs = loader.load()
        return docs
 
    @classmethod
    def csv_loader(cls, filepath):
        """
        :param filepath: 
        :return: 
        """""
        loader = CSVLoader(file_path=filepath, encoding='utf8')
        docs = loader.load()
        return docs

class TextSpliter():
    @classmethod
    def text_split_by_manychar_or_charnum(cls, docs, separator=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=20,
                               length_function=len, is_separator_regex=True):
        """
        :param docs: 文档，必须为 str，如果是 langchain 加载进来的需要转换一下
        :param separator: 分割字符，默认以列表中的字符去分割 ["\n\n", "\n", " ", ""]
        :param chunk_size: 每块大小
        :param chunk_overlap: 允许字数重叠大小
        :param length_function:
        :param is_separator_regex:
        :return:
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # 指定每块大小
            chunk_overlap=chunk_overlap,  # 指定每块可以重叠的字符数
            length_function=length_function,
            is_separator_regex=is_separator_regex,
            separators=separator  # 指定按照什么字符去分割，如果不指定就按照 chunk_size +- chunk_overlap（100+-20）个字去分割
        )
        docs = docs[0].page_content  # langchian 加载的 txt 转换为 str
        split_text = text_splitter.create_documents([docs])
        return split_text

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    separators =['(?<=\。)','(?<=\；)','(?<=\！)','\n\n','\n'],
    chunk_size = 512,
    chunk_overlap  = 20)
    doc = docs[0].page_content
    texts = text_splitter.create_documents([docs])
    return texts


class TextVec:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences, max_length=512):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

class VectorStore:
    def __init__(self, dim):
        self.vectors = []
        self.metadata = []
        self.index = faiss.IndexFlatL2(dim)  # Use L2 distance (Euclidean)

    def add_vector(self, vector, meta):
        self.vectors.append(vector)
        self.metadata.append(meta)
        self.index.add(vector.numpy().reshape(1, -1))

    def search_vectors(self, query_vector, top_k=10):
        D, I = self.index.search(query_vector.numpy().reshape(1, -1), top_k)
        return [(self.metadata[i], D[0][idx]) for idx, i in enumerate(I[0])]

    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.vectors, self.metadata), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.vectors, self.metadata = pickle.load(f)
        self.index = faiss.IndexFlatL2(self.vectors[0].shape[0])
        for vector in self.vectors:
            self.index.add(vector.numpy().reshape(1, -1))

def build_vector_store(df, text2vec, column, save_path=None):
    vector_store = VectorStore(dim=text2vec.get_embedding(["test"]).shape[1])
    for _, row in df.iterrows():
        text = row[column]
        meta = row.to_dict()
        if pd.notna(text):
            vector = text2vec.get_embedding([text])[0]
            vector_store.add_vector(vector, meta)
    if save_path:
        vector_store.save(save_path)
    return vector_store

def bm25_search(query, corpus, top_k=10):
    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], scores[i]) for i in top_n_indices]


def load_odps_data():
    o = env_utils.get_odps_instance()
    reader = TableReader.from_ODPS_type(o, "antefi_dev.news_content_0605")
    df = reader.to_pandas()
    return df

df = load_odps_data()

# BM25 搜索标题
title_corpus = df['title'].tolist()
bm25_title_results = bm25_search("港股利好", title_corpus, top_k=5)
print("BM25 Title Results:\n", bm25_title_results)

# 向量搜索内容
text2vec = TextVec()
title_vector_store_path = "title_vector_store.pkl"
content_vector_store_path = "content_vector_store.pkl"

# 检查是否已经存在存储文件，如果存在则加载，否则创建新的存储文件
if os.path.exists(title_vector_store_path):
    title_vector_store = VectorStore(dim=text2vec.get_embedding(["test"]).shape[1])
    title_vector_store.load(title_vector_store_path)
else:
    title_vector_store = build_vector_store(df, text2vec, 'title', save_path=title_vector_store_path)
print("title is done")

if os.path.exists(content_vector_store_path):
    content_vector_store = VectorStore(dim=text2vec.get_embedding(["test"]).shape[1])
    content_vector_store.load(content_vector_store_path)
else:
    content_vector_store = build_vector_store(df, text2vec, 'content', save_path=content_vector_store_path)
print("content is done")

# # 组合 BM25 和向量搜索结果
bm25_matched_contents = []
for title, score in bm25_title_results:
#     print(df[df['title'] == title]['content'])
    content = df[df['title'] == title]['content'].values[0]
    bm25_matched_contents.append(content)

query_vector = text2vec.get_embedding(["港股利好"])[0]
vector_search_results = []
print('/n bm25_matched_contents',bm25_matched_contents)
