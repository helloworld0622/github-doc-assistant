from typing import List, Any

import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from minsearch import VectorSearch


class SearchTool:
    def __init__(self, text_index, records, model_name="multi-qa-distilbert-cos-v1"):
        self.text_index = text_index
        self.records = records
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_index = self._build_vector_index()

    def _build_vector_index(self):
        embeddings = []

        for record in tqdm(self.records, desc="Building vector index"):
            text = record.get("content", "")
            vector = self.embedding_model.encode(text)
            embeddings.append(vector)

        embeddings = np.array(embeddings)

        vindex = VectorSearch()
        vindex.fit(embeddings, self.records)
        return vindex

    def text_search(self, query: str, num_results: int = 5) -> List[Any]:
        return self.text_index.search(query, num_results=num_results)

    def vector_search(self, query: str, num_results: int = 5) -> List[Any]:
        q = self.embedding_model.encode(query)
        return self.vector_index.search(q, num_results=num_results)

    def hybrid_search(self, query: str, num_results: int = 5) -> List[Any]:
        text_results = self.text_search(query, num_results=num_results)
        vector_results = self.vector_search(query, num_results=num_results)

        seen_ids = set()
        combined_results = []

        for result in text_results + vector_results:
            doc_id = f"{result.get('filename', '')}:{result.get('start', '')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_results.append(result)

        return combined_results[:num_results]

    def search(self, query: str) -> List[Any]:
        """
        Perform a hybrid search on the indexed repository documents.

        Args:
            query: Search query string.

        Returns:
            A list of up to 5 search results from hybrid search.
        """
        return self.hybrid_search(query, num_results=5)