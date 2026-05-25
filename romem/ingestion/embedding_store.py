import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd
import json

from romem.utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        self.insert_records([{"content": t} for t in texts])

    def insert_records(self, records: List[Dict[str, Any]]):
        """
        Insert records with optional `embed_text` and `meta`.
        - `content` is the canonical identity string (hash_id computed from it).
        - `embed_text` (if provided) is what gets embedded.
        - `meta` (if provided) is stored as JSON.
        """
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            content = rec.get("content")
            if not content:
                continue
            hid = compute_mdhash_id(str(content), prefix=self.namespace + "-")
            nodes_dict[hid] = {
                "content": str(content),
                "embed_text": str(rec.get("embed_text") or content),
                "meta": rec.get("meta") or {},
            }

        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return

        existing = self.hash_id_to_row.keys()
        missing_ids = [hid for hid in all_hash_ids if hid not in existing]

        logger.debug(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} already exist."
        )
        if not missing_ids:
            return {}

        texts_to_embed = [nodes_dict[hid]["embed_text"] for hid in missing_ids]
        contents = [nodes_dict[hid]["content"] for hid in missing_ids]
        metas = [nodes_dict[hid]["meta"] for hid in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_embed)
        self._upsert(missing_ids, contents, missing_embeddings, metas=metas, embed_texts=texts_to_embed)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            if "embed_text" in df.columns:
                self.embed_texts = df["embed_text"].values.tolist()
            else:
                self.embed_texts = list(self.texts)
            if "meta" in df.columns:
                raw_meta = df["meta"].values.tolist()
                metas = []
                for m in raw_meta:
                    try:
                        metas.append(json.loads(m) if isinstance(m, str) else (m or {}))
                    except Exception:
                        metas.append({})
                self.metas = metas
            else:
                self.metas = [{} for _ in self.hash_ids]
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "embed_text": et, "meta": meta}
                for h, t, et, meta in zip(self.hash_ids, self.texts, self.embed_texts, self.metas)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.debug(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.embed_texts, self.metas = [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings,
            "embed_text": self.embed_texts,
            "meta": [json.dumps(m or {}) for m in self.metas],
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "embed_text": et, "meta": meta}
            for h, t, et, meta in zip(self.hash_ids, self.texts, self.embed_texts, self.metas)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.debug(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings, metas=None, embed_texts=None):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embed_texts.extend(embed_texts if embed_texts is not None else list(texts))
        self.metas.extend(metas if metas is not None else [{} for _ in texts])

        logger.debug("Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.debug("Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return dict(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
