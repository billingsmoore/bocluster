import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import math

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfTransformer
import botok

logging.basicConfig(level=logging.INFO)

class BoClusterClassifier:
    def __init__(
        self,
        sample_size = 100_000,
        embed_model_name="billingsmoore/minilm-bo",
        embed_device="cpu",
        embed_batch_size=64,
        embed_max_seq_length=512,
        embed_agg_strategy=None,
        projection_args = {},
        clustering_args = {},
        summary_create=True,
    ):
        """
        Initializes the object with parameters for embedding generation, dimensionality reduction,
        clustering, and summarization of text data.

        Args:
            embed_model_name (str): Name of the pre-trained embedding model to use (default is "all-MiniLM-L6-v2").
            embed_device (str): The device to use for embedding generation. Options are 'cpu' or 'cuda' (default is 'cpu').
            embed_batch_size (int): Number of samples per batch during embedding generation (default is 64).
            embed_max_seq_length (int): Maximum sequence length for the embedding model (default is 512).
            embed_agg_strategy (str, optional): Aggregation strategy for embeddings (e.g., 'mean', 'sum', or None).
            projection_args (dict): Additional arguments for the projection algorithm (default is an empty dictionary).
            clustering_args (dict): Additional arguments for the clustering algorithm (default is an empty dictionary).
            summary_create (bool): Whether to create summaries for each cluster (default is True).

        Attributes:
            embeddings (numpy.ndarray): The embeddings for the input texts.
            faiss_index (faiss.Index): The FAISS index for fast retrieval.
            cluster_labels (numpy.ndarray): The cluster labels for each document.
            texts (list): The input texts.
            projections (numpy.ndarray): The 2D or 3D projections for visualization.
            mapper (object): The mapper for dimensionality reduction (e.g., UMAP, PCA).
            id2label (dict): Mapping from document ID to cluster label.
            label2docs (dict): Mapping from cluster label to list of document indices.
            embed_model (SentenceTransformer): The SentenceTransformer model used for embedding generation.

        """
        
        self.sample_size = sample_size
        
        # Embedding model parameters
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length
        self.embed_agg_strategy = embed_agg_strategy

        # Projection algorithm parameters (e.g., UMAP, PCA)
        self.projection_args = projection_args

        # Clustering algorithm parameters
        self.clustering_args = clustering_args

        # Summary creation parameter
        self.summary_create = summary_create

        # Initialize attributes for embeddings, projections, and clustering
        self.embeddings = None
        self.cluster_labels = None
        self.texts = None
        self.projections = None
        self.mapper = None
        self.id2label = None
        self.label2docs = None
        self.cluster_summaries = {}


        # Initialize the embedding model
        self.embed_model = SentenceTransformer(
            self.embed_model_name, device=self.embed_device
        )
        self.embed_model.max_seq_length = self.embed_max_seq_length


    def fit(self, 
            texts=None,
            projection_args=None,
            clustering_args=None
            ):
        """
        This method performs the complete process of fitting the model, including embedding the texts, projecting the embeddings into a lower-dimensional space,
        clustering the projections, and optionally summarizing the clusters.

        Args:
            texts (list): List of input texts to process. If not provided, the existing `self.texts` is used.
            projection_args (dict, optional): Additional parameters for the projection algorithm (e.g., UMAP settings).
            clustering_algorithm (str, optional): Clustering algorithm to apply. Options include 'dbscan', 'kmeans', etc. Defaults to `self.clustering_algorithm`.
            clustering_args (dict, optional): Additional parameters for the clustering algorithm (e.g., DBSCAN settings).

        Returns:
            tuple: A tuple containing:
                - embeddings (numpy.ndarray): The embeddings for the input texts.
                - cluster_labels (numpy.ndarray): The cluster labels assigned to each document.
                - cluster_summaries (dict, optional): The summaries of each cluster, if `self.summary_create` is True.
        """

        # Update internal settings with new or default parameters
        self.texts = texts or self.texts
        self.projection_args = projection_args or self.projection_args
        self.clustering_args = clustering_args or self.clustering_args

        # Embedding generation: either from scratch or using precomputed embeddings
        if self.embeddings is None:
            logging.info("Embedding texts...")
            self.embeddings = self.embed(self.texts)
        else:
            logging.info("Using precomputed embeddings...")

        # Projection: Apply dimensionality reduction (e.g., UMAP)
        if self.projections is None:
            logging.info(f"Projecting embeddings...")
            self.projections, self.mapper = self.project(self.embeddings, self.projection_args)
        else:
            logging.info("Using precomputed projections...")

        # Clustering: Apply clustering to the projections
        logging.info("Clustering...")
        self.cluster(self.projections, self.clustering_args)

        # Summarization: Optionally create summaries for each cluster
        if self.summary_create:
            logging.info("Summarizing cluster centers...")
            self.summarize()

    def embed(self, texts):
        """
        Generates embeddings for a list of text strings using the specified embedding model.

        Args:
            texts (list): List of text strings to embed.

        Returns:
            embeddings (numpy.ndarray): Array of embeddings generated for each text string.
        """

        # Generate embeddings for the input texts with specified parameters
        embeddings = self.embed_model.encode(
            texts,
            batch_size=self.embed_batch_size,      # Process texts in batches to optimize performance
            show_progress_bar=True,                # Display a progress bar for embedding generation
            convert_to_numpy=True,                 # Convert embeddings to a NumPy array format
            normalize_embeddings=True,             # Normalize embeddings to unit length
        )

        return embeddings

    def project(self, embeddings, projection_args, sample_size=None):
        """
        Projects embeddings into a lower-dimensional space using a specified dimensionality reduction algorithm.

        Args:
            embeddings (numpy.ndarray): Array of embeddings to project.
            projection_args (dict): Additional arguments for the projection algorithm, such as the number of components.

        Returns:
            tuple: A tuple containing:
                - projections (numpy.ndarray): The lower-dimensional representations of the embeddings.
                - mapper (object): The trained projection model instance.

      """

        # Set or update the projection algorithm to be used
        self.sample_size = sample_size or self.sample_size

        if len(embeddings) <= self.sample_size:
            mapper = UMAP(**projection_args).fit(embeddings)  # Fit UMAP model to embeddings
            return mapper.embedding_, mapper                  # Return UMAP projections and the model instance
        else:
            # Fit UMAP model on a random sample
            mapper = UMAP(**projection_args).fit(random.sample(list(embeddings), self.sample_size))

            num_embeddings = len(embeddings)
            embedding_dim = mapper.embedding_.shape[1]  # Get the dimensionality of the projections
            num_batches = (num_embeddings - self.sample_size) // self.sample_size  # Calculate number of batches

            # Initialize an empty NumPy array for projections
            projections = np.zeros((num_embeddings, embedding_dim))

            start = 0
            end = self.sample_size

            # Use tqdm to show progress over batches
            for batch_idx in tqdm(range(num_batches), desc="Projecting embeddings"):
                batch_projection = mapper.transform(embeddings[start:end])
                projections[start:end] = batch_projection
                start = end
                end += self.sample_size

            # Handle remaining embeddings if any
            if start < num_embeddings:
                projections[start:] = mapper.transform(embeddings[start:])

            return projections, mapper
        
    def cluster(self, embeddings, clustering_args):

        """
        Applies a specified clustering algorithm to the given embeddings and stores the resulting cluster labels.

        Args:
            embeddings (np.ndarray): Array of embeddings to cluster, with shape (num_samples, embedding_dim).
            clustering_args (dict): Dictionary of arguments specific to the chosen clustering algorithm.

        Returns:
            None

        Notes:
            - The resulting cluster labels are stored for further analysis or downstream tasks.
        """

        # Apply HDBSCAN clustering
        print(f"Using HDBSCAN params={clustering_args}")
        clustering = HDBSCAN(**clustering_args).fit(embeddings)

        # Store the resulting cluster labels
        self.store_cluster_info(clustering.labels_)

    def classify_outliers(self):
        """
        Classifies outlier data points by assigning them to the closest cluster based on their projection coordinates.
        Outliers are identified as data points with the label `-1` in `self.label2docs`. This function updates the cluster
        labels and other related features of the classifier to reflect the new assignments.

        Args:
            None

        Returns:
            None

        Notes:
            - Outliers are identified as data points with the label `-1` in `self.label2docs`.
            - The function calculates the Euclidean distance between each outlier's projection coordinates and the cluster
            centers to determine the closest cluster.
            - The cluster labels (`self.cluster_labels`) are updated to reflect the new assignments.
            - The function calls `self.store_cluster_info` to update other features of the classifier based on the new
            cluster labels.
        """

        # Keep [original key, projection coords] for outlier data points
        outlier_projs = [[elt, self.projections[elt]] for elt in self.label2docs[-1]]

        # Precompute cluster centers (excluding the -1 key, which represents outliers)
        cluster_centers = [[key, val] for key, val in self.cluster_centers.items() if key != -1]

        # Find the best label for each outlier by finding the closest cluster center
        best_labels = []
        for proj in outlier_projs:
            # Calculate distances to all cluster centers and find the closest one
            closest_label = min(cluster_centers, key=lambda center: math.dist(proj[1], center[1]))[0]
            best_labels.append([proj[0], closest_label])

        # Update cluster labels to reflect the new assignments
        for new in best_labels:
            self.cluster_labels[new[0]] = new[1]

        # Update other features of the classifier based on the new cluster labels
        self.store_cluster_info(self.cluster_labels)

    def store_cluster_info(self, cluster_labels):
        """
        Stores information about clustering results, including cluster labels, document-to-cluster mappings,
        and calculated cluster centers in a 2D projection space.

        Args:
            cluster_labels (list): List of cluster labels assigned to each document.

        Returns:
            None

        Attributes Updated:
            self.cluster_labels (list): Stores the cluster labels for each document.
            self.id2cluster (dict): Maps document indices to their assigned cluster labels.
            self.label2docs (defaultdict): Maps each cluster label to a list of document indices that belong to that cluster.
            self.cluster_centers (dict): Stores the calculated center coordinates (x, y) of each cluster in the 2D projection space.

        Notes:
            - `self.cluster_centers` is computed based on the mean coordinates of documents belonging to each cluster label.
            - Assumes `self.projections` contains 2D coordinates for each document for use in calculating cluster centers.
        """

        # Store the cluster labels provided as input
        self.cluster_labels = cluster_labels

        # Create a mapping from document index to cluster label
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }

        # Create a mapping from each cluster label to the list of document indices within that cluster
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        # Calculate the center coordinates for each cluster in the 2D projection space
        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    # create a summary label for one cluster
    def label_cluster(self, idxs: list, count_vectorizer, tfidf_res):
        
        average_tfidf_scores = np.asarray(tfidf_res[idxs].mean(axis=0)).flatten()
        top_indices = np.argsort(average_tfidf_scores)[::-1]  # Sort descending
        top_words = [(count_vectorizer.get_vocabulary()[i], average_tfidf_scores[i]) for i in top_indices]
        
        label = '__'.join([word for word, score in top_words[:3]])

        return label

    # summarize all of the clusters
    def summarize(self):

        count_vectorizer = BoCountVectorizer()
        matrix = count_vectorizer.fit_transform(self.texts)
        tfidf_tranformer = TfidfTransformer()
        tfidf_res = tfidf_tranformer.fit_transform(X=matrix)

        for cluster in self.label2docs.keys():
            idxs = self.label2docs[cluster]
            label = self.label_cluster(idxs, count_vectorizer, tfidf_res)
            self.cluster_summaries[int(cluster)] = label
    
    
    def save(self, folder):

        """
        Saves various components of the model and related data to the specified folder. If the folder doesn't exist, 
        it is created. This function saves embeddings, projections, cluster labels, texts, and optional cluster summaries 
        to disk in a structured format.

        Args:
            folder (str): The path to the folder where the model data will be saved. If the folder doesn't exist, 
                        it will be created.

        Returns:
            None

        Notes:
            - The function saves the following files in the specified folder:
                - `embeddings.npy`: The model's embeddings as a NumPy binary file.
                - `projections.npy`: The projections of the data points as a NumPy binary file.
                - `cluster_labels.npy`: The cluster labels associated with the data points.
                - `texts.json`: The raw input texts associated with the embeddings.
                - `cluster_summaries.json` (optional): Summaries of the clusters, saved if available.
            - The function uses NumPy and FAISS libraries to save arrays and indexes efficiently.
        """

        # Ensure the folder exists or create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save embeddings as a binary NumPy file
        if self.embeddings:
            with open(f"{folder}/embeddings.npy", "wb") as f:
                np.save(f, self.embeddings)

        # Save projections as a binary NumPy file
        if self.projections:
            with open(f"{folder}/projections.npy", "wb") as f:
                np.save(f, self.projections)

        # Save cluster labels as a binary NumPy file
        if self.cluster_labels:
            with open(f"{folder}/cluster_labels.npy", "wb") as f:
                np.save(f, self.cluster_labels)

        # Save the raw texts as a JSON file
        if self.texts:
            with open(f"{folder}/texts.json", "w") as f:
                json.dump(self.texts, f)

        # Optionally, save the cluster summaries if available
        if self.cluster_summaries is not None:
            with open(f"{folder}/cluster_summaries.json", "w") as f:
                json.dump(self.cluster_summaries, f)  

    def load(self, folder):
        """
        Loads model data and related information from the specified folder. If the folder doesn't exist, an error is raised.
        This function restores embeddings, projections, cluster labels, texts, and optional cluster summaries. It also 
        infers additional information based on the loaded data.

        Args:
            folder (str): The path to the folder from which the model data will be loaded. The folder must contain the necessary files.

        Raises:
            ValueError: If the specified folder does not exist.

        Returns:
            None

        Notes:
            - The function loads the following files from the specified folder:
                - `embeddings.npy`: The model's embeddings as a NumPy binary file.
                - `faiss.index`: The FAISS index object for nearest neighbor search.
                - `projections.npy`: The projections of the data points as a NumPy binary file.
                - `cluster_labels.npy`: The cluster labels associated with the data points.
                - `texts.json`: The raw input texts associated with the embeddings.
                - `cluster_summaries.json` (optional): Summaries of the clusters, loaded if available.
            - The function also infers the following based on the loaded data:
                - `id2cluster`: A mapping from document index to cluster label.
                - `label2docs`: A mapping from cluster label to a list of document indices belonging to that cluster.
                - `cluster_centers`: A dictionary of cluster centers, computed as the mean of the projections for each cluster.
        """

        # Check if the folder exists
        if not os.path.exists(folder):
            raise ValueError(f"The folder '{folder}' does not exist.")

        # Load embeddings from file
        if os.path.exists(f"{folder}/embeddings.npy"):
            with open(f"{folder}/embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)

        # Load projections from file
        if os.path.exists(f"{folder}/projections.npy"):
            with open(f"{folder}/projections.npy", "rb") as f:
                self.projections = np.load(f)

        # Load cluster labels from file
        if os.path.exists(f"{folder}/cluster_labels.npy"):
            with open(f"{folder}/cluster_labels.npy", "rb") as f:
                self.cluster_labels = np.load(f)

        # Load raw texts from file
        if os.path.exists(f"{folder}/texts.json"):
            with open(f"{folder}/texts.json", "r") as f:
                self.texts = json.load(f)

        # Optionally load cluster summaries if available
        if os.path.exists(f"{folder}/cluster_summaries.json"):
            with open(f"{folder}/cluster_summaries.json", "r") as f:
                self.cluster_summaries = json.load(f)
                keys = list(self.cluster_summaries.keys())
                for key in keys:
                    self.cluster_summaries[int(key)] = self.cluster_summaries.pop(key)

        # Infer additional objects based on loaded data
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        # Compute cluster centers based on the projections
        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    def show(self):
        """
        Visualizes the projections of the data points, optionally in 2D or 3D, with cluster labels and associated text content.
        The method displays the projections using either Matplotlib or Plotly for interactive or static plotting.

        Args:
            None
        Returns:
            None

        Notes:
            - The `labels` represent the cluster labels for each data point.
            - The function relies on the `projections` (data points' projections), `cluster_labels` (assigned clusters), and `texts` (the content for each data point).

       """
        
        # Load Tibetan font
        fpath = Path("DDC_Uchen.ttf")


        df = pd.DataFrame(
                        data={
                            "X": self.projections[:, 0],
                            "Y": self.projections[:, 1],
                            "labels": self.cluster_labels,
                            "content_display": [
                                textwrap.fill(txt[:1024], 64) for txt in self.texts
                            ],
                        }
                    )

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        df["color"] = df["labels"].apply(lambda x: "C0" if x == -1 else f"C{(x % 9) + 1}")

        df.plot(
                kind="scatter",
                x="X",
                y="Y",
                s=0.75,
                alpha=0.8,
                linewidth=0,
                color=df["color"],
                ax=ax,
                colorbar=False,
            )

        if self.cluster_summaries is not None:
            for label in self.cluster_summaries.keys():
                            if label == -1:
                                continue  # Skip the outlier cluster
                            summary = self.cluster_summaries[label]
                            position = self.cluster_centers[label]
                            t = ax.text(
                                position[0],
                                position[1],
                                summary,
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=6,
                                font=fpath
                            )
                            # Set the background for the text annotation for better readability
                            t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))

        # Turn off the axis for a cleaner plot
        ax.set_axis_off()

class BoCountVectorizer:

    def __init__(self, vocabulary=None, batch_size=5_000):
         
         """
        Custom implementation of CountVectorizer for Tibetan text
        :param vocabulary: Optional predefined vocabulary.
         """

         self.vocabulary = vocabulary
         self.word_to_index = {}
         self.batch_size = batch_size

    def fit(self, corpus):

        """
        Learns the vocabulary from the given corpus.
        :param corpus: List of texts in Tibetan.
        :return: List of results from botok WordTokenizer
        """

        if self.vocabulary is None:

            tokenizer = botok.WordTokenizer()

            if len(corpus) <= self.batch_size:
                tokenized_corpus = [tokenizer.tokenize(elt) for elt in corpus]
                
            else:
                tokenized_corpus = []
                for i in range(0,len(corpus), self.batch_size):
                    tokenized_corpus += [tokenizer.tokenize(elt) for elt in corpus[i:i+self.batch_size]]

            self.vocabulary = sorted(set([token.text for lst in tokenized_corpus for token in lst if (token.pos in ['NOUN', 'VERB'])]))

        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}

        return tokenized_corpus

    def transform(self, corpus):

        """
        Transforms texts into a document-term matrix.
        :param corpus: List of tokenized texts (each as a list of words).
        :return: NumPy array representing word counts.
        """

        try: 
            assert type(corpus[0][0]) == botok.tokenizers.token.Token
        except:
            tokenizer = botok.WordTokenizer()
            if len(corpus) <= self.batch_size:
                corpus = [tokenizer.tokenize(elt) for elt in corpus]
            else:
                corpus = []
                for i in range(0,len(corpus), self.batch_size):
                    corpus += [tokenizer.tokenize(elt) for elt in corpus[i:i+self.batch_size]]

        matrix = np.zeros((len(corpus), len(self.vocabulary)), dtype=int)

        for doc_idx, text in enumerate(corpus):
            word_counts = defaultdict(int)
            text = [elt.text for elt in text]
            for word in text:
                if word in self.word_to_index:  # Ignore words not in vocabulary
                    word_counts[word] += 1

            # Fill the row in the matrix
            for word, count in word_counts.items():
                matrix[doc_idx, self.word_to_index[word]] = count

        return matrix
    
    def fit_transform(self, corpus):

        """
        Fits the vectorizer and transforms the corpus in one step.
        """

        tokenized_corpus = self.fit(corpus)
        return self.transform(tokenized_corpus)

    def get_vocabulary(self):

        """ Returns the vocabulary (ordered list of words). """
        
        return self.vocabulary