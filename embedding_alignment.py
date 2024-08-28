import pickle as pkl
import tqdm
import numpy as np
from scipy.spatial import distance

FILES = ['liberal_traces.pkl', 'conservative_traces.pkl']

class Projection:
    """
    A class for projecting vectors between different semantic spaces.

    This class loads vector embeddings from files, computes alignment between
    different semantic spaces, and provides methods for projecting vectors and
    calculating distances in the projected space.
    """

    def __init__(self, file_paths):
        """
        Initialize the Projection object.

        Args:
            file_paths (list): List of file paths containing vector embeddings.
        """
        self.vectors = self.load_vectors(file_paths)
        self.domain = sorted(list(self.vectors.keys()))[0]  # Project ONTO
        self.range = sorted(list(self.vectors.keys()))[1]   # Project FROM
        self.rotation_matrix = self.compute_alignment()

    def load_vectors(self, file_paths):
        """
        Load vector embeddings from pickle files.

        Args:
            file_paths (list): List of file paths to load.

        Returns:
            dict: A dictionary of loaded vector embeddings.
        """
        vectors = {}
        for file_path in file_paths:
            category = file_path.split('_')[0]
            with open(f'{category}_embedding.pkl', 'rb') as f:
                vectors[category] = pkl.load(f)
        return vectors

    def compute_alignment(self, root=None):
        """
        Compute the alignment matrix between two semantic spaces.

        Args:
            root (str, optional): The root space for alignment. Defaults to self.domain.

        Returns:
            numpy.ndarray: The computed rotation matrix.
        """
        if root is None:
            root = self.domain

        vectors = self.vectors
        source_vectors = []
        target_vectors = []
        source_keys = []
        target_keys = []

        for category in vectors:
            if category != root:
                source_keys.extend(vectors[root].keys())
            else:
                target_keys.extend(vectors[root].keys())

        common_keys = sorted(list(set(source_keys).intersection(target_keys)))

        for category in vectors:
            if category != root:
                source_space = category
                source_vectors.extend([vectors[category][key] for key in common_keys])
            else:
                target_space = category
                target_vectors.extend([vectors[category][key] for key in common_keys])

        source_matrix = np.array(source_vectors)
        target_matrix = np.array(target_vectors)

        cross_correlation = np.matmul(np.transpose(target_matrix), source_matrix)
        u, _, v = np.linalg.svd(cross_correlation)
        rotation_matrix = np.matmul(u, v)
        return rotation_matrix

    def get_vector(self, key):
        """
        Get the vector for a given key from the range space.

        Args:
            key (str): The key to retrieve the vector for.

        Returns:
            numpy.ndarray: The vector corresponding to the key.
        """
        return self.vectors[self.range][key]

    def project_vector(self, vector):
        """
        Project a vector from the range space to the domain space.

        Args:
            vector (numpy.ndarray): The vector to project.

        Returns:
            numpy.ndarray: The projected vector.
        """
        return np.matmul(vector, self.rotation_matrix)

    def project_key(self, key):
        """
        Project a vector associated with a key from the range space to the domain space.

        Args:
            key (str): The key of the vector to project.

        Returns:
            numpy.ndarray: The projected vector.
        """
        return np.matmul(self.get_vector(key), self.rotation_matrix)

    def projection_distance(self, key):
        """
        Calculate the distance between a projected vector and its counterpart in the domain space.

        Args:
            key (str): The key of the vector to calculate the distance for.

        Returns:
            float: The Euclidean distance between the projected vector and its domain counterpart.
        """
        range_vector = self.get_vector(key)
        projected_vector = np.matmul(range_vector, self.rotation_matrix)
        domain_vector = self.vectors[self.domain][key]
        distance = np.linalg.norm(domain_vector - projected_vector)
        return distance

    def neighborhood_distance(self, key, n=10, domain=None):
        """
        Find the n nearest neighbors of a vector in the specified domain.

        Args:
            key (str): The key of the vector to find neighbors for.
            n (int, optional): The number of neighbors to return. Defaults to 10.
            domain (str, optional): The domain to search in. Defaults to self.range.

        Returns:
            list: A list of tuples containing (key, distance) pairs for the n nearest neighbors.
        """
        if domain is None:
            domain = self.range

        vector = self.get_vector(key)
        distances = {}
        for other_key, other_vector in self.vectors[domain].items():
            other_vector = np.matmul(other_vector, self.rotation_matrix)
            dist = np.linalg.norm(vector - other_vector)
            distances[other_key] = np.clip(dist, 0, 1)

        return sorted(distances.items(), key=lambda item: item[1])[:n]

    def projected_neighborhood_distance(self, key, n=10, domain=None):
        """
        Find the n nearest neighbors of a projected vector in the specified domain.

        Args:
            key (str): The key of the vector to find neighbors for.
            n (int, optional): The number of neighbors to return. Defaults to 10.
            domain (str, optional): The domain to search in. Defaults to self.domain.

        Returns:
            list: A list of tuples containing (key, distance) pairs for the n nearest neighbors.
        """
        if domain is None:
            domain = self.domain

        vector = np.matmul(self.get_vector(key), self.rotation_matrix)
        distances = {}
        for other_key, other_vector in self.vectors[domain].items():
            dist = np.linalg.norm(vector - other_vector)
            distances[other_key] = np.clip(dist, 0, 1)

        return sorted(distances.items(), key=lambda item: item[1])[:n]

    def get_diff(self):
        """
        Get the difference between the original and projected vectors for each key.

        Returns:
            dict: A dictionary of key-value pairs where the key is the vector key and the value is the difference.
        """
        differences = {}
        for key in self.vectors[self.range]:
            original_vector = self.get_vector(key)
            projected_vector = self.project_key(key)
            cos_distance = distance.cosine(original_vector,projected_vector)
            '''
            if cos_distance > 1:
                cos_distance = 1
            if cos_distance<0:
                cos_distance = 0
            '''
            differences[key] = cos_distance
            
        return differences
proj = Projection(FILES)
diff = proj.get_diff()

differences = sorted(diff.items(), key=lambda item: item[1])