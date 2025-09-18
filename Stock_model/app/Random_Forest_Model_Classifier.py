import numpy as np
import pandas as pd
from collections import Counter
import itertools

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features=None, random_state=None,
                criterion='gini'):
        # Initial setup, lets consider the structure of the tree
        self.max_depth = max_depth # how tall tree can grow
        self.min_samples_split = min_samples_split # avoid the splitting of tree too small
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.feature_importances_ = None
        # Placeholder for tree structure
        self.tree = None
        
        # Set impurity function based on criterion
        if criterion == 'gini':
            self._impurity_function = self._calculate_gini
            self._gain_function = self._gini_gain
        elif criterion == 'entropy':
            self._impurity_function = self._calculate_entropy
            self._gain_function = self._information_gain
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    # So we define a function that describes what is required to be a leaf node
    def _make_leaf_node(self, y):
        # Return the most common class label
        counts = Counter(y) # This counts the occurrences of each class label
        majority_class = max(counts, key=counts.get) # This finds the class label with the highest count

        prob = counts[majority_class] / len(y) # This calculates the probability of the majority class
        return {'class': majority_class,
                'probability': prob,
                'samples': len(y),
        } # We are returning the number of samples that reached the leaf node (for feature importance)
    
    # Now we think when should the splitting of child nodes stop?
    def _should_stop_splitting(self, y, depth):
        # Stop if all samples belong to the same class
        if len(set(y)) == 1:
            return True
        # Stop if max depth is reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        # Stop if there are too few samples to split
        if len(y) < self.min_samples_split:
            return True
        return False
    
    # Now we we want to know when to stop and how to create leaves (recursive tree making)
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) target vector
        """

        # Check the stopping conditions
        if self._should_stop_splitting(y, depth):
            return self._make_leaf_node(y)
        
        # Random Forest Models work by taking a random subset of features at each split
        # So we will define two common strategies for selecting features: sqrt and log2
        if self.max_features is not None:
            if self.max_features == 'sqrt':
                n_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == 'log2':
                n_features = int(np.log2(X.shape[1]))
            else:
                n_features = self.max_features
            feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
            # Where n_samples, total_features = X.shape
        else:
            feature_indices = np.arange(X.shape[1])
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)

        # If no split is found we make a leaf node
        if best_gain <= 0:
            return self._make_leaf_node(y)
        
        # Split the dataset into left and right
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Return the decision node
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'gain': best_gain,
            'samples': X.shape[0]
        }

    # When measuring the quality of the split we can use the entropy  and we want to reduce the 
    # disorder in the child nodes

    # Low entropy - node mostly one class
    # High entropy - node mixed classes
    @staticmethod
    def _calculate_entropy(y): # This tells us how mixed the nodes are
        n_samples = len(y)
        if n_samples == 0:
            return 0
        counts = Counter(y)
        entropy = 0.0
        for count in counts.values():
            p = count / n_samples
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    # Information Gain = Entropy(Parent) - [Weighted Average] * Entropy(Children)
    def _information_gain(self, y, y_left, y_right):
        
        n_total = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_total == 0:
            return 0
        if n_left == 0 or n_right == 0:
            return 0

        parent_entropy = self._calculate_entropy(y)
        left_entropy = self._calculate_entropy(y_left)
        right_entropy = self._calculate_entropy(y_right)

        weighted_child_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy

        return parent_entropy - weighted_child_entropy

    """Higher Information Gain means the split better separates the classes
            So now we can caluclate the information gain for a split. But which feature
            and which threshold should we use to split the data?
            So, we should iterate over all features and potential thresholds
            and calculate the information gain for each split and then choose the one
            with the highest information gain"""
    @staticmethod
    def _calculate_gini(y):
        n_samples = len(y)
        if n_samples == 0:
            return 0
        counts = Counter(y)
        gini = 1.0
        for count in counts.values():
            p = count / n_samples
            gini -= p ** 2
        return gini

    def _gini_gain(self, y, y_left, y_right):
        n_total = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_total == 0 or n_left == 0 or n_right == 0:
            return 0
            
        parent_gini = self._calculate_gini(y)
        left_gini = self._calculate_gini(y_left)
        right_gini = self._calculate_gini(y_right)
        
        weighted_child_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini
        return parent_gini - weighted_child_gini
    
    def _find_best_split_optimised(self, X, y, feature_indices):
        """
        Optimized version of find_best_split using precomputed parent impurity
        and incremental updates to child impurities.
        """
        best_gain = -1
        best_feature, best_threshold = None, None
        
        # Precompute parent impurity
        parent_impurity = self._impurity_function(y)
        n_total = len(y)
        
        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            
            # Sort the feature values and corresponding labels
            sorted_indices = np.argsort(feature_values)
            sorted_features = feature_values[sorted_indices]
            sorted_labels = y[sorted_indices]
            
            # Initialize counts for left and right nodes
            left_counts = Counter()
            right_counts = Counter(sorted_labels)
            
            # Iterate through potential split points
            for i in range(1, len(sorted_features)):
                # Update counts
                label = sorted_labels[i-1]
                left_counts[label] += 1
                right_counts[label] -= 1
                
                # Skip if same feature value
                if sorted_features[i] == sorted_features[i-1]:
                    continue
                    
                # Skip if minimum samples condition not met
                if i < self.min_samples_leaf or (n_total - i) < self.min_samples_leaf:
                    continue
                
                # Calculate impurity for both sides
                left_impurity = self._calculate_impurity_from_counts(left_counts, i)
                right_impurity = self._calculate_impurity_from_counts(right_counts, n_total - i)
                
                # Calculate gain
                gain = parent_impurity - (i/n_total * left_impurity + (n_total-i)/n_total * right_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = (sorted_features[i] + sorted_features[i-1]) / 2
        
        return best_feature, best_threshold, best_gain
    
    def _calculate_impurity_from_counts(self, counts, total):
        """Calculate impurity from counts dictionary"""
        if total == 0:
            return 0
            
        if self.criterion == 'gini':
            impurity = 1.0
            for count in counts.values():
                p = count / total
                impurity -= p ** 2
            return impurity
        else:  # entropy
            impurity = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    impurity -= p * np.log2(p)
            return impurity
    
    def _find_best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on.
        Uses the optimised version for better performance.
        """
        return self._find_best_split_optimised(X, y, feature_indices)

    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if 'class' in node:  # Leaf node
            return node['class']
            
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def predict(self, X):
        """Predict for multiple samples"""
        if hasattr(X, 'values'):
            X = X.values
            
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def predict_proba(self, X):
        """Predict probabilities for multiple samples"""
        if hasattr(X, 'values'):
            X = X.values
            
        def _predict_proba_single(x, node):
            if 'class' in node:  # Leaf node
                # For binary classification, return [prob_class_0, prob_class_1]
                # Assuming classes are 0 and 1
                if node['class'] == 0:
                    return [node['probability'], 1 - node['probability']]
                else:
                    return [1 - node['probability'], node['probability']]
                    
            if x[node['feature']] <= node['threshold']:
                return _predict_proba_single(x, node['left'])
            else:
                return _predict_proba_single(x, node['right'])
                
        return np.array([_predict_proba_single(x, self.tree) for x in X])
    
    def fit(self, X, y):
        """
        Fit the decision tree to the data.
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) target vector
        """
        # Safety for pandas numpy
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Build the tree
        self.tree = self._build_tree(X, y)

        # After building the tree, we can compute feature importances
        #self._calculate_feature_importances()

        return self
    
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", max_samples=None,
                 random_state=None, window_size=20, criterion='gini'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.window_size = window_size
        self.criterion = criterion
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _windowed_bootstrap(self, X, y):
        n_samples = X.shape[0]
        
        # Ensure window size is valid
        effective_window_size = min(self.window_size, n_samples)
        
        # Determine sample size for each tree
        if self.max_samples is not None:
            if isinstance(self.max_samples, float):
                sample_size = int(self.max_samples * n_samples)
            else:
                sample_size = self.max_samples
        else:
            sample_size = n_samples
        
        # Adjust sample_size if it's larger than the window
        sample_size = min(sample_size, effective_window_size)
        
        bootstrap_samples = []
        for i in range(self.n_estimators):
            # Randomly select a window start position
            if n_samples == effective_window_size:
                start = 0
            else:
                start = np.random.randint(0, n_samples - effective_window_size + 1)
            
            end = start + effective_window_size
            window_indices = np.arange(start, end)
            
            # Sample with replacement within the window
            bootstrap_indices = np.random.choice(
                window_indices, size=sample_size, replace=True
            )
            
            # Extract the bootstrap samples
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            bootstrap_samples.append((X_boot, y_boot, start, bootstrap_indices))
        
        return bootstrap_samples
    
    def fit(self, X, y, progress_callback=None):
        if X.shape[0] == 0:
            raise ValueError("Cannot fit model with no data")
            
        # keep original X/y types for bootstrap function to handle pandas or numpy
        if hasattr(X, 'values'):
            X_for_boot = X.values
        else:
            X_for_boot = X
        if hasattr(y, 'values'):
            y_for_boot = y.values
        else:
            y_for_boot = y

        # Generate bootstrap samples
        self.bootstrap_samples_ = self._windowed_bootstrap(X_for_boot, y_for_boot)

        # Train trees
        self.trees = []
        n_boots = len(self.bootstrap_samples_)
        for i, sample in enumerate(self.bootstrap_samples_):
            # support both (X_boot, y_boot, window_start) and
            # (X_boot, y_boot, window_start, bootstrap_indices)
            if len(sample) == 4:
                X_boot, y_boot, window_start, bootstrap_indices = sample
            elif len(sample) == 3:
                X_boot, y_boot, window_start = sample
                bootstrap_indices = None
            else:
                raise ValueError(f"Unexpected bootstrap sample format (length={len(sample)})")

            # create and fit the decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=(self.random_state + i) if self.random_state is not None else None,
                criterion=self.criterion
            )

            # allow DecisionTreeClassifier.fit to handle numpy/pandas conversion internally
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Update progress if callback provided (use actual number of bootstrap samples)
            if progress_callback:
                progress_callback((i + 1) / max(1, n_boots))
        
        return self
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        # Collect predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting for classification
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)
    
    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        # Collect probability predictions from all trees
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        
        # Average probabilities across all trees
        return np.mean(all_proba, axis=0)