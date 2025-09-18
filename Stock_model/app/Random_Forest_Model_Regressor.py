import numpy as np
import pandas as pd
from collections import Counter

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features=None, random_state=None,
                criterion='mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.tree = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _make_leaf_node(self, y):
        """For regression, leaf nodes predict the mean value"""
        return {
            'value': np.mean(y),
            'samples': len(y)
        }
    
    @staticmethod
    def _calculate_mse(y):
        """Calculate mean squared error for regression splitting"""
        if len(y) <= 1:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_variance_reduction(self, y, y_left, y_right):
        """Calculate variance reduction for regression"""
        n_total = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_total == 0 or n_left == 0 or n_right == 0:
            return 0
            
        total_mse = self._calculate_mse(y)
        left_mse = self._calculate_mse(y_left)
        right_mse = self._calculate_mse(y_right)
        
        weighted_mse = (n_left / n_total) * left_mse + (n_right / n_total) * right_mse
        return total_mse - weighted_mse
    
    def _should_stop_splitting(self, y, depth):
        """Check if we should stop splitting"""
        if len(y) <= 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if np.std(y) < 1e-5:  # Stop if variance is very small
            return True
        return False
    
    def _find_best_split_optimized(self, X, y, feature_indices):
        """Optimized version of find_best_split for regression"""
        best_reduction = -1
        best_feature, best_threshold = None, None
        
        # Precompute total MSE
        total_mse = self._calculate_mse(y)
        n_total = len(y)
        
        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            
            # Sort the feature values and corresponding target values
            sorted_indices = np.argsort(feature_values)
            sorted_features = feature_values[sorted_indices]
            sorted_targets = y[sorted_indices]
            
            # Initialize running statistics
            left_sum = 0.0
            left_sq_sum = 0.0
            left_count = 0
            
            right_sum = np.sum(sorted_targets)
            right_sq_sum = np.sum(sorted_targets ** 2)
            right_count = n_total
            
            # Iterate through potential split points
            for i in range(1, len(sorted_features)):
                # Update running statistics
                value = sorted_targets[i-1]
                left_sum += value
                left_sq_sum += value ** 2
                left_count += 1
                
                right_sum -= value
                right_sq_sum -= value ** 2
                right_count -= 1
                
                # Skip if same feature value
                if sorted_features[i] == sorted_features[i-1]:
                    continue
                    
                # Skip if minimum samples condition not met
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                
                # Calculate MSE for both sides
                left_mse = left_sq_sum / left_count - (left_sum / left_count) ** 2
                right_mse = right_sq_sum / right_count - (right_sum / right_count) ** 2
                
                # Calculate variance reduction
                reduction = total_mse - (left_count/n_total * left_mse + right_count/n_total * right_mse)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_index
                    best_threshold = (sorted_features[i] + sorted_features[i-1]) / 2
        
        return best_feature, best_threshold, best_reduction
    
    def _find_best_split(self, X, y, feature_indices):
        """Find best split using variance reduction"""
        return self._find_best_split_optimized(X, y, feature_indices)
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        if self._should_stop_splitting(y, depth):
            return self._make_leaf_node(y)
        
        # Feature selection
        if self.max_features is not None:
            if self.max_features == 'sqrt':
                n_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == 'log2':
                n_features = int(np.log2(X.shape[1]))
            else:
                n_features = self.max_features
            feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
        else:
            feature_indices = np.arange(X.shape[1])
        
        best_feature, best_threshold, best_reduction = self._find_best_split(X, y, feature_indices)
        
        if best_reduction <= 0:
            return self._make_leaf_node(y)
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'reduction': best_reduction,
            'samples': X.shape[0]
        }
    
    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if 'value' in node:  # Leaf node
            return node['value']
            
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def fit(self, X, y):
        """Fit the decision tree to the data"""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        self.tree = self._build_tree(X, y)
        return self
    
    def predict(self, X):
        """Predict for multiple samples"""
        if hasattr(X, 'values'):
            X = X.values
            
        return np.array([self._predict_single(x, self.tree) for x in X])


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", max_samples=None,
                 random_state=None, window_size=20, criterion='mse'):
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

        # keep original types for bootstrap
        X_for_boot = X
        y_for_boot = y

        if hasattr(X_for_boot, 'values'):
            X_for_boot = X_for_boot.values
        if hasattr(y_for_boot, 'values'):
            y_for_boot = y_for_boot.values

        # Generate bootstrap samples
        self.bootstrap_samples_ = self._windowed_bootstrap(X_for_boot, y_for_boot)

        # Train trees
        self.trees = []
        n_boots = len(self.bootstrap_samples_) or 1
        for i, sample in enumerate(self.bootstrap_samples_):
            # support both (X_boot, y_boot, window_start) and (X_boot, y_boot, window_start, bootstrap_indices)
            if len(sample) == 4:
                X_boot, y_boot, window_start, bootstrap_indices = sample
            elif len(sample) == 3:
                X_boot, y_boot, window_start = sample
                bootstrap_indices = None
            else:
                raise ValueError(f"Unexpected bootstrap sample format (length={len(sample)})")

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=(self.random_state + i) if self.random_state is not None else None,
                criterion=self.criterion
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if progress_callback:
                progress_callback((i + 1) / n_boots)

        return self
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        # For regression, average the predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(all_predictions, axis=0)