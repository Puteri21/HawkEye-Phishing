import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BHHO:
    def __init__(self, objective_function, num_agents=10, max_iter=20, dim=100, alpha=0.99):
        """
        Binary Harris Hawks Optimization for Feature Selection
        
        :param objective_function: Function to evaluate the fitness of a binary mask.
        :param num_agents: Number of hawks (population size).
        :param max_iter: Maximum number of iterations.
        :param dim: Dimension of the problem (total number of features).
        :param alpha: Weight for accuracy vs feature reduction in fitness (typically high for accuracy).
        """
        self.objective_function = objective_function
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.dim = dim
        self.alpha = alpha
        
        # Initialize population randomly (0 or 1)
        self.X = np.random.randint(2, size=(self.num_agents, self.dim))
        
        # Initialize best solution
        self.Rabbit_Location = np.zeros(self.dim)
        self.Rabbit_Energy = float('inf')  # We want to minimize the fitness score
        
    def sigmoid(self, x):
        # Clip x to prevent overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-10 * (x - 0.5))) # modified sigmoid for steeper transition
        
    def evaluate_fitness(self, position):
        """
        Evaluates a single position (feature mask).
        If no features are selected, return a very high penalty.
        """
        if np.sum(position) == 0:
            return float('inf')
        return self.objective_function(position)

    def optimize(self):
        fitness = np.zeros(self.num_agents)
        history = [] # Track convergence
        
        for t in range(self.max_iter):
            # Evaluate fitness of the population
            for i in range(self.num_agents):
                fitness[i] = self.evaluate_fitness(self.X[i, :])
                
                # Update the best solution (Rabbit)
                if fitness[i] < self.Rabbit_Energy:
                    self.Rabbit_Energy = fitness[i]
                    self.Rabbit_Location = self.X[i, :].copy()
            
            # Record the best fitness at this iteration
            history.append(self.Rabbit_Energy)
            
            # Update the escaping energy of rabbit
            E0 = 2 * random.random() - 1  # -1 < E0 < 1
            E = 2 * E0 * (1 - (t / self.max_iter))  # Escaping energy
            
            # Update position of hawks
            for i in range(self.num_agents):
                if abs(E) >= 1:
                    # Exploration phase
                    q = random.random()
                    if q >= 0.5:
                        # perch based on other family members
                        rand_Hawk_index = math.floor(self.num_agents * random.random())
                        X_rand = self.X[rand_Hawk_index, :]
                        r1 = random.random()
                        r2 = random.random()
                        step = X_rand - r1 * np.abs(X_rand - 2 * r2 * self.X[i, :])
                    else:
                        # perch on a random tall tree
                        r3 = random.random()
                        r4 = random.random()
                        X_m = np.mean(self.X, axis=0)
                        step = (self.Rabbit_Location - X_m) - r3 * (0 + r4 * 1) # Simplified range 0 to 1
                        
                else:
                    # Exploitation phase
                    r = random.random() # probability of escaping
                    
                    if r >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege
                        J = 2 * (1 - random.random()) # jump strength
                        delta_X = self.Rabbit_Location - self.X[i, :]
                        step = delta_X - E * np.abs(J * self.Rabbit_Location - self.X[i, :])
                        
                    elif r >= 0.5 and abs(E) < 0.5:
                        # Hard besiege
                        step = self.Rabbit_Location - E * np.abs(self.Rabbit_Location - self.X[i, :])
                        
                    elif r < 0.5 and abs(E) >= 0.5:
                        # Soft besiege with progressive rapid dives
                        J = 2 * (1 - random.random())
                        Y = self.Rabbit_Location - E * np.abs(J * self.Rabbit_Location - self.X[i, :])
                        # Convert to binary
                        Y_bin = np.where(random.random() < self.sigmoid(Y), 1, 0)
                        
                        if self.evaluate_fitness(Y_bin) < fitness[i]:
                            step = Y
                        else:
                            Z = Y + np.random.randn(self.dim) * 0.01 # Levy flight simplified
                            step = Z
                            
                    elif r < 0.5 and abs(E) < 0.5:
                        # Hard besiege with progressive rapid dives
                        J = 2 * (1 - random.random())
                        X_m = np.mean(self.X, axis=0)
                        Y = self.Rabbit_Location - E * np.abs(J * self.Rabbit_Location - X_m)
                        Y_bin = np.where(random.random() < self.sigmoid(Y), 1, 0)
                        
                        if self.evaluate_fitness(Y_bin) < fitness[i]:
                            step = Y
                        else:
                            Z = Y + np.random.randn(self.dim) * 0.01 # Levy flight simplified
                            step = Z

                # Binarize the new position
                continuous_new_pos = self.X[i, :] + step
                # Apply sigmoid to convert to probability
                prob = self.sigmoid(continuous_new_pos)
                # Convert to binary
                self.X[i, :] = np.where(np.random.rand(self.dim) < prob, 1, 0)
                
            print(f"Iteration {t+1}/{self.max_iter} - Best Fitness: {self.Rabbit_Energy:.4f} - Features Selected: {np.sum(self.Rabbit_Location)}/{self.dim}")
            
        return self.Rabbit_Location, history

# Helper function to create the fitness function for the optimizer
def create_fitness_function(X, y, alpha=0.99):
    # Internal split just for evaluating the fitness of a feature subset
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def fitness(feature_mask):
        selected_indices = np.where(feature_mask == 1)[0]
        if len(selected_indices) == 0:
            return float('inf')
            
        X_train_sel = X_train_fit[:, selected_indices]
        X_val_sel = X_val_fit[:, selected_indices]
        
        # Use a realistic Random Forest for fitness evaluation
        rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
        rf.fit(X_train_sel, y_train_fit)
        preds = rf.predict(X_val_sel)
        error_rate = 1.0 - accuracy_score(y_val_fit, preds)
        
        # Fitness formula: α * Error + (1-α) * (Selected Features / Total Features)
        num_selected = len(selected_indices)
        total_features = len(feature_mask)
        score = alpha * error_rate + (1 - alpha) * (num_selected / total_features)
        return score
        
    return fitness
