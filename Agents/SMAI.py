import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from deap import base, creator, tools, algorithms
import random
import logging
import datetime
import pickle
import cv2

class SelfModifyingAI:
    def __init__(self, data_dir, model_dir, frame_dir, model_type=None, versioning=True, real_time_training=False):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.frame_dir = frame_dir
        self.model_type = model_type
        self.versioning = versioning
        self.real_time_training = real_time_training
        self.model = None
        self.scaler = StandardScaler()
        self.setup_logging()
        self.load_model()

    def setup_logging(self):
        log_filename = os.path.join(self.model_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.info("Logging setup complete.")

    def log_message(self, message):
        print(message)
        logging.info(message)

    def load_model(self):
        model_path = os.path.join(self.model_dir, 'model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.log_message(f"Model loaded from {model_path}")
        else:
            self.select_best_model()
            self.log_message("Initialized new model.")

    def save_model(self):
        if self.versioning:
            model_path = os.path.join(self.model_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        else:
            model_path = os.path.join(self.model_dir, 'model.pkl')
        joblib.dump(self.model, model_path)
        self.log_message(f"Model saved to {model_path}")

    def select_best_model(self):
        data = self.load_data()
        frames = self.load_frames()
        if data.empty or frames.size == 0:
            self.log_message("No data or frames available for model selection.")
            self.model = LogisticRegression()
            self.model_type = 'logistic'
            return

        X = data.drop('target', axis=1)
        y = data['target']
        X = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'logistic': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'mlp': MLPClassifier(),
            'decision_tree': DecisionTreeClassifier()
        }

        best_model = None
        best_accuracy = 0
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.log_message(f"{model_name} model accuracy: {accuracy}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                self.model_type = model_name

        self.model = best_model
        self.log_message(f"Selected {self.model_type} model with accuracy: {best_accuracy}")

    def preprocess_data(self, X):
        return self.scaler.fit_transform(X)

    def load_data(self):
        # This function would be customized based on how data is stored and retrieved.
        # Here we assume that data is in CSV files within data_dir
        data_files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.csv')]
        if not data_files:
            self.log_message("No data files found.")
            return pd.DataFrame()
        data_list = []
        for file in data_files:
            try:
                data = pd.read_csv(file, encoding='utf-8')
                data_list.append(data)
            except UnicodeDecodeError:
                self.log_message(f"UnicodeDecodeError encountered while reading {file} with utf-8 encoding.")
                try:
                    data = pd.read_csv(file, encoding='latin1')
                    data_list.append(data)
                except Exception as e:
                    self.log_message(f"Failed to read {file} with latin1 encoding. Error: {e}")
        if not data_list:
            return pd.DataFrame()
        return pd.concat(data_list, ignore_index=True)

    def load_frames(self):
        frame_files = [os.path.join(self.frame_dir, file) for file in os.listdir(self.frame_dir) if file.endswith('.png')]
        if not frame_files:
            self.log_message("No frame files found.")
            return np.array([])

        frames = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in frame_files]
        return np.array(frames)

    def train(self):
        data = self.load_data()
        frames = self.load_frames()
        if data.empty or frames.size == 0:
            self.log_message("No data or frames available for training.")
            return
        # Combine data and frames as needed for training
        X = data.drop('target', axis=1)
        y = data['target']
        X = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.log_message(f"Training completed. Accuracy: {accuracy}")
        self.save_model()

    def modify_behavior(self):
        # Implement model modification logic
        self.log_message("Modifying behavior using DEAP.")
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.1, 10.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            return self.evaluate_model(individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=10)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=False)
        best_individual = tools.selBest(population, k=1)[0]

        self.model.C = best_individual[0]
        self.log_message(f"Adjusted model hyperparameter C to {self.model.C}")

    def evaluate_model(self, individual):
        self.model.C = individual[0]
        data = self.load_data()
        if data.empty:
            return -1,
        X = data.drop('target', axis=1)
        y = data['target']
        X = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy,

    def ask_question(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)
        return prediction[0]

    def run(self):
        if self.real_time_training:
            self.log_message("Real-time training is enabled.")
            while True:
                self.train()
                self.modify_behavior()
        else:
            self.train()
            self.modify_behavior()

if __name__ == "__main__":
    data_directory = "E:\\BAI\\save_directory"
    model_directory = "E:\\BAI"
    frame_directory = "E:\\BAI\\frame_directory"

    # Example configuration. Modify these values as needed.
    versioning = True
    real_time_training = False

    SAMI = SelfModifyingAI(data_directory, model_directory, frame_directory, versioning=versioning, real_time_training=real_time_training)
    SAMI.run()

    while True:
        sample_input = input("How can I help you? ")
        if 'exit' in sample_input.lower():
            break
        answer = SAMI.ask_question(sample_input)
        print(f"The model's prediction for the given input is: {answer}")
