import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import shap

# To ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the graph
G = nx.DiGraph()
G.add_nodes_from(
    ['TNFa', 'TNFR1', 'IKKKa', 'IKKa', 'NFkB', 'p53', 'p53-p', 'A20 mRNA', 'A20', 'IkBa mRNA',
    'Mdm2 cyt', 'IkBa', 'Wip mRNA', 'Wip1', 'p53 mRNA', 'ATM mRNA', 'ATM', 'ATM-p', 'ATMa-p', 'MRN-p',
    'Chk2-p', 'CREB', 'KSRP-p', 'AKT-p', 'Mdm2-p nuc', 'PTEN mRNA', 'PTEN', 'PIP2', 'PIP3', 'Bax mRNA',
    'Bax', 'apoptosis', 'Mdm2 mRNA', 'p21 mRNA', 'p21', 'cell cycle arrest', 'Chk2 mRNA', 'Chk2', 'IR', 'DSB',
    'miR-16', 'pre-miR-16'])
G.add_edges_from([
    ('TNFa', 'TNFR1'), ('TNFR1', 'IKKKa'), ('IKKKa', 'IKKa'),
    ('IKKa', 'NFkB'), ('TNFR1', 'NFkB'), ('ATMa-p', 'IKKa'), ('p53', 'p53-p'),
    ('PIP2', 'PIP3'), ('p53 mRNA', 'p53'), ('DSB', 'ATM-p'), ('p53-p', 'ATM mRNA'),
    ('ATMa-p', 'p53-p'), ('ATMa-p', 'AKT-p'), ('ATMa-p', 'KSRP-p'), ('ATMa-p', 'CREB'),
    ('ATMa-p', 'Chk2-p'), ('ATM-p', 'MRN-p'), ('DSB', 'MRN-p'), ('CREB', 'ATM mRNA'),
    ('MRN-p', 'ATMa-p'), ('CREB', 'Wip mRNA'), ('p53-p', 'Chk2 mRNA'), ('p53-p', 'p21 mRNA'),
    ('p53-p', 'PTEN mRNA'), ('p53-p', 'Wip1 mRNA'), ('Wip1 mRNA', 'Wip1'), ('KSRP-p', 'pre-miR-16'),
    ('KSRP-p', 'ATM-p'), ('Chk2 mRNA', 'Chk2'), ('Chk2-p', 'p53-p'), ('A20', 'Bax mRNA'),
    ('Bax mRNA', 'Bax'), ('Bax', 'apoptosis'), ('p21 mRNA', 'p21'), ('p21', 'cell cycle arrest'),
    ('IR', 'DSB'), ('PTEN', 'PIP2'), ('PIP3', 'AKT-p'), ('AKT-p', 'Mdm2-p cyt'),
    ('IKKa', 'NFkB'), ('NFkB', 'IkBa mRNA'), ('NFkB', 'p53 mRNA'), ('NFkB', 'Wip mRNA'),
    ('IkBa mRNA', 'IkBa'), ('IkBa', 'Wip1 mRNA'), ('PTEN mRNA', 'PTEN'), ('pre-miR-16', 'miR-16'),
    ('A20 mRNA', 'A20'), ('ATM mRNA', 'ATM'), ('p53-p', 'Bax mRNA'), ('p53-p', 'Mdm2 mRNA'),
    ('Mdm2 mRNA', 'Mdm2-p cyt'), ('Mdm2 cyt', 'Mdm2-p cyt'), ('Mdm2-p cyt', 'Mdm2-p nuc'),
    ('Chk2', 'Chk2-p'), ('ATM-p', 'ATM mRNA'), ('ATM', 'ATM-p'), ('p53 mRNA', 'p53'), ('p53', 'p53-p')
])

# Generate the adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()
adj_matrix = np.array(adj_matrix)

# Generate node features using degree centrality, clustering coefficient, and betweenness centrality
degree_centrality = np.array([nx.degree_centrality(G)[node] for node in G.nodes()])
clustering_coefficient = np.array([nx.clustering(G.to_undirected())[node] for node in G.nodes()])
betweenness_centrality = np.array([nx.betweenness_centrality(G)[node] for node in G.nodes()])

# Combine the features into a feature matrix
feature_matrix = np.vstack((degree_centrality, clustering_coefficient, betweenness_centrality)).T

# Standardize features
scaler = StandardScaler()
feature_matrix = scaler.fit_transform(feature_matrix)

# Split the data into training (70%), validation (15%), and test (15%) sets
num_nodes = adj_matrix.shape[0]
indices = np.arange(num_nodes)
np.random.shuffle(indices)

train_idx = indices[:int(0.7 * num_nodes)]
val_idx = indices[int(0.7 * num_nodes):int(0.85 * num_nodes)]
test_idx = indices[int(0.85 * num_nodes):]

X_train = feature_matrix[train_idx].reshape(-1, 3, 1)
X_val = feature_matrix[val_idx].reshape(-1, 3, 1)
X_test = feature_matrix[test_idx].reshape(-1, 3, 1)

# Placeholder for target labels (binary classification: 0 or 1 for important nodes)
y_train = np.random.randint(0, 2, size=len(train_idx))
y_val = np.random.randint(0, 2, size=len(val_idx))
y_test = np.random.randint(0, 2, size=len(test_idx))

# Define the model
model = Sequential()

# Convolutional layer
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(3, 1)))
model.add(MaxPooling1D(pool_size=2))

# LSTM layer
model.add(LSTM(128, return_sequences=False))

# Fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate classification report
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# SHAP for model interpretability
explainer = shap.KernelExplainer(model.predict, X_train.reshape(X_train.shape[0], -1))
shap_values = explainer.shap_values(X_train.reshape(X_train.shape[0], -1))

# Plot SHAP summary
shap.summary_plot(shap_values, X_train.reshape(X_train.shape[0], -1), feature_names=['Degree', 'Clustering', 'Betweenness'])
\end{lstlisting}
