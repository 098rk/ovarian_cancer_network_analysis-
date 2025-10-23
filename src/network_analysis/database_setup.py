
import mysql.connector
import pandas as pd

# Establish a connection to the MySQL database
db_config = {
    'user': 'root',
    'password': 'root1',
    'host': 'localhost',
    'database': 'genomic_data_ovarian_cancer'
}

# Function to create a database connection
def create_connection(config):
    try:
        conn = mysql.connector.connect(**config)
        print("Connection to MySQL database was successful.")
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Create a new database schema
def create_schema(cursor):
    cursor.execute("DROP TABLE IF EXISTS AnimalTF")
    cursor.execute("DROP TABLE IF EXISTS CellTalk")
    cursor.execute("DROP TABLE IF EXISTS Genomic_Data")
    cursor.execute("DROP TABLE IF EXISTS PathwayCommons")

    # Create AnimalTF Table
    cursor.execute("""
    CREATE TABLE AnimalTF (
        tf_id VARCHAR(20) PRIMARY KEY,
        gene_id VARCHAR(20),
        FOREIGN KEY (gene_id) REFERENCES Genomic_Data(gene_id)
    )
    """)

    # Create CellTalk Table
    cursor.execute("""
    CREATE TABLE CellTalk (
        interaction_id INT AUTO_INCREMENT PRIMARY KEY,
        source_gene_id VARCHAR(20),
        target_gene_id VARCHAR(20),
        FOREIGN KEY (source_gene_id) REFERENCES Genomic_Data(gene_id),
        FOREIGN KEY (target_gene_id) REFERENCES Genomic_Data(gene_id)
    )
    """)

    # Create Genomic_Data Table
    cursor.execute("""
    CREATE TABLE Genomic_Data (
        gene_id VARCHAR(20) PRIMARY KEY,
        gene_symbol VARCHAR(20),
        gene_name VARCHAR(100),
        annotation TEXT
    )
    """)

    # Create PathwayCommons Table
    cursor.execute("""
    CREATE TABLE PathwayCommons (
        pathway_id VARCHAR(20) PRIMARY KEY,
        pathway_name VARCHAR(100),
        gene_id VARCHAR(20),
        FOREIGN KEY (gene_id) REFERENCES Genomic_Data(gene_id)
    )
    """)

# Function to load data into the Genomic_Data table
def load_genomic_data(cursor, file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        cursor.execute("""
        INSERT INTO Genomic_Data (gene_id, gene_symbol, gene_name, annotation)
        VALUES (%s, %s, %s, %s)
        """, (row['gene_id'], row['gene_symbol'], row['gene_name'], row['annotation']))

# Function to load AnimalTF data into the table
def load_animaltf_data(cursor, file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        cursor.execute("""
        INSERT INTO AnimalTF (tf_id, gene_id)
        VALUES (%s, %s)
        """, (row['tf_id'], row['gene_id']))

# Function to load CellTalk data into the table
def load_celltalk_data(cursor, file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        cursor.execute("""
        INSERT INTO CellTalk (source_gene_id, target_gene_id)
        VALUES (%s, %s)
        """, (row['source_gene_id'], row['target_gene_id']))

# Function to load PathwayCommons data into the table
def load_pathwaycommons_data(cursor, file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        cursor.execute("""
        INSERT INTO PathwayCommons (pathway_id, pathway_name, gene_id)
        VALUES (%s, %s, %s)
        """, (row['pathway_id'], row['pathway_name'], row['gene_id']))

# Main script execution
if __name__ == "__main__":
    connection = create_connection(db_config)
    
    if connection:
        cursor = connection.cursor()
        
        # Create database schema
        create_schema(cursor)

        # Load data into the database tables
        load_genomic_data(cursor, 'path_to_genomic_data.csv')
        load_animaltf_data(cursor, 'path_to_animaltf_data.csv')
        load_celltalk_data(cursor, 'path_to_celltalk_data.csv')
        load_pathwaycommons_data(cursor, 'path_to_pathwaycommons_data.csv')

        # Commit changes and close the connection
        connection.commit()
        cursor.close()
        connection.close()
        print("Database setup and data loading completed successfully.")
