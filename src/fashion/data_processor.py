# data_cleaning.py
import pandas as pd
import os
from PIL import Image

def clean_data(df: pd.DataFrame, image_folder: str) -> pd.DataFrame:
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Fill missing values
    df.fillna({'label': 'Unknown'}, inplace=True)
    
    # Ensure every image has a row in the CSV
    image_files = set(os.listdir(image_folder))
    df_images = set(df['image_id'].astype(str) + '.jpg')
    missing_images = image_files - df_images
    
    # Add missing images to the DataFrame
    for img in missing_images:
        df = df.append({'image_id': img.replace('.jpg', ''), 'label': 'Unknown'}, ignore_index=True)
    
    return df

# feature_engineering.py
def add_features(df: pd.DataFrame, image_folder: str) -> pd.DataFrame:
    # Extract image size (width, height) and aspect ratio
    image_sizes = []
    for img_id in df['image_id']:
        img_path = os.path.join(image_folder, img_id + '.jpg')
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                image_sizes.append((width, height, aspect_ratio))
        except:
            image_sizes.append((0, 0, 0))
    
    df[['width', 'height', 'aspect_ratio']] = pd.DataFrame(image_sizes, index=df.index)
    return df

# encoding.py
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    label_encoders = {}
    categorical_columns = ['label']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# data_processor.py
class DataProcessor:
    def __init__(self, csv_path: str, image_folder: str):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_path)
    
    def process(self):
        self.df = clean_data(self.df, self.image_folder)
        self.df = add_features(self.df, self.image_folder)
        self.df, self.encoders = encode_categorical(self.df)
        return self.df
    
    def save(self, output_path: str):
        self.df.to_csv(output_path, index=False)
    
# __init__.py
from .data_cleaning import clean_data
from .feature_engineering import add_features
from .encoding import encode_categorical
from .data_processor import DataProcessor

def preprocess_pipeline(csv_path: str, image_folder: str):
    processor = DataProcessor(csv_path, image_folder)
    df = processor.process()
    return df, processor.encoders
