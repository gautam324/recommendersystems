# Fashion Recommender System

This project aims to build a fashion recommender system using deep learning techniques. The system allows users to upload an image of a fashion item and receive recommendations for similar items.

## Project Structure

The project consists of two main parts:
1. **Feature Extraction (`app.py`)**
2. **Recommendation System (`main.py`)**

### Feature Extraction (`app.py`)

This script uses a pre-trained ResNet50 model to extract features from images in the dataset and stores these features in a pickle file.

### Recommendation System (`main.py`)

This script provides a Streamlit web application where users can upload images and get recommendations for similar fashion items.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fashion-recommender-system.git
   cd fashion-recommender-system
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Feature Extraction

Run the following script to extract features from images and save them:
```bash
python app.py
```

## Running the Recommendation System

Start the Streamlit application to get fashion recommendations:
```bash
streamlit run main.py
```



## Usage

1. **Launch the Streamlit application:**
   Open your browser and navigate to `http://localhost:8501` to access the fashion recommender system interface.

2. **Upload an Image:**
   Click on "Choose an image" to upload a fashion item image.

3. **View Recommendations:**
   The application will display the uploaded image along with five similar fashion item images.
