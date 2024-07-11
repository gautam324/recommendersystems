# Book Recommendation System

This project demonstrates a simple book recommendation system using collaborative filtering and popularity-based filtering techniques. The project leverages three datasets: `books`, `users`, and `ratings`.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/book-recommendation-system.git
    cd book-recommendation-system
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Place the `books.csv`, `users.csv`, and `ratings.csv` files in the project directory.

## Datasets

- **books.csv**: Contains book details.
- **users.csv**: Contains user details.
- **ratings.csv**: Contains book ratings provided by users.

## Usage

1. Import the necessary libraries and read the datasets:
    ```python
    import numpy as np
    import pandas as pd

    books = pd.read_csv('books.csv')
    users = pd.read_csv('users.csv')
    ratings = pd.read_csv('ratings.csv')
    ```

2. Handling mixed data types warning:
    ```python
    books = pd.read_csv('books.csv', dtype={'ColumnName': 'str'})
    ```

3. Display the first few rows of the datasets:
    ```python
    users.head()
    ratings.head()
    ```

4. Print the shape of the datasets:
    ```python
    print(books.shape)
    print(ratings.shape)
    print(users.shape)
    ```

5. Check for missing values:
    ```python
    books.isnull().sum()
    users.isnull().sum()
    ratings.isnull().sum()
    ```

6. Check for duplicated rows:
    ```python
    books.duplicated().sum()
    users.duplicated().sum()
    ratings.duplicated().sum()
    ```

## Popularity-Based Filtering

1. Merge the ratings and books datasets:
    ```python
    ratings_with_name = ratings.merge(books, on='ISBN')
    ```

2. Aggregate ratings by book title:
    ```python
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    ```

3. Handle non-numeric values:
    ```python
    ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
    ratings_with_name.dropna(subset=['Book-Rating'], inplace=True)
    ```

4. Calculate average ratings:
    ```python
    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    ```

5. Combine the datasets to get the final popularity-based recommendations:
    ```python
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]
    ```

## Collaborative Filtering

1. Filter users and books:
    ```python
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(users)]

    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    ```

2. Create a pivot table:
    ```python
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    ```

3. Compute similarity scores:
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(pt)
    ```

4. Define the recommendation function:
    ```python
    def recommend(book_name):
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
        
        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)
        return data
    ```

5. Get recommendations:
    ```python
    recommendations = recommend('Harry Potter and the Sorcerer\'s Stone (Book 1)')
    for book in recommendations:
        print(book)
    ```

## Conclusion

This project showcases how to build a book recommendation system using both popularity-based and collaborative filtering methods. It helps in recommending books based on user preferences and ratings.


## Acknowledgments

- Dataset provided by [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).
- Inspired by various data science and machine learning tutorials.
