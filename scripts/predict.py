import pandas as pd
import joblib

def predict_new_movie(new_data):
    # Load the trained model
    model = joblib.load('models/movie_rating_model.pkl')
    
    # Process the new data as we did with the training data
    # This should match the preprocessing steps you used earlier
    # For simplicity, new_data should already be preprocessed.
    
    # Predict the rating
    predicted_rating = model.predict(new_data)
    return predicted_rating

if __name__ == "__main__":
    # Example: Predict the rating for a new movie
    new_movie_data = pd.DataFrame({
        'Name': ['New Movie'],
        'Year': [2022],
        'Duration': [120],
        'Genre': ['Comedy'],
        'Votes': [500],
        'Director': ['Famous Director'],
        'Actor 1': ['Famous Actor 1'],
        'Actor 2': ['Famous Actor 2'],
        'Actor 3': ['Famous Actor 3']
    })
    
    rating = predict_new_movie(new_movie_data)
    print(f"Predicted Rating: {rating}")
