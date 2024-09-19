from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    data = pd.read_csv('data/preprocessed_movies.csv')  # Use the preprocessed data
    
    # Features and target
    X = data.drop(columns=['Rating'])
    y = data['Rating']
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Validate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation Mean Squared Error: {mse}")
    
    # Save the model
    joblib.dump(model, 'models/movie_rating_model.pkl')

if __name__ == "__main__":
    train_model()
