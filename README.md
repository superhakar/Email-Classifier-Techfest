# Installing pre-requisites:
* pip install -r requirements.txt

# Instructions to follow before running our Program:
* create folders named Code , CSV , Testset , Dataset , Model , Result

# Running the program:
* To run the UI
  * streamlit run app.py
* To run the API server
  * uvicorn server:app
## Uploading the train data
![Upload train files](Pictures/upload.png?raw=true "Upload data to be trained")
### Viewing word clouds of selected class
![Upload train files](Pictures/mdu.png?raw=true "MDU word cloud")
### 3D interactive visualization of how well each class of train data is clustered
![Upload train files](Pictures/3dplot.png?raw=true "3D visualization")

## Tuning Hyperparameters and Training the models
![Upload train files](Pictures/train.png?raw=true "Models used")
### Tuning Hyperparameters
![Upload train files](Pictures/trainhyper.png?raw=true "Tuning the hyperparameters of the LSTM model")
### Training
![Upload train files](Pictures/trainsuccess.png?raw=true "Training state, accuracy, and loss")

## Testing
* Upload data
* Check confidence score (If score is low, one needs to retrain the model on the test data)
* Predict (Test)
* Download prediction result
![Upload train files](Pictures/test+confidence.png?raw=true "Upload Data, Confidence Score, Predict (Test), Download Result")
