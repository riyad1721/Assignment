# Instruction about this repository
### Folder Stucture are 
- Codebase -(where i am provide two script for training and evaluation , one is load_and_prepossesing module, and model deployment script)
- Performance Metrics - (where i am provide pdf file for a report of the model’s performance on the test set, including accuracy,precision, recall, and F1 score.)
- Explanation Document - (where a i am provide a .ipynb for code explaination and one pdf file)
- Optional Deliverables - (here one fast api deploment (note: this also available on codebase folder) and docker file  )
## Setup Instructions:
- Clone the repository: git clone <repository-url>
- Create virtual environment and activate it : python -m venv venv
                                            for linux source venv/bin/activate  
                                            for Windows use `venv\Scripts\activate`
- Install the required dependencies: pip install -r requirements.txt
- Run each part of the pipeline as instructed in the respective sections above.
 1. first run python ./codebase/train.py
 2. second run python ./codebase/evaluate_model.py
 2. third cd ./codebase and run - uvicorn app:app --reload
Note: the model deploying fast api. so need to get swager documentation on fast api. so chose any brouser and Access the API at http://127.0.0.1:8000/
To make predictions on new data, use the /predict endpoint.

## Optional Deliverables
- The FastAPI script that serves the trained model for inference.
Run: uvicorn app:app --reload
Endpoints:
/predict: To make predictions on new sentences.
/health: To check the health status of the API.
- Dockerfile
For Dockerfile and related instructions for containerizing the application.
Instruction about Docker File: make sure Docker demon are available on you system 
Build the Docker image: docker build -t ner_pos_model .
Run the Docker container: docker run -p 8000:8000 ner_pos_model
Access the API at http://127.0.0.1:8000/.