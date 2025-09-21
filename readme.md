# Author: Aniol Molero Grau

## Mini project – End-to-End ML Pipeline 
## Basic MLflow tracking + Docker 

For a simpler approach without MLflow Server, refer to the  [**NoMLFlowApproach branch**](https://github.com/AniolMG/mini-ml-pipeline/tree/NoMLFlowServerApproach).

---

Dataset: Kaggle “Titanic” 

Steps: 
- Clean & preprocess data. 
- Train a simple ML model (XGBoost). 
- Track experiments using MLflow. 
- Containerize the FastAPI app using Docker. 

---

Install all dependencies in requirements.txt with ``pip install -r requirements.txt`` beforehand if you want.

Using virtual environments is recommended.

---

So first we should do some Exploratory Data Analysis (EDA). You can see my full EDA in the [jupyter notebook](titanic_EDA.ipynb). This is only for completition, since it's not the main focus of this project.

Then I trained a very simple XGBoost model, using some of the knowledge obtained in the EDA as guidance. You can see it in my [training code](train_model.py). Again, the specific model and its results are not the focus of this project.


I used MLflow to track experiments and later serve them. 
For now, with ``mlflow ui`` we can see a simple local view of our models.

I created a [main.py](main.py) file to run a FastAPI local server that allowed me to use the MLflow models with api calls.

---

# NO MLflow Server approach:

First set the run_id variable: ``set RUN_ID=<run id>``. You can find it in the MLflow UI. They look like: 65d12fe841e44243a29b8bb953529248 

Then run with: ``uvicorn main:app --reload`` 

And call the API with: ``curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Age\":29,\"Sex\":1,\"Pclass\":3}"`` 

or ``curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"Age\":29,\"Sex\":1,\"Pclass\":3}"``

You can also test it easily by simply running ``pytest``.

I created a DockerFile and built it by running a command inside the project folder: ``docker build -t titanic-ml-api .`` 
I can check that it has been built correctly with: ``docker images``

---

An option would be running (adapt parameters as needed): `docker run -p 8000:8000 -v "E:\LearningProjects\mini-ml-pipeline\saved_model:/app/saved_model" -e RUN_ID=<run id> titanic-ml-api`

This runs the container on port 8000, and uses volumes to use the models in the local system.

Once the container is running, run `pytest` to quickly check that the container is working as expected.

---

You have now Dockerized a simple ML app with basic MLflow tracking! 🎉