# Author: Aniol Molero Grau

## Mini project – End-to-End ML Pipeline 
## MLFlow Server + PostgreSQL + Docker + AWS S3 version. 

For a simpler approach without MLflow Server, refer to the  [**NoMLFlowApproach branch**](https://github.com/AniolMG/mini-ml-pipeline/tree/NoMLFlowServerApproach).

---

Dataset: Kaggle “Titanic” 

Steps: 
- Clean & preprocess data. 
- Train a simple ML model (XGBoost). 
- Track experiments using MLflow. 

Finally, also add:
- MLflow server + PostgreSQL for versioning and tracking
- Deploy locally via FastAPI. 
- Containerize the FastAPI app using Docker. 
- AWS S3 as a remote storage service for serving the model.

---

Install all dependencies in requirements.txt with ``pip install > requirements.txt`` beforehand if you want.

Using virtual environments is recommended.

---

So first we should do some Exploratory Data Analysis (EDA). You can see my full EDA in the [jupyter notebook](titanic_EDA.ipynb). This is only for completition, since it's not the main focus of this project.

Then I trained a very simple XGBoost model, using some of the knowledge obtained in the EDA as guidance. You can see it in my [training code](train_model.py). Again, the specific model and its results are not the focus of this project.

I used MLflow to track experiments and later serve them. 
For now, with ``mlflow ui`` we can see a simple local view of our models.

I created a [main.py](main.py) file to run a FastAPI local server that allowed me to use the MLflow models with api calls with ``uvicorn main:app --reload``.

---

# MLFlow Server + PostgreSQL + Docker + AWS S3 approach:

I decided to use a local MLflow server + a local postgreSQL server to better version the models. After installing postgreSQL and needed libraries, run: 

````
CREATE DATABASE mlflow_db; 
CREATE DATABASE
````

And optionally: 
````
CREATE USER mlflow_user WITH PASSWORD 'mlflow_pass'; 
CREATE ROLE GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
````

Then, the start the MLflow server with: 
````
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 
````

Now by specifying the address of the MLflow server in the updated python training script (train_model.py) with the ``mlflow.set_tracking_uri("http://127.0.0.1:5000")`` line, the metadata (meta-info, parameters, metrics...) will also be stored in postgreSQL, allowing for better management. No need for the ``mlflow ui`` command anymore, since we will be running a local server now.

Meanwhile, the artifacts (models, plots...) will be stored locally. An object storage such as an S3, GCS, or Azure Blob Storage could also be used.

---

In the main.py file, after adding the ``mlflow.set_tracking_uri("http://127.0.0.1:5000")`` line, we don't have to rely on run IDs, and we can load the model from the MLflow Model Registry, using a "path" like ``"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"``.

Again, this can be tested by running it locally with ``uvicorn main:app --reload``, and running ``pytest`` or any customized request like ``curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Age\":29,\"Sex\":1,\"Pclass\":3}"``.

However, to run a container while still using postgreSQL, we will need a remote artifact storage, since postgreSQL stores the URI where the model is saved. The URI will always point to an absolute path in the local filesystem, which means that the container won't find it. We could bake the model inside the image or use volumes, but that would defeat the purpose of using MLflow + postgreSQL to automate the deployment.

So, I will use AWS S3 to store the artifacts.

---

To set everything up, we'll need an AWS account. There, we will need to: 
- Create an S3 bucket, which I've called "titanic-ml-artifacts"
- Create an IAM user, which I've called "titanic-mlflow-user"
- Attach a policy to the IAM user. I've used a basic policy to ONLY allow access to the "titanic-ml-artifacts" bucket:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowMLflowBucketAccess",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::titanic-ml-artifacts",
                "arn:aws:s3:::titanic-ml-artifacts/*"
            ]
        }
    ]
}
```
- Have AWS CLI installed in our system
- Get the Access Key ID and Secret Access Key for the IAM user.
- Run `aws configure` and input the Access Key ID and Secret Acess Key.

Now that we have our credentials set up, we have to run MLflow server specifying that we want to store our artifacts (models, images...) in s3:

````
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db --default-artifact-root s3://titanic-ml-artifacts --host 127.0.0.1 --port 5000
````

Also, make sure to set a new experiment name, in the training code with ``mlflow.set_experiment("titanic_s3")``, since MLflow will ignore the default-artifact-root if the experiment already had one. Make sure too that boto3 is installed (install with ``pip install boto3``).

Now if we run the training code, the model (and any other artificat we try to log) will be stored in S3. 

If it worked, we can now use MLFlow to set the new model to Staging and run ``uvicorn main:app --reload`` to test that it works.

If everything worked up until this point, we can proceed with the containerization. Remember to run ``pip freeze > requirements.txt`` beforehand.

Then build the image with:
```
docker build -t titanic-ml-api .
```

And run the container with (change \<user> with your actual user):

````
docker run -p 8000:8000 -v "C:\Users\<user>\.aws:/root/.aws:ro" -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000"  -e REGISTERED_MODEL_NAME="TitanicModel" -e MODEL_STAGE="Staging" titanic-ml-api
````

And test again that everything works with pytest.

If it works, we've built a full end-to-end pipeline in a hybrid environment that uses both our local system and cloud storage! 🎉🎉