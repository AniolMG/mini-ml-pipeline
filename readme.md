# Author: Aniol Molero Grau

## Mini project – End-to-End ML Pipeline 
## MLFlow Server + PostgreSQL + FastAPI + Docker + AWS S3 version. 

For a simpler approach without MLflow Server, refer to the  [**NoMLFlowApproach branch**](https://github.com/AniolMG/mini-ml-pipeline/tree/NoMLFlowServerApproach).

---

Dataset: Kaggle “Titanic” (https://www.kaggle.com/c/titanic/data)

### Steps  
- Clean & preprocess data.  
- Train a simple ML model (XGBoost).  
- Track experiments using **MLflow (no server)**.  
- Set up **MLflow server + PostgreSQL** for versioning and tracking.  
- Deploy locally via **FastAPI**.  
- Containerize the FastAPI app using **Docker**.  
- Use **AWS S3** as a remote storage service for serving the model.  
- (Optional) Automate deployment with **CI/CD** using GitHub Actions + self-hosted runners.  




---

Install all dependencies in requirements.txt with ``pip install -r requirements.txt`` beforehand if you want.

Using virtual environments is recommended.

---

So first we should do some Exploratory Data Analysis (EDA). You can see my full EDA in the [jupyter notebook](titanic_EDA.ipynb). This is only for completition, since it's not the main focus of this project.

Then I trained a very simple XGBoost model, using some of the knowledge obtained in the EDA as guidance. You can see it in my [training code](src/train_model.py). Again, the specific model and its results are not the focus of this project. Keep in mind this won't work until we run the MLflow server since the script uses it. If you want, you can comment the ``mlflow.set_tracking_uri`` and ``set_experiment`` lines to see that the training script works, or simply run the [training code without MLflow Server](src/train_model_without_mlflowserver.py).

I used MLflow to track experiments and later serve them. 
For now, with ``mlflow ui`` we can see a simple local view of our models.

I created a [main.py](src/main.py) file to run a FastAPI local server that allowed me to use the MLflow models with api calls with ``uvicorn main:app --reload``. HOWEVER, this won't work, again, because it's expecting MLflow server to be set up correctly. Refer to the  [**NoMLFlowApproach branch**](https://github.com/AniolMG/mini-ml-pipeline/tree/NoMLFlowServerApproach) if you want a simpler set up.

---

# MLFlow Server + PostgreSQL + FastAPI + Docker + AWS S3 approach:

I decided to use a local MLflow server + a local postgreSQL server to better version the models. After installing postgreSQL and needed libraries, run: 

````
CREATE DATABASE mlflow_db; 
CREATE DATABASE
````

And: 
````
CREATE USER mlflow_user WITH PASSWORD 'mlflow_pass'; 
CREATE ROLE GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
````

Then, the start the MLflow server with (change any parameter as needed): 
````
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 
````

Now by specifying the address of the MLflow server in the updated python training script (train_model.py) with the ``mlflow.set_tracking_uri("http://127.0.0.1:5000")`` line, the metadata (meta-info, parameters, metrics...) will also be stored in postgreSQL, allowing for better management. No need for the ``mlflow ui`` command anymore, since we will be running a local server now.

Meanwhile, the artifacts (models, plots...) will be stored locally. An object storage such as an S3, GCS, or Azure Blob Storage could also be used.

---

In the main.py file, after adding the ``mlflow.set_tracking_uri("http://127.0.0.1:5000")`` line, we don't have to rely on run IDs, and we can load the model from the MLflow Model Registry, using a "path" like ``"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"``. We simply have to use MLflow interface to register a model and then put it into Staging stage (could also be production, but you will have to edit main.py).

Again, this can be tested by running it locally with ``uvicorn main:app --reload``, and running ``pytest`` or any customized request like ``curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"Age\":29,\"Sex\":1,\"Pclass\":3}"``.

However, to run a container while still using postgreSQL, we will need a remote artifact storage, since postgreSQL stores the URI where the model is saved. The URI will always point to an absolute path in the local filesystem, which means that the container won't find it. We could bake the model inside the image or use volumes, but that would defeat the purpose of using MLflow + postgreSQL to automate the deployment.

So, let's use AWS S3 to store the artifacts.

---

To set everything up, we'll need an AWS account. There, we will need to: 
- Create an S3 bucket, which I've called "titanic-ml-artifacts".
- Create an IAM user, which I've called "titanic-mlflow-user".
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
- Have AWS CLI installed in our system.
- Get the Access Key ID and Secret Access Key for the IAM user.
- Run `aws configure` and input the Access Key ID and Secret Acess Key.

Now that we have our credentials set up, we have to run MLflow server specifying that we want to store our artifacts (models, images...) in s3 (close previous mlflow server execution):

````
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db --default-artifact-root s3://titanic-ml-artifacts --host 127.0.0.1 --port 5000
````

Also, make sure to set a new experiment name, in the training code with ``mlflow.set_experiment("titanic_s3")``, since MLflow will ignore the default-artifact-root if the experiment already had one. Make sure too that boto3 is installed (install with ``pip install boto3``).

Now if we run the training code, the model (and any other artifiact we try to log) will be stored in S3. 

If it worked, we can now use MLFlow to set the new model to Staging and run ``uvicorn main:app --reload`` to test that it works.

If everything worked up until this point, we can proceed with the containerization. Remember to run ``pip freeze > requirements.txt`` beforehand if you installed anything else.

Then build the image with:
```
docker build -t titanic-ml-api .
```

Alternatively, you can download my public image in GitHub Container Registry with:

```
docker pull ghcr.io/aniolmg/titanic-ml-api:419c6001d25773c251d1b0c40c38b6ea4ae02d21
```

419c6001d25773c251d1b0c40c38b6ea4ae02d21 is the last working tested image. You could also use ``docker pull ghcr.io/aniolmg/titanic-ml-api:latest`` to get the last version at the moment of pulling. 

We will have to give the container the permissions to access S3, and also specify the docker port, model name, and model stage, so run it with (change \<user> with your actual user):

````
docker run -p 8000:8000 -v "C:\Users\<user>\.aws:/root/.aws:ro" -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000"  -e REGISTERED_MODEL_NAME="TitanicModel" -e MODEL_STAGE="Staging" titanic-ml-api
````

If you downloaded my public image instead of building it yourself, simple replace the image name at the end. The complete command should look like this:

````
docker run -p 8000:8000 -v "C:\Users\<user>\.aws:/root/.aws:ro" -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000"  -e REGISTERED_MODEL_NAME="TitanicModel" -e MODEL_STAGE="Staging" ghcr.io/aniolmg/titanic-ml-api
````

And test again that everything works with ``pytest``.

---

## Optional CI/CD Automation:

To automate the deployment process and implement a basic CI/CD workflow, we can use GitHub Actions with .yaml workflow files. Specifically, the deployment can be automated using GitHub self-hosted runners.

If you want to reproduce this setup:

- Set up your own GitHub repository and replace any references to my account with your GitHub username.

- Configure GitHub Secrets and Variables:

    - Go to your repo → Settings → Secrets and Variables → Actions

    - For this project, you only need to set AWS_CREDENTIALS_PATH to point to your AWS credentials folder, e.g., C:\Users\<user>\.aws.

- Install a local GitHub Runner:

    - Go to your repo → Actions → Runners → Self-hosted runners

    - Follow GitHub’s step-by-step guide to set it up.

---
If everything is set up correctly, you will have a full end-to-end pipeline running in a hybrid environment, combining your local system with cloud storage, and optionally automating CI/CD! 🎉
---
<br><br><br><br>
<br><br><br><br>



---
# How to fully remove MLflow + PostgreSQL experiments:

First, delete the experiment using the MLflow UI. However, this won't "fully" delete it, and you won't be able to create a new experiment with the same name.

So, we then have to remove it fully fom postgreSQL. To do that, go into pgAdmin 4, go into the mlflow_db database, look for the Query tool (Alt + Shift + Q by default) and paste and run these queries:

````
-- 1. Experiment-level children
DELETE FROM experiment_tags 
WHERE experiment_id IN (
    SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
);

DELETE FROM logged_model_params
WHERE experiment_id IN (
    SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
);

DELETE FROM logged_model_tags
WHERE experiment_id IN (
    SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
);

-- 2. Run-level children
DELETE FROM params 
WHERE run_uuid IN (
    SELECT run_uuid FROM runs WHERE experiment_id IN (
        SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
    )
);

DELETE FROM metrics 
WHERE run_uuid IN (
    SELECT run_uuid FROM runs WHERE experiment_id IN (
        SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
    )
);

DELETE FROM latest_metrics 
WHERE run_uuid IN (
    SELECT run_uuid FROM runs WHERE experiment_id IN (
        SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
    )
);

DELETE FROM tags 
WHERE run_uuid IN (
    SELECT run_uuid FROM runs WHERE experiment_id IN (
        SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
    )
);

-- 3. Runs
DELETE FROM runs 
WHERE experiment_id IN (
    SELECT experiment_id FROM experiments WHERE lifecycle_stage='deleted'
);

-- 4. Finally, experiments
DELETE FROM experiments 
WHERE lifecycle_stage='deleted';
````

After this, the experiment should have been cleanly removed! 🧹✨