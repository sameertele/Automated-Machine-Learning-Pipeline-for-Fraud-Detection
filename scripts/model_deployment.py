#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

def deploy_model(model):
    #Azure Container Instance
    ws = Workspace.from_config()

    inference_config = InferenceConfig(
        entry_script='score.py',
        environment=Environment(name="AzureML-TensorFlow-2.3-CPU")
    )

    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    # Deploying the model
    service = Model.deploy(
        workspace=ws,
        name='fraud-detection-service',
        models=[model],
        inference_config=inference_config,
        deployment_config=aci_config
    )
    service.wait_for_deployment(show_output=True)
    print(f"Model deployed to: {service.scoring_uri}")

if __name__ == "__main__":
    model_path = "path_to_best_model.pkl"
    model = Model.load(model_path)
    deploy_model(model)

