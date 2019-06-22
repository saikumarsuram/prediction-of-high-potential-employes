
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_3eafcfdff0744d37a831108565de4b6a = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='s_MUSDsUpR1ByOsPJ1O5EdmehpCpbJwyb3oDIMGz8KSt',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_3eafcfdff0744d37a831108565de4b6a.get_object(Bucket='predictionofhighpotentialemployee-donotdelete-pr-vic9eed2w0y7sz',Key='turnover.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()



# In[4]:


df.head()


# In[55]:


df['sales'].value_counts()


# In[6]:


x=df.iloc[:,0:9].values


# In[7]:


y=df.iloc[:,9:10].values


# In[8]:


x


# In[9]:


y


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


le=LabelEncoder()


# In[12]:


x[:,8]=le.fit_transform(x[:,8])
y[:,0]=le.fit_transform(y[:,0])


# In[13]:


x


# In[14]:


y


# In[15]:


from sklearn.preprocessing import OneHotEncoder


# In[16]:


ohe=OneHotEncoder(categorical_features=[8])
ohe1=OneHotEncoder(categorical_features=[0])


# In[17]:


x=ohe.fit_transform(x).toarray()
y=ohe1.fit_transform(y).toarray()


# In[18]:


x


# In[19]:


y


# In[20]:


x=x[:,1:]
y=y[:,1:]


# In[21]:


x


# In[22]:


y


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[25]:


x_train


# In[26]:


y_train


# In[27]:


x_test


# In[28]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[54]:


x_train.shape


# In[30]:


y_train


# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[32]:


y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[33]:


y_pred


# In[34]:


x_test


# In[35]:


y_test


# In[36]:


x_train


# In[37]:


y_train


# In[38]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[39]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[42]:


wml_credentials={
   "access_key": "5J5gyJlF8TQv2HN8iAeeHqDr3IcaLK53zCt-VSs7AR-G",
  "instance_id": "50bef39e-7493-4f62-a354-e75c1b7b9a79",
  "password": "043cf043-b169-464c-8a7c-c59b17908fbc",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "c167b9f1-7e61-49a3-9ed2-635c91703de3"
}


# In[43]:


client = WatsonMachineLearningAPIClient(wml_credentials)
import json


# In[44]:


instance_details = client.service_instance.get_details()
print(json.dumps(instance_details, indent=2))


# In[45]:


model_props = {client.repository.ModelMetaNames.AUTHOR_NAME: "Arun", 
               client.repository.ModelMetaNames.AUTHOR_EMAIL: "abc@gmail.com", 
               client.repository.ModelMetaNames.NAME: "Decission Tree"}


# In[46]:


model_artifact =client.repository.store_model(dt, meta_props=model_props)


# In[47]:


published_model_uid = client.repository.get_model_uid(model_artifact)


# In[48]:


published_model_uid


# In[49]:


created_deployment = client.deployments.create(published_model_uid, name="Employee")


# In[50]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployment)
scoring_endpoint

client.deployments.delete('dcf24e87-42f8-4a6c-a6b9-f4c0bcea1141')
# In[53]:


client.deployments.list()

