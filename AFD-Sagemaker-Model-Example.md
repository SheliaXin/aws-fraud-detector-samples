### Main Steps:

1. Train and host a Sagemaker model - sagemaker
2. Import the sagemaker endpoint to AFD and set up the detector - AFD
3. Test the detector - GEP/Batch Prediction - AFD



```python
import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()
s3_prefix = "sagemaker/DEMO-afd-sagemaker-endpoint-0322"
version_prefix = '0322'

# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import time
import json
from IPython.display import display
from time import strftime, gmtime
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
```

### Step 1: Train and Host a Sagemaker model

Code Reference: https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn.ipynb


```python
data = pd.read_csv("fraud_data_20K_sample.csv")
data['EVENT_LABEL'].value_counts()
```




    legit    18996
    fraud     1004
    Name: EVENT_LABEL, dtype: int64




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EVENT_LABEL</th>
      <th>EVENT_TIMESTAMP</th>
      <th>ip_address</th>
      <th>email_address</th>
      <th>order_amt</th>
      <th>prev_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>legit</td>
      <td>10/8/2019 20:44</td>
      <td>46.41.252.160</td>
      <td>fake_acostasusan@example.org</td>
      <td>153.71</td>
      <td>58.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>legit</td>
      <td>5/23/2020 19:44</td>
      <td>152.58.247.12</td>
      <td>fake_christopheryoung@example.com</td>
      <td>2.57</td>
      <td>11.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>legit</td>
      <td>4/24/2020 18:26</td>
      <td>12.252.206.222</td>
      <td>fake_jeffrey09@example.org</td>
      <td>30.96</td>
      <td>52.41</td>
    </tr>
    <tr>
      <th>3</th>
      <td>legit</td>
      <td>4/22/2020 19:07</td>
      <td>170.81.164.240</td>
      <td>fake_ncastro@example.org</td>
      <td>63.87</td>
      <td>34.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>legit</td>
      <td>12/31/2019 17:08</td>
      <td>165.182.68.217</td>
      <td>fake_charles99@example.org</td>
      <td>70.36</td>
      <td>66.58</td>
    </tr>
  </tbody>
</table>
</div>




```python
# prepare data for sagemaker model training
model_data = pd.get_dummies(data[['order_amt', 'prev_amt', 'EVENT_LABEL']])
model_data = pd.concat([model_data["EVENT_LABEL_fraud"], model_data.drop(["EVENT_LABEL_fraud", "EVENT_LABEL_legit"], axis=1)], axis=1)
```


```python
# split to train valid and test data
train_data, validation_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
)
train_data.to_csv("train.csv", header=False, index=False)
validation_data.to_csv("validation.csv", header=False, index=False)
```


```python
# upload to s3
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(s3_prefix, "train/train.csv")
).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(s3_prefix, "validation/validation.csv")
).upload_file("validation.csv")
```


```python
# specify the locations of the XGBoost algorithm containers - 
container = sagemaker.image_uris.retrieve("xgboost", sess.boto_region_name, "1.5-1")
display(container)
```


    '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1'



```python
s3_input_train = TrainingInput(
    s3_data="s3://{}/{}/train".format(bucket, s3_prefix), content_type="csv"
)
s3_input_validation = TrainingInput(
    s3_data="s3://{}/{}/validation/".format(bucket, s3_prefix), content_type="csv"
)
```


```python
sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path="s3://{}/{}/output".format(bucket, s3_prefix),
    sagemaker_session=sess,
)
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    verbosity=0,
    objective="binary:logistic",
    num_round=100,
)

xgb.fit({"train": s3_input_train, "validation": s3_input_validation})
```

    2023-03-22 07:41:39 Starting - Starting the training job...
    2023-03-22 07:42:06 Starting - Preparing the instances for trainingProfilerReport-1679470899: InProgress
    ......
    2023-03-22 07:43:09 Downloading - Downloading input data...
    2023-03-22 07:43:34 Training - Downloading the training image......
    2023-03-22 07:44:25 Training - Training image download completed. Training in progress..[34m[2023-03-22 07:44:35.935 ip-10-0-196-142.us-west-2.compute.internal:7 INFO utils.py:28] RULE_JOB_STOP_SIGNAL_FILENAME: None[0m
    [34m[2023-03-22 07:44:36.016 ip-10-0-196-142.us-west-2.compute.internal:7 INFO profiler_config_parser.py:111] User has disabled profiler.[0m
    [34m[2023-03-22:07:44:36:INFO] Imported framework sagemaker_xgboost_container.training[0m
    [34m[2023-03-22:07:44:36:INFO] Failed to parse hyperparameter objective value binary:logistic to Json.[0m
    [34mReturning the value itself[0m
    [34m[2023-03-22:07:44:36:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2023-03-22:07:44:36:INFO] Running XGBoost Sagemaker in algorithm mode[0m
    [34m[2023-03-22:07:44:36:INFO] Determined 0 GPU(s) available on the instance.[0m
    [34m[2023-03-22:07:44:36:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2023-03-22:07:44:36:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2023-03-22:07:44:36:INFO] files path: /opt/ml/input/data/train[0m
    [34m[2023-03-22:07:44:36:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2023-03-22:07:44:36:INFO] files path: /opt/ml/input/data/validation[0m
    [34m[2023-03-22:07:44:36:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2023-03-22:07:44:36:INFO] Single node training.[0m
    [34m[2023-03-22:07:44:36:INFO] Train matrix has 14000 rows and 2 columns[0m
    [34m[2023-03-22:07:44:36:INFO] Validation matrix has 4000 rows[0m
    [34m[2023-03-22 07:44:36.483 ip-10-0-196-142.us-west-2.compute.internal:7 INFO json_config.py:92] Creating hook from json_config at /opt/ml/input/config/debughookconfig.json.[0m
    [34m[2023-03-22 07:44:36.483 ip-10-0-196-142.us-west-2.compute.internal:7 INFO hook.py:206] tensorboard_dir has not been set for the hook. SMDebug will not be exporting tensorboard summaries.[0m
    [34m[2023-03-22 07:44:36.484 ip-10-0-196-142.us-west-2.compute.internal:7 INFO hook.py:259] Saving to /opt/ml/output/tensors[0m
    [34m[2023-03-22 07:44:36.484 ip-10-0-196-142.us-west-2.compute.internal:7 INFO state_store.py:77] The checkpoint config file /opt/ml/input/config/checkpointconfig.json does not exist.[0m
    [34m[2023-03-22:07:44:36:INFO] Debug hook created from config[0m
    [34m[2023-03-22 07:44:36.573 ip-10-0-196-142.us-west-2.compute.internal:7 INFO hook.py:427] Monitoring the collections: metrics[0m
    [34m[2023-03-22 07:44:36.577 ip-10-0-196-142.us-west-2.compute.internal:7 INFO hook.py:491] Hook is writing from the hook with pid: 7[0m
    [34m[0]#011train-logloss:0.54690#011validation-logloss:0.54770[0m
    [34m[1]#011train-logloss:0.44941#011validation-logloss:0.45117[0m
    [34m[2]#011train-logloss:0.38077#011validation-logloss:0.38320[0m
    [34m[3]#011train-logloss:0.33148#011validation-logloss:0.33463[0m
    [34m[4]#011train-logloss:0.29524#011validation-logloss:0.29907[0m
    [34m[5]#011train-logloss:0.26829#011validation-logloss:0.27269[0m
    [34m[6]#011train-logloss:0.24807#011validation-logloss:0.25303[0m
    [34m[7]#011train-logloss:0.23308#011validation-logloss:0.23870[0m
    [34m[8]#011train-logloss:0.22192#011validation-logloss:0.22808[0m
    [34m[9]#011train-logloss:0.21362#011validation-logloss:0.22051[0m
    [34m[10]#011train-logloss:0.20752#011validation-logloss:0.21513[0m
    [34m[11]#011train-logloss:0.20303#011validation-logloss:0.21114[0m
    [34m[12]#011train-logloss:0.19992#011validation-logloss:0.20827[0m
    [34m[13]#011train-logloss:0.19732#011validation-logloss:0.20650[0m
    [34m[14]#011train-logloss:0.19554#011validation-logloss:0.20526[0m
    [34m[15]#011train-logloss:0.19377#011validation-logloss:0.20488[0m
    [34m[16]#011train-logloss:0.19267#011validation-logloss:0.20420[0m
    [34m[17]#011train-logloss:0.19186#011validation-logloss:0.20393[0m
    [34m[18]#011train-logloss:0.19143#011validation-logloss:0.20379[0m
    [34m[19]#011train-logloss:0.19076#011validation-logloss:0.20357[0m
    [34m[20]#011train-logloss:0.19033#011validation-logloss:0.20346[0m
    [34m[21]#011train-logloss:0.18995#011validation-logloss:0.20350[0m
    [34m[22]#011train-logloss:0.18977#011validation-logloss:0.20361[0m
    [34m[23]#011train-logloss:0.18925#011validation-logloss:0.20404[0m
    [34m[24]#011train-logloss:0.18906#011validation-logloss:0.20441[0m
    [34m[25]#011train-logloss:0.18904#011validation-logloss:0.20442[0m
    [34m[26]#011train-logloss:0.18874#011validation-logloss:0.20472[0m
    [34m[27]#011train-logloss:0.18865#011validation-logloss:0.20484[0m
    [34m[28]#011train-logloss:0.18838#011validation-logloss:0.20501[0m
    [34m[29]#011train-logloss:0.18830#011validation-logloss:0.20511[0m
    [34m[30]#011train-logloss:0.18797#011validation-logloss:0.20530[0m
    [34m[31]#011train-logloss:0.18783#011validation-logloss:0.20555[0m
    [34m[32]#011train-logloss:0.18768#011validation-logloss:0.20561[0m
    [34m[33]#011train-logloss:0.18749#011validation-logloss:0.20567[0m
    [34m[34]#011train-logloss:0.18722#011validation-logloss:0.20578[0m
    [34m[35]#011train-logloss:0.18707#011validation-logloss:0.20598[0m
    [34m[36]#011train-logloss:0.18707#011validation-logloss:0.20598[0m
    [34m[37]#011train-logloss:0.18707#011validation-logloss:0.20601[0m
    [34m[38]#011train-logloss:0.18707#011validation-logloss:0.20601[0m
    [34m[39]#011train-logloss:0.18707#011validation-logloss:0.20600[0m
    [34m[40]#011train-logloss:0.18707#011validation-logloss:0.20601[0m
    [34m[41]#011train-logloss:0.18694#011validation-logloss:0.20598[0m
    [34m[42]#011train-logloss:0.18685#011validation-logloss:0.20609[0m
    [34m[43]#011train-logloss:0.18676#011validation-logloss:0.20610[0m
    [34m[44]#011train-logloss:0.18661#011validation-logloss:0.20611[0m
    [34m[45]#011train-logloss:0.18654#011validation-logloss:0.20615[0m
    [34m[46]#011train-logloss:0.18655#011validation-logloss:0.20613[0m
    [34m[47]#011train-logloss:0.18638#011validation-logloss:0.20648[0m
    [34m[48]#011train-logloss:0.18625#011validation-logloss:0.20667[0m
    [34m[49]#011train-logloss:0.18625#011validation-logloss:0.20667[0m
    [34m[50]#011train-logloss:0.18602#011validation-logloss:0.20652[0m
    [34m[51]#011train-logloss:0.18596#011validation-logloss:0.20665[0m
    [34m[52]#011train-logloss:0.18580#011validation-logloss:0.20665[0m
    [34m[53]#011train-logloss:0.18580#011validation-logloss:0.20663[0m
    [34m[54]#011train-logloss:0.18580#011validation-logloss:0.20663[0m
    [34m[55]#011train-logloss:0.18543#011validation-logloss:0.20673[0m
    [34m[56]#011train-logloss:0.18530#011validation-logloss:0.20685[0m
    [34m[57]#011train-logloss:0.18518#011validation-logloss:0.20693[0m
    [34m[58]#011train-logloss:0.18502#011validation-logloss:0.20696[0m
    [34m[59]#011train-logloss:0.18483#011validation-logloss:0.20715[0m
    [34m[60]#011train-logloss:0.18471#011validation-logloss:0.20702[0m
    [34m[61]#011train-logloss:0.18471#011validation-logloss:0.20703[0m
    [34m[62]#011train-logloss:0.18470#011validation-logloss:0.20708[0m
    [34m[63]#011train-logloss:0.18458#011validation-logloss:0.20708[0m
    [34m[64]#011train-logloss:0.18448#011validation-logloss:0.20685[0m
    [34m[65]#011train-logloss:0.18441#011validation-logloss:0.20684[0m
    [34m[66]#011train-logloss:0.18441#011validation-logloss:0.20688[0m
    [34m[67]#011train-logloss:0.18433#011validation-logloss:0.20700[0m
    [34m[68]#011train-logloss:0.18433#011validation-logloss:0.20701[0m
    [34m[69]#011train-logloss:0.18425#011validation-logloss:0.20708[0m
    [34m[70]#011train-logloss:0.18425#011validation-logloss:0.20705[0m
    [34m[71]#011train-logloss:0.18419#011validation-logloss:0.20719[0m
    [34m[72]#011train-logloss:0.18412#011validation-logloss:0.20721[0m
    [34m[73]#011train-logloss:0.18412#011validation-logloss:0.20717[0m
    [34m[74]#011train-logloss:0.18402#011validation-logloss:0.20728[0m
    [34m[75]#011train-logloss:0.18402#011validation-logloss:0.20727[0m
    [34m[76]#011train-logloss:0.18402#011validation-logloss:0.20728[0m
    [34m[77]#011train-logloss:0.18391#011validation-logloss:0.20730[0m
    [34m[78]#011train-logloss:0.18392#011validation-logloss:0.20728[0m
    [34m[79]#011train-logloss:0.18384#011validation-logloss:0.20729[0m
    [34m[80]#011train-logloss:0.18384#011validation-logloss:0.20728[0m
    [34m[81]#011train-logloss:0.18384#011validation-logloss:0.20729[0m
    [34m[82]#011train-logloss:0.18384#011validation-logloss:0.20730[0m
    [34m[83]#011train-logloss:0.18384#011validation-logloss:0.20732[0m
    [34m[84]#011train-logloss:0.18379#011validation-logloss:0.20731[0m
    [34m[85]#011train-logloss:0.18379#011validation-logloss:0.20729[0m
    [34m[86]#011train-logloss:0.18379#011validation-logloss:0.20731[0m
    [34m[87]#011train-logloss:0.18379#011validation-logloss:0.20731[0m
    [34m[88]#011train-logloss:0.18379#011validation-logloss:0.20734[0m
    [34m[89]#011train-logloss:0.18372#011validation-logloss:0.20759[0m
    [34m[90]#011train-logloss:0.18364#011validation-logloss:0.20743[0m
    [34m[91]#011train-logloss:0.18358#011validation-logloss:0.20747[0m
    [34m[92]#011train-logloss:0.18356#011validation-logloss:0.20751[0m
    [34m[93]#011train-logloss:0.18356#011validation-logloss:0.20748[0m
    [34m[94]#011train-logloss:0.18346#011validation-logloss:0.20749[0m
    [34m[95]#011train-logloss:0.18323#011validation-logloss:0.20743[0m
    [34m[96]#011train-logloss:0.18323#011validation-logloss:0.20742[0m
    [34m[97]#011train-logloss:0.18323#011validation-logloss:0.20742[0m
    [34m[98]#011train-logloss:0.18323#011validation-logloss:0.20742[0m
    [34m[99]#011train-logloss:0.18323#011validation-logloss:0.20746[0m
    
    2023-03-22 07:45:04 Uploading - Uploading generated training model
    2023-03-22 07:45:04 Completed - Training job completed
    Training seconds: 107
    Billable seconds: 107



```python
# deploy sagemaker endpoint
xgb_predictor = xgb.deploy(
    initial_instance_count=1, instance_type="ml.m4.xlarge", serializer=CSVSerializer(),
    endpoint_name = f"sagemaker-xgb-endpoint-{version_prefix}"
)
```

    -------!


```python
f"sagemaker-xgb-endpoint-{version_prefix}"
```




    'sagemaker-xgb-endpoint-0322'




```python
def predict(data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ""
    for array in split_array:
        predictions = "".join([predictions, xgb_predictor.predict(array).decode("utf-8")])

    return predictions.split("\n")[:-1]


predictions = predict(test_data.to_numpy()[:, 1:])
```


```python
predictions = np.array([float(num) for num in predictions])
print(len(predictions), predictions)
```

    2000 [0.0716714  0.03765393 0.02415792 ... 0.05634578 0.06239426 0.03940216]


### Step 2: Import the SageMaker model to AFD and set up the detector


```python
fraudDetector = boto3.client('frauddetector')
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/boto3/compat.py:88: PythonDeprecationWarning: Boto3 will no longer support Python 3.6 starting May 30, 2022. To continue receiving service updates, bug fixes, and security updates please upgrade to Python 3.7 or later. More information can be found here: https://aws.amazon.com/blogs/developer/python-support-policy-updates-for-aws-sdks-and-tools/
      warnings.warn(warning, PythonDeprecationWarning)



```python
### create afd variables, entity and event type
try:
    resp = fraudDetector.get_variables(name = 'order_amt')
except:
    resp = fraudDetector.create_variable(name = 'order_amt', dataType = 'FLOAT', dataSource ='EVENT', defaultValue = '0.0')

try:
    resp = fraudDetector.get_variables(name = 'prev_amt')
except:  
    resp = fraudDetector.create_variable(name = 'prev_amt', dataType = 'FLOAT', dataSource ='EVENT', defaultValue = '0.0')

response = fraudDetector.put_entity_type(name = f'sagemaker-xgb-entity-{version_prefix}')

response = fraudDetector.put_event_type (
        name           = f'sagemaker-xgb-transaction-{version_prefix}',
        eventVariables = ['order_amt', 'prev_amt'],
        entityTypes    = [f'sagemaker-xgb-entity-{version_prefix}'])
```


```python
### create external model score variable
resp = fraudDetector.create_variable(name = f'sagemaker_xgb_score_{version_prefix}', dataType = 'FLOAT', dataSource ='EXTERNAL_MODEL_SCORE', defaultValue = '0.0')

```


```python
### put external model
# https://docs.aws.amazon.com/frauddetector/latest/ug/import-an-amazon-sagemaker-model.html
fraudDetector.put_external_model(
    modelSource = 'SAGEMAKER',
    modelEndpoint = f'sagemaker-xgb-endpoint-{version_prefix}',
    invokeModelEndpointRoleArn = role, #'your_SagemakerExecutionRole_arn',
    inputConfiguration = {
        'useEventVariables' : True,
        'eventTypeName' : f'sagemaker-xgb-transaction-{version_prefix}',
        'format' : 'TEXT_CSV',
        'csvInputTemplate' : '{{order_amt}}, {{prev_amt}}' # add afd enrichment, how the config works
    },
    outputConfiguration = {
        'format' : 'TEXT_CSV',
        'csvIndexToVariableMap' : {
        '0' : f'sagemaker_xgb_score_{version_prefix}'
        }
    },
    modelEndpointStatus = 'ASSOCIATED'
)
```




    {'ResponseMetadata': {'RequestId': 'c592749d-6907-4550-a5fb-0114d1729024',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'date': 'Wed, 22 Mar 2023 23:21:55 GMT',
       'content-type': 'application/x-amz-json-1.1',
       'content-length': '2',
       'connection': 'keep-alive',
       'x-amzn-requestid': 'c592749d-6907-4550-a5fb-0114d1729024'},
      'RetryAttempts': 0}}




```python
### create a detector
DETECTOR_NAME = f"afd-with-sagemaker-model-{version_prefix}"
response = fraudDetector.put_detector(
    detectorId    = DETECTOR_NAME, 
    eventTypeName = f'sagemaker-xgb-transaction-{version_prefix}' )
```


```python
### Create rules

def create_outcomes(outcomes):
    """ 
    Create Fraud Detector Outcomes 
    """   
    for outcome in outcomes:
        print("creating outcome variable: {0} ".format(outcome))
        response = fraudDetector.put_outcome(name = outcome, description = outcome)

def create_rules(score_cuts, outcomes, MODEL_SCORE_NAME, DETECTOR_NAME):
    """
    Creating rules 
    
    Arguments:
        score_cuts  - list of score cuts to create rules
        outcomes    - list of outcomes associated with the rules
    
    Returns:
        a rule list to used when create detector
    """
    
    if len(score_cuts)+1 != len(outcomes):
        logging.error('Your socre cuts and outcomes are not matched.')
    
    rule_list = []
    for i in range(len(outcomes)):
        # rule expression
        if i < (len(outcomes)-1):
            rule = "${0} > {1}".format(MODEL_SCORE_NAME,score_cuts[i])
        else:
            rule = "${0} <= {1}".format(MODEL_SCORE_NAME,score_cuts[i-1])
    
        # append to rule_list (used when create detector)
        rule_id = "rules_{0}_{1}".format(i, MODEL_SCORE_NAME)
        
        rule_list.append({
            "ruleId": rule_id, 
            "ruleVersion" : '1',
            "detectorId"  : DETECTOR_NAME
        })
        
        # create rules
        print("creating rule: {0}: IF {1} THEN {2}".format(rule_id, rule, outcomes[i]))
        try:
            response = fraudDetector.create_rule(
                ruleId = rule_id,
                detectorId = DETECTOR_NAME,
                expression = rule,
                language = 'DETECTORPL',
                outcomes = [outcomes[i]]
                )
        except:
            print("this rule already exists in this detector")
            
    return rule_list

score_cuts = [0.5,0.9]                         
outcomes = ['fraud', 'investigate', 'approve']  
create_outcomes(outcomes)
rule_list = create_rules(score_cuts, outcomes, f'sagemaker_xgb_score_{version_prefix}', DETECTOR_NAME)
```

    creating outcome variable: fraud 
    creating outcome variable: investigate 
    creating outcome variable: approve 
    creating rule: rules_0_sagemaker_xgb_score_0322: IF $sagemaker_xgb_score_0322 > 0.5 THEN fraud
    creating rule: rules_1_sagemaker_xgb_score_0322: IF $sagemaker_xgb_score_0322 > 0.9 THEN investigate
    creating rule: rules_2_sagemaker_xgb_score_0322: IF $sagemaker_xgb_score_0322 <= 0.9 THEN approve



```python
# -- create detector version --
response =fraudDetector.create_detector_version(
    detectorId    = DETECTOR_NAME ,
    rules         = rule_list,
    externalModelEndpoints = [f'sagemaker-xgb-endpoint-{version_prefix}'],
    ruleExecutionMode = 'FIRST_MATCHED'
)
```


```python
response = fraudDetector.update_detector_version_status(
    detectorId        = DETECTOR_NAME,
    detectorVersionId = '1',
    status            = 'ACTIVE'
)
```


```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EVENT_LABEL_fraud</th>
      <th>order_amt</th>
      <th>prev_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1459</th>
      <td>0</td>
      <td>156.00</td>
      <td>135.04</td>
    </tr>
    <tr>
      <th>13935</th>
      <td>0</td>
      <td>41.58</td>
      <td>89.56</td>
    </tr>
    <tr>
      <th>6843</th>
      <td>0</td>
      <td>21.40</td>
      <td>404.08</td>
    </tr>
    <tr>
      <th>17103</th>
      <td>0</td>
      <td>35.17</td>
      <td>135.47</td>
    </tr>
    <tr>
      <th>2286</th>
      <td>0</td>
      <td>91.72</td>
      <td>122.84</td>
    </tr>
  </tbody>
</table>
</div>



### Step 3: Test the detector using boto3 SDK


```python
pred = fraudDetector.get_event_prediction(
    detectorId        = f"afd-with-sagemaker-model-{version_prefix}",
    detectorVersionId = '1',
    eventId           = '1459',
    eventTypeName     = f'sagemaker-xgb-transaction-{version_prefix}',
    eventTimestamp    = '2019-10-05T22:50:48Z',
    entities          = [{
        'entityType': f'sagemaker-xgb-entity-{version_prefix}', 
        'entityId':"UNKNOWN"
    }],
    eventVariables    = {
        'order_amt': '156',
        'prev_amt':'135.04'
    }) 
```


```python
pred
```




    {'modelScores': [],
     'ruleResults': [{'ruleId': 'rules_2_sagemaker_xgb_score_0322',
       'outcomes': ['approve']}],
     'externalModelOutputs': [{'externalModel': {'modelEndpoint': 'sagemaker-xgb-endpoint-0322',
        'modelSource': 'SAGEMAKER'},
       'outputs': {'sagemaker_xgb_score_0322': '0.07167139649391174\n'}}],
     'ResponseMetadata': {'RequestId': 'dd59c122-f16b-4c99-9971-ce3e6bf7fe03',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'date': 'Wed, 22 Mar 2023 23:25:16 GMT',
       'content-type': 'application/x-amz-json-1.1',
       'content-length': '283',
       'connection': 'keep-alive',
       'x-amzn-requestid': 'dd59c122-f16b-4c99-9971-ce3e6bf7fe03'},
      'RetryAttempts': 0}}




```python

```
