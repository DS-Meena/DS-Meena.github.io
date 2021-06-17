# How to download Kaggle Datasets

Today we will see how we can download the kaggle datasets using kaggle API in your notebook.

## 1. Need a Kaggle Account:

If don't have one create it, then you need to get your Authorized key.

Go to kaggle -> Account->API

now  click on get new token.

You can also expire the previous token and get a new token. You should do this because your old token can be unsecure now. 

![kaggle api token](/images/Kaggle_api_token.png)

Download the kaggle.json file.
This file contains your Kaggle username and your personal authorized key to access to different public datasets.

## 2. Set the Kaggle API

Install the kaggle library using pip command.

`! pip install kaggle`

Their are two ways to set your key:-

a) Move the downloaded kaggle.json file into the /root/.kaggle folder.
This folder automatically created after installing kaggle. 

or 

b) `import os
kaggle_data={"username":"As in kaggle.json","key":"As in kaggle.json"}
os.environ['KAGGLE_USERNAME']=kaggle_data["username"]
os.environ['KAGGLE_KEY']=kaggle_data["key"]`

Now, import the kaggle api.

`from kaggle import api`

## 3. Download the desired Dataset

You need the specific command for specific dataset.

Go to the Dataset on Kaggle and copy the api command.


![copy api command](/images/Kaggle_api_command.png)

Now paste it in your notebook and then it will start downloading the data in a zip file.

![copy api command](/images/Kaggle_api_dataset_download.png)

Since, the data will be inside a zip file, so we need to first unzip the zip folder. 
For extracting the files we can use:

`!unzip \*.zip`

![Unzip the files](/images/zip_Unzip_file.png)

Now, you can see the files extracted. So, go ahead do what matters.

**Now, work done.**



>Note - we can also use wget command to download a dataset direct from website.

>like `! wget url_to_zip_file.zip`