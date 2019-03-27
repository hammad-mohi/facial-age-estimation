#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def organize_files(in_path, out_path, min_age, max_age):
    '''Function to split the samples into training, validation and test folders'''
    import os
    import shutil
    
    files = os.listdir(in_path)
    
    count_dict = {}
    for f in files:
        labels = f.split('_')[:3]
        name = '_'.join(labels)
        
        if min_age <= int(labels[0]) <= max_age:
            if name in count_dict:
                count_dict[name] += 1
            else:
                count_dict[name] = 1    
    
    # Making folders for each dataset
    for dirs in ['train', 'valid', 'test']:
        os.makedirs(out_path + dirs, exist_ok=True)
        for i in range(min_age, max_age):
            os.makedirs(out_path + dirs + '/' + str(i), exist_ok=True)
    
    for label in count_dict:
        fs = list(filter(lambda x: '_'.join(x.split('_')[:3]) == label, files))
        train_count = len(fs)*0.7
        valid_count = len(fs)*0.85
        train_copied = 0
        valid_copied = 0
        
        for f in fs:
            age = f.split('_')[0]
            
            if (train_copied < train_count):
                shutil.copy(in_path + f, out_path + 'train/' + age + '/' + f)
                train_copied += 1
            elif (valid_copied < valid_count - train_count):
                shutil.copy(in_path + f, out_path + 'valid/' + age + '/' + f)
                valid_copied += 1
            else:
                shutil.copy(in_path + f, out_path + 'test/' + age + '/' + f)
  
    return count_dict


# In[ ]:


def unzip_data_archive():
    get_ipython().system('pip install PyDrive')
    
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # Authenticate and create the PyDrive client.
    # This only needs to be done once in a notebook.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    
    fileId = drive.CreateFile({'id': '0BxYys69jI14kYVM3aVhKS1VhRUk'}) #DRIVE_FILE_ID is file id example: 1iytA1n2z4go3uVCwE_vIKouTKyIDjEq
    print(fileId['title'])  # folder_data.zip
    fileId.GetContentFile('UTKFace.tar.gz')  # Save Drive file as a local file
    
    # unzip
    get_ipython().system('apt-get install p7zip-full')
    get_ipython().system('p7zip -d UTKFace.tar.gz')
    get_ipython().system('tar -xvf UTKFace.tar.gz')

