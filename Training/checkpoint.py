from google.cloud import storage
import torch


def load_checkpoint_from_cloud(bucket_name, file_to_load, temp_file_name):
    
    client = storage.Client()
    #bucket_name = bucketName #"aps360team12"
    bucket = client.get_bucket(bucket_name)
    blob_name = file_to_load #"VGG_features_bs64_lr5e-05_epoch10"
    blob = bucket.get_blob(blob_name)
    blob.download_to_filename(temp_file_name)
    trained_data = torch.load(temp_file_name)
    return trained_data