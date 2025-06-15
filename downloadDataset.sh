#!/bin/sh

# Download the dataset from the given URL
curl -Lo infant-cry-dataset.zip https://www.kaggle.com/api/v1/datasets/download/sanmithasadhish/infant-cry-dataset

# Unzip the downloaded file
unzip infant-cry-dataset.zip
# Remove the zip file after extraction
rm infant-cry-dataset.zip
# Rename to 'cry_data' because of data.csv
mv Dataset cry_data
