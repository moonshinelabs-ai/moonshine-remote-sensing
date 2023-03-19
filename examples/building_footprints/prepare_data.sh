#!/bin/bash
# Run this script in an environment that has the requirements installed!

# Download the tiles
aws s3 cp s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_8band.tar.gz .
tar xvf SN1_buildings_train_AOI_1_Rio_8band.tar.gz 

# Download the labels
aws s3 cp s3://spacenet-dataset/spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz .
tar xvf SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz 

# Create the masks
mkdir mask
python generate_masks ./