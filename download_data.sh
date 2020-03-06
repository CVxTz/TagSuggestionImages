# sudo apt install awscli
wget -c https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv
wget -c https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv
#wget -c https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv
#wget -c https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
wget -c https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
wget -c --no-check-certificate --no-proxy 'http://open-images-dataset.s3.amazonaws.com/tar/validation.tar.gz'
wget -c --no-check-certificate --no-proxy 'http://open-images-dataset.s3.amazonaws.com/tar/train_c.tar.gz'
wget -c --no-check-certificate --no-proxy 'http://open-images-dataset.s3.amazonaws.com/tar/train_3.tar.gz'
wget -c --no-check-certificate --no-proxy 'http://open-images-dataset.s3.amazonaws.com/tar/train_4.tar.gz'
wget -c --no-check-certificate --no-proxy 'http://open-images-dataset.s3.amazonaws.com/tar/train_5.tar.gz'
wget -c --no-check-certificate --no-proxy 'http://open-images-dataset.s3.amazonaws.com/tar/train_6.tar.gz'