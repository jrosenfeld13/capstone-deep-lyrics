#!/bin/bash

MODELS_DIR=../models
mkdir ${MODELS_DIR}
wget 'https://storage.googleapis.com/w210-capstone/models/4.2-LM-108k-lines-genre-song_title_itos.pkl' -P ${MODELS_DIR}
wget 'https://storage.googleapis.com/w210-capstone/models/4.2-LM-108k-lines-genre-song_title_architecture.pkl' -P ${MODELS_DIR}
wget 'https://storage.googleapis.com/w210-capstone/models/4.3-MM-108k-post-genre-song_title_itos.pkl' -P ${MODELS_DIR}
wget 'https://storage.googleapis.com/w210-capstone/models/4.3-MM-108k-post-genre-song_title_preprocessor.pkl' -P ${MODELS_DIR}
wget 'https://storage.googleapis.com/w210-capstone/models/4.3-MM-108k-post-genre-song_title_architecture.pkl' -P ${MODELS_DIR}
