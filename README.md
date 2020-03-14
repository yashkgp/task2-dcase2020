# task2-dcase2020

Extract VGGish features:

Set up VGGish model by following the instructions [here](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)

Extract features by running 
`python get_vggish_embeddings.py <machine type eg. pump> <train/test>`

This will save a pickle file with VGGish embeddings for all audio files in the specified directory

Running baseline similarity script:
`python baseline_similarity.py <machine type>`
