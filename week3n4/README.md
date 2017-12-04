# Report


## Train the network to converge
I trained the model 60 epochs on full dataset (train/val=80%/20%) and it converged at around 40 epoches, below are the training&validation loss&accuracy.

### Terminal snapshot
- <img width="635" alt="converge" src="https://user-images.githubusercontent.com/5523662/33538473-82717c2a-d88f-11e7-8820-43ec0c1e4a61.png">
- training loss: 0.156535
- training accurary: 0.999362
- validation loss: 0.151774
- validation accuracy: 1.000


### Tensorboard snapshot
Snapshots belows cover metric for both training and validation

- <img width="378" alt="screen shot 2017-12-04 at 1 13 53 am" src="https://user-images.githubusercontent.com/5523662/33538633-7e72e95a-d890-11e7-914b-5568a0c5b5ff.png">
- <img width="347" alt="screen shot 2017-12-04 at 1 14 01 am" src="https://user-images.githubusercontent.com/5523662/33538635-802d41dc-d890-11e7-8b81-a32dd1a04a67.png">
- <img width="372" alt="screen shot 2017-12-04 at 1 14 09 am" src="https://user-images.githubusercontent.com/5523662/33538639-819e6a96-d890-11e7-8989-910902b2ff7a.png">
- <img width="409" alt="screen shot 2017-12-04 at 1 13 17 am" src="https://user-images.githubusercontent.com/5523662/33538628-7ace05be-d890-11e7-8e50-d72fff4bc4d4.png">
- <img width="405" alt="screen shot 2017-12-04 at 1 13 23 am" src="https://user-images.githubusercontent.com/5523662/33538631-7c458232-d890-11e7-8154-0bfda5fcd6f8.png">
- <img width="383" alt="screen shot 2017-12-04 at 1 13 34 am" src="https://user-images.githubusercontent.com/5523662/33538632-7d55ea40-d890-11e7-8a8a-2d9f4d91a817.png">


## Build the inference script and do wild testing on random google images contaning cats in
- see wild_test.ipynb