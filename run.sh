#!/bin/bash
PATH=$PATH:/home/runner/work/aiap-bootcamp1-liew-chooi-chin/aiap-bootcamp1-liew-chooi-chin/src
export PATH
cd /home/runner/work/aiap-bootcamp1-liew-chooi-chin/aiap-bootcamp1-liew-chooi-chin/src
# get the db file from url
python get_score_db.py
# do the running of all algo,
# write the necessary output/features.conf,
# and also the output/params.conf,
# and all the other ouput and conf info.
python main_score.py
# need to copy the models.conf as
# an initial params config file
cp ../output/models.conf ../output params.conf
# various configuration is possible
# make a polynomial linear regression
python user_score.py -m polynomial
# make a stochastic gradient descent
python user_score.py -m sgd
# make a random forest
python user_score.py -m forest
# make a svm regressor
python user_score.py -m svr
# make a k nearest neightbors
python user_score.py -m knn
# program help
python user_score.py --help
