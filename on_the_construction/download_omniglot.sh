#!/usr/bin/env bash
DATADIR=datasets/omniglot

mkdir -p $DATADIR
wget -O $DATADIR/images_background.zip https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true
wget -O $DATADIR/images_evaluation.zip https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true
unzip $DATADIR/images_background.zip -d $DATADIR
unzip $DATADIR/images_evaluation.zip -d $DATADIR
mv $DATADIR/images_background/* $DATADIR/
mv $DATADIR/images_evaluation/* $DATADIR/
rmdir $DATADIR/images_background
rmdir $DATADIR/images_evaluation
