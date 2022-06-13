# Ultra-Lite-Convolutional-Neural-Network-for-Automatic-Modulation-Classification

In this paper, we designed a ultra lite CNN for AMC, and its simulation is based on RML2016A


#!/bin/bash
rows=%filltext:name=Rows:default=6%
columns=%filltext:name=Columns:default=4%
echo -n "|"
    for i in `seq 1 $columns`
    do
        echo -n -e "\t\t\t|"
    done
echo ""
echo -n "|"
    for i in `seq 1 $columns`
    do
        echo -n -e ":-\t\t\t|"
    done
echo ""
for i in `seq 1 $rows`
do
    echo -n "|"
        for i in `seq 1 $columns`
            do
            echo -n -e "\t\t\t|"
        done
    echo ""
done
