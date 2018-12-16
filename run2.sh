#!/bin/sh
echo "Borrando data.csv existente..."
rm data.csv
echo "Creando data.csv nuevo..."
printf "alg,time,N \n" >>data.csv
echo "Nuevo archivo creado..."
echo "Comenzando a iterar algorimo..."
for i in {0..2}
do
    for ((j=1024;j<=1048576;j*=2))
    do
        output=$(./prog $i $j)
        printf "$output \n" >> data.csv
        echo "$output"
    done
    echo "Cambiando algoritmo..."
done
echo "Listo"

