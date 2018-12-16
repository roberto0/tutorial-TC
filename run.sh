#!/bin/sh
echo "Borrando data.csv existente..."
rm old_layout.csv
rm new_layout.csv
echo "Creando data.csv nuevo..."
echo "Nuevo archivo creado..."
echo "Comenzando a iterar algorimo..."
for ((i=0;i<100;i++))
do
    output=$(./prog 1 1048576)
    printf "$output \n" >> old_layout.csv
done
echo "Cambiando algoritmo..."
for ((i=0;i<100;i++))
do
    output=$(./prog 2 1048576)
    printf "$output \n" >> new_layout.csv
done
echo "Listo"

