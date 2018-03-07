echo "Iniciando Bateria"
for i in {1..200}
do
  /usr/bin/time -v -o time_"$i".txt ./host $i > output"$i".file
done
echo "Finished"