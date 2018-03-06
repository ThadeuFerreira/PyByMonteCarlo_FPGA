echo "Iniciando Bateria"
for i in {1..8}
do
  var =  $(/usr/bin/time -v ./host $i)
  echo $var >> test_host.txt
done  
echo "Finished"