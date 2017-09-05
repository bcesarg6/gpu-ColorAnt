#!/bin/bash
#Author Bruno Cesar Puli Dala Rosa, bcesar.g6@gmail.com
#4/07/2017 16:00
#Script automatizador de testes do Novo colorant

#dsjc250.5;    k = 28
#dsjr500.5;    k = 122
#le450_15d;    k = 15
#flat300_28_0; k = 28
#qg.order60;   k = 60

main(){
    user=$(whoami)
    echo "Script automatizador de testes - Rodando como $user"
    params="-E 3600 -A 1024 -p 1 -n 10 -a 2 -b 8 -r 0.9 -t 100000 -g 10 -m 25 -M -d 0.9 -N 100 -v -z"
    instances=(dsjc250.5 dsjr500.5 le450_15d flat300_28_0 qg.order60)
    colors=(28 122 15 28 60) #Match instances
    testes=(1 2 3 4 5)

    mkdir testes
    declare -i kc=-1
    for i in ${instances[*]}
    do
        echo -e "Começando testes de $i\n"
        mkdir testes/"$i"
        kc=$(($kc+1))
        for t in ${testes[*]}
        do
          data=$(date +"%T, %d/%m/%y, %A")
          echo -e "Código executado via script automatizador de testes.\n$data\nparams: $params -k ${colors[kc]}\n" >./testes/"$i"/gpu_"$i"_"$t".txt
          ./gpu_colorant $params -k "${colors[kc]}" ./instances/"$i".col >>./testes/"$i"/gpu_"$i"_"$t".txt

          echo -e " GPU $i - $t Feito!\n"
        done
    done

    zip -r testes.zip testes

    data=$(date +"%T, %d/%m/%y, %A")
    echo -e "Terminado em $data"
}

main $*
