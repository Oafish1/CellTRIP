#!/bin/bash
# SESSION="session"
# screen -dmS $SESSION
# screen -S $SESSION -X screen -t "screen1" 0 bash -c "conda init && conda activate $1 && ray up -y aws_config.yaml && ray attach aws_config.yaml -p 10001; exec bash"
# screen -S $SESSION -X screen -t "screen2" 1 bash -c "conda init && conda activate $1 && sleep 10 && ray dashboard aws_config.yaml; exec bash"
# screen -S $SESSION -X screen -t "screen3" 2 bash -c "conda init && conda activate $1; exec bash"
# screen -r $SESSION
