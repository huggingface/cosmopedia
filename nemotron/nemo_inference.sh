NEMO_FILE=$1
WEB_PORT=1424

depends_on () {
    HOST=$1
    PORT=$2
    STATUS=$(curl -X PUT http://$HOST:$PORT >/dev/null 2>/dev/null; echo $?)
    while [ $STATUS -ne 0 ]
    do
         echo "waiting for server ($HOST:$PORT) to be up"
         sleep 10
         STATUS=$(curl -X PUT http://$HOST:$PORT >/dev/null 2>/dev/null; echo $?)
    done
    echo "server ($HOST:$PORT) is up running"
}


/usr/bin/python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
        gpt_model_file=$NEMO_FILE \
        pipeline_model_parallel_split_rank=0 \
        server=True tensor_model_parallel_size=8 \
        trainer.precision=bf16 pipeline_model_parallel_size=2 \
        trainer.devices=8 \
        trainer.num_nodes=2 \
        web_server=False \
        port=${WEB_PORT} &
    SERVER_PID=$!

    readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
    if [ $SLURM_NODEID -eq 0 ] && [ $local_rank -eq 0 ]; then
        depends_on "0.0.0.0" ${WEB_PORT}

        echo "start get json"
        sleep 5

        echo "SLURM_NODEID: $SLURM_NODEID"
        echo "local_rank: $local_rank"
        /usr/bin/python3 /scripts/call_server.py
        echo "clean up dameons: $$"
        kill -9 $SERVER_PID
        pkill python
    fi
    wait