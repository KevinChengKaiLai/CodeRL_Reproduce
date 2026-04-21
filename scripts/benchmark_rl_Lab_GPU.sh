code_path=outputs/codes_rl
output_path=outputs/results_rl/
test_path=data/APPS/test/
example_tests=0
start=0
end=500
threads=16

mkdir -p $output_path

index=0
for (( i=$start;i<$end;i++ )); do
    ((index++))
    (
    ulimit -v 8000000
    python test_one_solution.py \
        --code_path ${code_path} \
        --output_path ${output_path} \
        --test_path $test_path \
        --example_tests $example_tests \
        --i $i

    # print exit code if non-zero
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Problem $i failed with exit code $EXIT_CODE"
    fi
    
    ) &
    if (( $index % $threads == 0 )); then wait; fi
done
wait
echo "Done"