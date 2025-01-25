# Input format: sh run_on_all.sh <script> <file type>
for file in ./*.$2
do
    sh $1 "$file"
done
