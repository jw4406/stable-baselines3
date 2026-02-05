WORKDIR=/home/jw4406/codebase/stable-baselines3/stable_baselines3/main/trained_models/tasks
SOURCEDIR=2464164
rest_of_path=/FightLadder/main/trained_models/tasks/
every_nth_file=3 # get every third file

for file in `find $WORKDIR/processing/ -type f | awk -v n="$every_nth_file" 'NR % n == 0'`;
do cp "$file" $WORKDIR/todo/ ;
#do echo "$file"
done

