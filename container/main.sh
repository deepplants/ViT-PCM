container_name="${USER}-${PWD##*/}-tf"
rebuild=false

print_usage() {
	mess='main.sh usage:\n\t-n [container name] specify a custom container name;\n\t-b rebuild the image whether already exists.\n'
  	printf "$mess"
}

while getopts :n:bh flag
	do
		case "${flag}" in
			n) container_name=${OPTARG};;
			b) rebuild=true;;
			h) print_usage ; exit 1;;
		esac
	done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

bash $SCRIPT_DIR/build.sh -n tf -f $SCRIPT_DIR/Dockerfile -i tensorflow/tensorflow:2.9.1-gpu -r ./requirements.txt -b $rebuild
bash $SCRIPT_DIR/run.sh -n tf -c $container_name
