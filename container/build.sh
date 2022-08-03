SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

name='tf'
file="$SCRIPT_DIR/Dockerfile"
image='tensorflow/tensorflow:latest-gpu'
req_path="./requirements.txt"
rebuild=false

while getopts n:i:f:r:b: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        i) image=${OPTARG};; # to complete
        f) file=${OPTARG};;
        r) req_path=${OPTARG};;
        b) rebuild=${OPTARG};;
    esac
done

image_name="${name}/${name}"

echo "file: ${file}"
echo "framework image name: ${image}"
echo "image name: ${image_name}"
echo "requirements path: ${req_path}"
echo "rebuild: ${rebuild}"

if [ ! "$(docker images -q ${image_name})" ] || [ "${rebuild}" = true ]; then
    # run your container
    docker build -t ${image_name} -f ${file} --build-arg framework_image=${image} --build-arg requirements_path=${req_path} "$( pwd )"
    echo "Image built."
fi
