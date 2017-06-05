#!/bin/bash


######
## Script that automatically finds all the zip files in the data directory, crops and scales videos, and then saves
## them in the data_pp folder
######
echo "Video Preprocessing Started"

cd ../TEST_DIR
declare -a data_ids=("Train" "Test" "Validation" "Validation_reference")
final_dir="../data_pp"

# create directory data_pp if it doesnt exist
if [ ! -d "$final_dir" ]; then
    mkdir "$final_dir"
fi

# create required subdirectories inside data_pp
for name in "${data_ids[@]}"
  do
    if [ ! -d "${final_dir}/${name}" ]; then
        mkdir "${final_dir}/${name}"
    fi
  done

yes | cp -a "Validation_reference/" "${final_dir}/"

# store names for all videos in data/
n_videos=$(find . -name "*.zip" | wc -l)
echo "Found $n_videos videos..."
i=1

### do preprocessing
for entry in $(find . -name "*.zip")
do
  #put together path for output
  data_type="$(dirname $entry)" 
  data_type="${data_type#./}"

  file_name="$(basename $entry)"
  file_name="${file_name%.zip}"

  pp_path="$final_dir/$data_type/$file_name"
  
  unzip -oq $entry -d $pp_path
  
  #construct names of videos and do preprocessing (crop and scale)
  declare -a name_arr=("_color" "_depth" "_user")
  for name in "${name_arr[@]}"
  do
    vid_name="${file_name}${name}.mp4"

    #need a temp file because otherwise ffmpeg overwrites the same file that is reading...
    temp="${file_name}_temp.mp4"

    ffmpeg -loglevel panic -i "${pp_path}/${vid_name}" -vf "[in] crop=320:400:140:10, scale=120:150" -y "${pp_path}/${temp}"

    mv "${pp_path}/${temp}" "${pp_path}/${vid_name}"
   
  done

  # zip preprocessed videos and put them in data_pp
  cd "$final_dir/$data_type/$file_name"
  zip -oqr "$file_name.zip" *
  mv "$file_name.zip" "../$file_name.zip"
  cd - > /dev/null
  rm -r "$pp_path"

## print progress bar
  echo -n "["
  for j in $(seq 1 $i) ; do echo -n " "; done
  echo -n "=>"
  for j in $(seq $i $n_videos) ; do echo -n " "; done
  echo -n "] $i / $n_videos" $'\r'
  i=$(( i + 1 ))

done

echo "FINISHED"


