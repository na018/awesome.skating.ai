for film in /home/data/videos/*; do
	film_name="$(cut -d'.' -f1 <<<"$(basename $film)")"
	eval "./build/examples/openpose/openpose.bin  \
	--video /home/data/videos/"$(basename $film)" \
	--write_json /home/data/keyframes/"$film_name" \
	-model_pose BODY_25 \
	--net_resolution '1312x640' \
	--scale_number 4 \
	--scale_gap 0.25  \
	--display 0 \
	--render_pose 0";
done
