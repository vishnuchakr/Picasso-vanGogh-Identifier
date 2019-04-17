import os, os.path
import shutil
import glob

#Returns the length of the input file.
def folder_size(input_folder):
	folder = os.listdir(input_folder)
	size = len(folder)
	return size;	

#Transfers the appropriate image files to the designated target folder.
def file_organizer(is_training, input_folder, target_folder):
	if is_training:
		total_length = file_size(input_folder)
		seventy_percent = int(0.7 * total_length)
		for file in glob.glob(input_folder):
			if seventy_percent > 0:
				seventy_percent -= 1
				shutil.move(file, target_folder)
			else:
				break

	else:
		for file in glob.glob(input_folder):
			shutil.move(file,target_folder)

