import os, os.path
import shutil

def file_size(input_file):
	file = os.listdir(input_file)
	file_size = len(file)
	return file_size;	

def file_organizer(is_training, input_file, target_file):
