import os, os.path
import shutil

#Returns the length of the input file.
def folder_size(input_folder):
	folder = os.listdir(input_folder)
	size = len(folder)
	return size;	

#Transfers the appropriate image files to the designated target folder.
def folder_organizer(is_training, input_folder, target_folder):
    files = os.listdir(input_folder)
    print(files)
    if is_training:
        seventy_percent = int(0.7 * folder_size(input_folder))
        for file in files:
            file_path = os.path.join(input_folder, file)
            if seventy_percent > 0:
                seventy_percent -= 1
                shutil.move(file_path, target_folder)
            else:
                break
    else:
        for file in files:
            file_path = os.path.join(input_folder, file)
            shutil.move(file_path, target_folder)

picasso_dataset = os.path.join('C:/', 'Users', 'vishn', 'skinIO_project', 'Picasso-vanGogh-Identifier', 'data', 'Picasso')
vangogh_dataset = os.path.join('C:/', 'Users', 'vishn', 'skinIO_project', 'Picasso-vanGogh-Identifier', 'data', 'vanGogh')
training_picasso = os.path.join('C:/', 'Users', 'vishn', 'skinIO_project', 'Picasso-vanGogh-Identifier', 'data', 'training', 'Picasso')
training_vangogh = os.path.join('C:/', 'Users', 'vishn', 'skinIO_project', 'Picasso-vanGogh-Identifier', 'data', 'training', 'vanGogh')
validation_picasso = os.path.join('C:/', 'Users', 'vishn', 'skinIO_project', 'Picasso-vanGogh-Identifier', 'data', 'validation', 'Picasso')
validation_vangogh = os.path.join('C:/', 'Users', 'vishn', 'skinIO_project', 'Picasso-vanGogh-Identifier', 'data', 'validation', 'Picasso')
folder_organizer(True, picasso_dataset, training_picasso)
folder_organizer(True, vangogh_dataset, training_vangogh)
folder_organizer(False, picasso_dataset, validation_picasso)
folder_organizer(False, vangogh_dataset, validation_vangogh)