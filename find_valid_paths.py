import glob


MRI_LOAD_PATH = './BRATS/Training/HGG/**/*T1c*.mha'
LABELS_LOAD_PATH = './BRATS/Training/HGG/**/*OT*.mha'

mri_paths = glob.glob(MRI_LOAD_PATH, recursive=True)
labels_paths = glob.glob(LABELS_LOAD_PATH, recursive=True)

def strip_ending(path):
	count_slash = 0
	for i in [len(path) - 1 - x  for x in range(len(path))]:
		if path[i] == "/":
			count_slash += 1
		if count_slash == 2:
			return path[0:i]

print (mri_paths[50])
print (labels_paths[50])

stripped_mri_paths = [strip_ending(path) for path in mri_paths]
stripped_labels_paths = [strip_ending(path) for path in labels_paths]

print (stripped_mri_paths)
print (all_equal)
print (len(stripped_mri_paths))
print (len(stripped_labels_paths))
merged_paths = set(stripped_mri_paths).intersection(stripped_labels_paths)
# print (merged_paths)
print (len(merged_paths))

