import pickle, os, json, codecs

BOXCARS_DATASET_ROOT = "/home/vivacity/datasets/boxcar116k/BoxCars116k/" 
BOXCARS_DATASET = os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl")

def load_cache(path, encoding="latin-1", fix_imports=True):
    """
    encoding latin-1 is default for Python2 compatibility
    """
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding, fix_imports=True)

if __name__ == "__main__":
	dataset = load_cache(BOXCARS_DATASET)
	set_to_list = list(dataset)

	json_file = "split.json" 
	json.dump(set_to_list, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)