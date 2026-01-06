import csv
import os
import json
GLD_image_path = "/PATH/TO/GLDv2" 
iNat_image_path = "/PATH/TO/inaturalist"

infoseek_base_path = "$ROOT_PATH/InfoSeek"
infoseek_val_jsonl = os.path.join(infoseek_base_path, "infoseek_val.jsonl")
infoseek_val_withkb_jsonl = "$ROOT_PATH/InfoSeek/infoseek_val_withkb.jsonl"
infoseek_images_path = "$ROOT_PATH/InfoSeek/wikipedia_images_full"

def load_infoseek_mappings():
    """Load mappings from InfoSeek validation files"""
    image_id_to_data_id = {}
    data_id_to_entity_id = {}
    
    # Step 1: Load image_id -> data_id mapping from infoseek_val.jsonl
    if os.path.exists(infoseek_val_jsonl):
        with open(infoseek_val_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'image_id' in data and 'data_id' in data:
                    image_id_to_data_id[data['image_id']] = data['data_id']
    else:
        print(f"Warning: InfoSeek validation file not found at {infoseek_val_jsonl}")
    
    # Step 2: Load data_id -> entity_id mapping from infoseek_val_withkb.jsonl
    if os.path.exists(infoseek_val_withkb_jsonl):
        with open(infoseek_val_withkb_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'data_id' in data and 'entity_id' in data:
                    data_id_to_entity_id[data['data_id']] = data['entity_id']
    else:
        print(f"Warning: InfoSeek withkb file not found at {infoseek_val_withkb_jsonl}")
    
    return image_id_to_data_id, data_id_to_entity_id
# def load_infoseek_mappings():
#     """Load mappings from InfoSeek validation files"""
#     image_id_to_data_id = {}
#     data_id_to_entity_id = {}
    
#     # Step 1: Load image_id -> data_id mapping from infoseek_val.jsonl
#     if os.path.exists(infoseek_val_jsonl):
#         with open(infoseek_val_jsonl, 'r') as f:
#             for line in f:
#                 data = json.loads(line.strip())
#                 if 'image_id' in data and 'data_id' in data:
#                     image_id_to_data_id[data['image_id']] = data['data_id']
#     else:
#         print(f"Warning: InfoSeek validation file not found at {infoseek_val_jsonl}")
    
#     # Step 2: Load data_id -> entity_id mapping from infoseek_val_withkb.jsonl
#     if os.path.exists(infoseek_val_withkb_jsonl):
#         with open(infoseek_val_withkb_jsonl, 'r') as f:
#             for line in f:
#                 data = json.loads(line.strip())
#                 if 'data_id' in data and 'entity_id' in data:
#                     data_id_to_entity_id[data['data_id']] = data['entity_id']
#     else:
#         print(f"Warning: InfoSeek withkb file not found at {infoseek_val_withkb_jsonl}")
    
#     return image_id_to_data_id, data_id_to_entity_id

def get_infoseek_image_path(image_id, image_id_to_data_id, data_id_to_entity_id):
    """
    Get InfoSeek image path following the three-step process:
    1. image_id -> data_id (from infoseek_val.jsonl)
    2. data_id -> entity_id (from infoseek_val_withkb.jsonl)
    3. entity_id -> image path using first 4 characters for folder structure
    """
    # Step 1: Get data_id from image_id
    if image_id not in image_id_to_data_id:
        return None

    data_id = image_id_to_data_id[image_id]
    
    # Step 2: Get entity_id from data_id
    if data_id not in data_id_to_entity_id:
        return None
    
    entity_id = data_id_to_entity_id[data_id]
    
    # Step 3: Construct image path using first 4 characters of entity_id

    # Create folder structure from first 4 characters: Q178185 -> Q178
    if len(entity_id) < 4:
        folder_path = entity_id
    else:
        folder_path = entity_id[:4]

    # Try different image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_path = os.path.join(infoseek_images_path, folder_path, entity_id + ext)
        # print(f"Checking image path: {image_path}")
        if os.path.exists(image_path):
            return image_path
    
    return None

# Load mappings once at module level
_image_id_to_data_id, _data_id_to_entity_id = load_infoseek_mappings()

def get_image(image_id, dataset_name, iNat_id2name=None):
    """
    Get the image file by image_id. 
    
    Args:
        image_id: the image id
        dataset_name: name of the dataset ('inaturalist', 'landmarks', 'infoseek')
        iNat_id2name: mapping for iNaturalist dataset
    
    Returns:
        str: path to the image file
    """
    if dataset_name == "inaturalist":
        if iNat_id2name is None or image_id not in iNat_id2name:
            raise ValueError(f"iNaturalist mapping not provided or image_id {image_id} not found")
        file_name = iNat_id2name[image_id]
        image_path = os.path.join(iNat_image_path, file_name)
        
    elif dataset_name == "landmarks":
        if len(image_id) < 3:
            raise ValueError(f"Invalid image_id {image_id} for landmarks dataset")
        image_path = os.path.join(GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg")
        
    elif dataset_name == "infoseek":
        print(f"Retrieving InfoSeek image for image_id: {image_id}")
        image_path = get_infoseek_image_path(image_id, _image_id_to_data_id, _data_id_to_entity_id)
        if image_path is None:
            # raise FileNotFoundError(f"InfoSeek image not found for image_id: {image_id}")
            return None
            
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not supported")
    
    # Verify the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return image_path
# def get_image(image_id, dataset_name, iNat_id2name=None):
#     """_summary_
#         get the image file by image_id. image id are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"
#     Args:
#         image_id : the image id
#     """
#     if dataset_name == "inaturalist":
#         file_name = iNat_id2name[image_id]
#         image_path = os.path.join(iNat_image_path, file_name)
#     elif dataset_name == "landmarks":
#         image_path = os.path.join(GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg")
#     elif dataset_name == "infoseek":

#         if os.path.exists(os.path.join(infoseek_test_path, image_id + ".jpg")):
#             image_path = os.path.join(infoseek_test_path, image_id + ".jpg")
#         elif os.path.exists(os.path.join(infoseek_test_path, image_id + ".JPEG")):
#             image_path = os.path.join(infoseek_test_path, image_id + ".JPEG")
#     else:
#         raise NotImplementedError("dataset name not supported")
#     return image_path

def load_csv_data(test_file):
    test_list = []
    with open(test_file, "r") as f:
        reader = csv.reader(f)
        test_header = next(reader)
        for row in reader:
            try: 
                if (row[test_header.index("question_type")] == "automatic" or row[test_header.index("question_type")] == "templated" or row[test_header.index("question_type")] == "multi_answer" or row[test_header.index("question_type")] == "infoseek"): 
                    test_list.append(row)
            except:
                # print row and line number
                print(row, reader.line_num)
                raise ValueError("Error in loading csv data")
    return test_list, test_header


def get_test_question(preview_index, test_list, test_header):
    return {test_header[i]: test_list[preview_index][i] for i in range(len(test_header))}

def remove_list_duplicates(test_list):
    # remove duplicates
    seen = set()
    return [x for x in test_list if not (x in seen or seen.add(x))]
    
