#@title Util Functions

from diploma import *

def getExampleModel():
    return """
    {
    "frame_path":    str, 
    "current_frame": int,
    "total_frames":  int,
    "class_num":     int, # 0 for no-lines 1 for b-lines
    "patient_ID":    int, 
    "video_id":      int,
    "Resolution":    str,
    "Probe":         str
    }
    """   

def get_train_and_val(test_fold,examples):
    examples_save = copy.deepcopy(examples)
    train = examples.filter(lambda x: x not in test_fold)
    train,val = splitDatasetPaths(train,split=0.2,MAX_PATIENT_EXAMPLES_PER_SET_RATIO=0.4,only_videos_in_split=False)
    train = equalize(train)
    return train, val

def get_test_folds(examples, number_of_folds=5):
    test_sets = []
    saved_examples = copy.deepcopy(examples)
    for i in range(1,number_of_folds):
        saved_examples, test = splitDatasetPaths(saved_examples, split=1/(number_of_folds+1-i),MAX_PATIENT_EXAMPLES_PER_SET_RATIO=0.4, only_videos_in_split=True)
        test_sets.append(test)
    test_sets.append(equalize(saved_examples))
    return test_sets

def splitDatasetPaths(examples,split=0.2, MAX_PATIENT_EXAMPLES_PER_SET_RATIO=0.5, only_videos_in_split=False):
    len_b_lines = examples.reduce(lambda x, y: x + y["class_num"],0)

    # Bulding validation dataset
    new_set_len = int(split*len_b_lines*2)

    new_set_b_lines = []
    new_set_no_lines = []
    removed_patients = []

    MAX_PATIENT_EXAMPLES_PER_SET = int(new_set_len*MAX_PATIENT_EXAMPLES_PER_SET_RATIO) 

    for example in examples:
        # continue if patient was solved
        if only_videos_in_split and example["total_frames"] == 1:
            continue
        if example["patient_ID"] in removed_patients:
            continue

        if example["class_num"] == 0:
            if(len(new_set_no_lines) > new_set_len//2):
                continue
        else:
             if(len(new_set_b_lines) > new_set_len//2):
                continue

        patient_examples = examples.filter(lambda e: e["patient_ID"] == example["patient_ID"])

        # if MAX_PATIENT_EXAMPLES_PER_SET skip patient
        if len(patient_examples) > MAX_PATIENT_EXAMPLES_PER_SET:
            continue
        
        for ex in patient_examples:        
            if ex["class_num"] == 0:                
                new_set_no_lines.append(ex)
            else:
                new_set_b_lines.append(ex)
        
        removed_patients.append(example["patient_ID"])

    train_examples = List([])
    for index, example in enumerate(examples):
        if example["patient_ID"] not in removed_patients:
            train_examples.append(example)

    return train_examples.shuffle(),List(new_set_b_lines + new_set_no_lines).shuffle()
      

def print_class_dist(examples, dataset_type):
    examples = List(examples)
    b_lines_examples_len = examples.filter(lambda x: x["class_num"] == 1).len()
    no_lines_examples_len = examples.filter(lambda x: x["class_num"] == 0).len()
    print(dataset_type)
    print(f"b_lines_examples_len: {b_lines_examples_len}")
    print(f"no_lines_examples_len: {no_lines_examples_len}")


def equalize(examples):
    before_equalize =  len(examples)
    b_lines_examples =  examples.filter(lambda x: x["class_num"] == 1)
    no_lines_examples = examples.filter(lambda x: x["class_num"] == 0)
    no_lines_examples = no_lines_examples.sort(lambda x: x["current_frame"]).sort(lambda x: x["video_id"])
    equalize_Examples = List(shuffle(b_lines_examples + no_lines_examples[:len(b_lines_examples)]))
    lost_examples = before_equalize - equalize_Examples.len()
    print(f"Number of lost examples after equalizing: {lost_examples} from {before_equalize} and it is {round(lost_examples/before_equalize,2)*100} %")
    return equalize_Examples
    
def upsampling(examples):
    b_lines_examples =  examples.filter(lambda x: x["class_num"] == 1)
    no_lines_examples = examples.filter(lambda x: x["class_num"] == 0)

    x = 0
    for i in range(0,len(no_lines_examples) - len(b_lines_examples)):
        if(x > len(b_lines_examples)):
            x = 0
        b_lines_examples.append(b_lines_examples[x])
        x+=1
    print(len(b_lines_examples))
    return List(shuffle(b_lines_examples + no_lines_examples))

def filterResolutions(examples,resolutions):
    def filterFun(example):
        for res in resolutions:
            if example["Resolution"] in resolutions:
                return False
        return True
    before_filtering =  len(examples)
    filtered_examples = examples.filter(filterFun)
    lost_examples = before_filtering - len(filtered_examples)
    print(f"Number of lost examples: {lost_examples} from {before_filtering} and it is {round(lost_examples/before_filtering,2)*100} %")
    return filtered_examples

def filterProbe(examples,Probe):
    return examples.filter(lambda x: x["Probe"] != Probe)

def filterFrameExamples(examples):
    return examples.filter(lambda x: x["total_frames"] >= 1)

def get_video_from_dataset(examples, video_id):
    return examples.filter(lambda x: x["video_id"]==video_id)

def get_patient_IDs(examples):
    return examples.map(lambda x: x["patient_ID"]).unique()

def get_patient_examples(examples,patient_ID):
    return examples.filter(lambda x: x["patient_ID"]==patient_ID)

def get_video_ids(examples):
    return  examples.filter(lambda x: x["total_frames"] > 1) \
                                .map(lambda x: x["video_id"]).unique()
def get_videos_count(examples):
    videos_ids_array =  examples.filter(lambda x: x["total_frames"] > 1) \
                                .map(lambda x: x["video_id"]).unique()
    print(f"Number of videos in set is : {len(videos_ids_array)}")
    return videos_ids_array
def get_video_distribution_in_dataset(examples):
    examples_len = examples.len()
    video_ids = get_videos_count(examples)
    class_nums = []
    videos_frames = 0
    for video_id in video_ids:
       video_frames = get_video_from_dataset(examples,video_id)
       class_nums.append(video_frames[0]["class_num"])
       videos_frames+=video_frames[0]["total_frames"]
    values, counts = np.unique(class_nums,return_counts=True)
    print(f"number of b_lines videos is: {counts[int(values[1] == 1)]}")
    print(f"number of no_lines videos is: {counts[int(values[1] == 0)]}")
    print(f"number of frames in video_dataset: {videos_frames}")
    print(f"number of all examples: {examples_len}")
    return values,counts
def hasOnlyFrameExamples(examples):
    examples = examples.filter(lambda x: x["total_frames"] == 1)
    print(f"Number of 1 frame examples {len(examples)}")
    return examples
def printFramesHistogram(examples, bins=5):
    sorted_num_of_frames = examples.map(lambda x: x["total_frames"]).unique().sort(lambda x:x)
    plt.figure(figsize=(10,10))
    plt.title("Histogram framov vo videach")
    plt.hist(sorted_num_of_frames, bins=bins)
    a = plt.xticks(rotation=90)
def setRootDir(examples,rootDir):
    if rootDir[-1] != "/":
        rootDir += "/"
    for example in examples:
        example["frame_path"] = rootDir + os.path.join(*example["frame_path"].split("/")[3:])
    return examples
def getExamplesFromList(examples, list_of_videos_names):
    return list_of_videos_names.map(
        lambda video_name: \
        examples.filter(lambda example: \
                        example["frame_path"].split("/")[-2] == video_name)).flatten()


def isVideoGood(video):
    numbers = video.map(lambda x: int(x["frame_path"].split("/")[-1].split(".")[-2])).sort(lambda x: x)
    for index, value in enumerate(numbers):
        if index+1 >= len(numbers):
            return True
        if numbers[index+1] - value > 1:
            return False
def checkOrder(examples):
    videos_checks = List([])
    for video_id in get_video_ids(examples):
        video = get_video_from_dataset(examples, video_id)
        is_good_video = isVideoGood(video)
        if not is_good_video:
            print(video_id)
        videos_checks.append(is_good_video)
    return videos_checks.reduce(lambda a,x: a and x,True)
def checkDuplicates(first,second):
    for e in first:
        if e in second:
            print(f"duplicate")
            return e
    return False
def checkDataset(train,test,val):
     hasValDuplicates = checkDuplicates(train,val)
     hasTestDuplicates = checkDuplicates(train,test)
     print(f"isvalid val: {checkOrder(train)}")
     print(f"isvalid test: {checkOrder(test) and not hasTestDuplicates}")
     print(f"isvalid train: {checkOrder(val) and not hasValDuplicates}")

def getDataset(id,output,output_dir="./"):
    gdown.download(id=id, output=output, quiet=False)
    with zipfile.ZipFile(f"./{output}", 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def printSamplesDistrbution(source_path):
    classes = glob.glob(source_path + "/*")

    if classes[0].split("/")[-1] == "no-lines":
        no_lines = classes[0]
        b_lines = classes[1]
    else:
        no_lines = classes[1]
        b_lines = classes[0]

    b_lines_paths = glob.glob(b_lines + "/*")
    no_lines_paths = glob.glob(no_lines + "/*")
    b_lines_len = len(b_lines_paths)
    no_lines_len = len(no_lines_paths)

    number_of_videos = b_lines_len + no_lines_len
    number_of_frames = 0
    video_len_max = 0
    video_len_min = 1000
    number_of_frames_blines = 0
    number_of_frames_nolines = 0

    for video_path in b_lines_paths:
       video_len = len(glob.glob(video_path + "/*"))
       number_of_frames_blines+=video_len
    for video_path in no_lines_paths:
       video_len = len(glob.glob(video_path + "/*"))
       number_of_frames_nolines+=video_len
    for video_path in (b_lines_paths + no_lines_paths):
        video_len = len(glob.glob(video_path + "/*"))
        number_of_frames += video_len
        if video_len_max < video_len:
            video_len_max = video_len
        if video_len_min > video_len:
            video_len_min = video_len

    print(f"number of b-lines videos: {b_lines_len}")
    print(f"number of no-lines videos: {no_lines_len}")
    print(f"number of b-lines frames: {number_of_frames_blines}")
    print(f"number of no-lines frames: {number_of_frames_nolines}")
    print(f"average number of frames: {number_of_frames // number_of_videos}")
    print(f"max length of video: {video_len_max}")
    print(f"min length of video: {video_len_min}")

    plt.bar(["b-lines", "no-lines"], [b_lines_len,no_lines_len])
    return b_lines_len,no_lines_len

def loadModel(model_name, models_root_path="./model/"):
    model_path = models_root_path + model_name
    with open(model_path + "/trainingInfo.pickle", 'rb') as f:
        trainingInfo = dill.load(f)
    model = tf.keras.Model().from_config(trainingInfo["model_config"])
    model.load_weights(model_path + "/model.h5")
    print(trainingInfo["hyperParameters"])
    return model, trainingInfo,trainingInfo["hyperParameters"]["input_image_size"],trainingInfo["hyperParameters"]["n_channels"]

class List(list):
    def __init__(self, *args, **kwargs):
        super(List, self).__init__(args[0])
    def __add__(self, x):
        return List(super().__add__(x))
    def filter(self,callback):
        self = List(filter(callback, self))
        return self
    def reduce(self,callback,init_val):
        return reduce(callback,self,init_val)
    def map(self,callback):
        self = List(map(callback, self))
        return self
    def find(self,callback):
        for val in self:
            if callback(val):
                 return val
    def findIndex(self,callback):
        for index,val in enumerate(self):
            if callback(val):
                 return index
    def len(self):
        return len(self)
    def unique(self,return_counts=False):
        return List(np.unique(self,return_counts=return_counts))
    def shuffle(self):
        return List(shuffle(self))
    def sort(self, key=None,reverse=False):
        self = List(sorted(self,key=key,reverse=reverse))
        return self
    def first(self):
        return self[0]
    def flatten(self):
        self = List(item for sublist in self for item in sublist)
        return self

