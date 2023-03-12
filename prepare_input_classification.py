import json
import shutil
import os



with open("/Users/prochetasen/Downloads/annotations_videos/video/cleaned_result/video_Exciting_clean.json") as json_data:
	data=json.load(json_data)

exciting_count=0
nonexciting_count=0

exciting_mapping={}

files=os.listdir("/Users/prochetasen/Downloads/videos/")


for file in files:
 filename=file.split("###")[1].split(".mp4")[0].strip()
 if data[filename] > 0.5:
  exciting_mapping[filename]=1
  exciting_count=exciting_count+1
 else:
  exciting_mapping[filename]=0
  nonexciting_count=nonexciting_count+1

print(len(exciting_mapping.keys()))
files=os.listdir("/Users/prochetasen/Downloads/videos/")
origin="/Users/prochetasen/Downloads/videos/"
target_exciting_train="/Users/prochetasen/Downloads/train_videos/exciting/"
target_exciting_val="/Users/prochetasen/Downloads/val_videos/exciting/"
target_exciting_test="/Users/prochetasen/Downloads/test_videos/exciting/"

target_nonexciting_train="/Users/prochetasen/Downloads/train_videos/nonexciting/"
target_nonexciting_val="/Users/prochetasen/Downloads/val_videos/nonexciting/"
target_nonexciting_test="/Users/prochetasen/Downloads/test_videos/nonexciting/"


train_count_exv=0
val_count_exv=0
test_count_exv=0

train_count_nonexv=0
val_count_nonexv=0
test_count_nonexv=0

train_count_ex=(int)(exciting_count*0.70)
val_count_ex=(int)(exciting_count*0.20)
test_count_ex=(int)(exciting_count*0.10)
print("ex_count ",exciting_count)
print("nonex_count ",nonexciting_count)
print("train_count ",train_count_ex)
print(val_count_ex)
print(test_count_ex)
train_count_nonex=(int)(nonexciting_count*0.70)
val_count_nonex=(int)(nonexciting_count*0.20)
test_count_nonex=(int)(nonexciting_count*0.10)



x1=0
for file in files:
 filename=file.split("###")[1].split(".mp4")[0].strip()
 if exciting_mapping[filename]==1:
   if train_count_exv < train_count_ex:
    train_count_exv=train_count_exv+1
    print(train_count_exv)
    shutil.move(origin + file, target_exciting_train)
   elif val_count_exv < val_count_ex:
    print("here")
    val_count_exv=val_count_exv+1
    shutil.move(origin + file, target_exciting_val)
   else:
    shutil.move(origin + file, target_exciting_test)
   x1=x1+1
 else:
   if train_count_nonexv < train_count_nonex:
        train_count_nonexv=train_count_nonexv+1
        shutil.move(origin + file, target_nonexciting_train)
   elif val_count_nonexv < val_count_nonex:
        val_count_nonexv=val_count_nonexv+1
        shutil.move(origin + file, target_nonexciting_val)
   else:
        shutil.move(origin + file, target_nonexciting_test)


print("x1 ",x1)
