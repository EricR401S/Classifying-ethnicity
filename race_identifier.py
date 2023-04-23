# importing cv2 and matplotlid
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
import json

# loading image
img = cv2.imread(
    r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-Val\Fake\Fake35.png"
)  # loading image
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# showing image using plt
# plt.imshow(img)
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(color_img)

prediction = DeepFace.analyze(color_img)


race_dictionary_real = {}
gender_dictionary_real = {}
bad_count_real = 0
# Looping through the faces in the folders
pics_passed = set()

flag = "Laptop"

if flag == "DQ-GM":
    real_path = (
        r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-140KRGB\Real"
    )
    fake_path = (
        r"C:\Users\Eric-DQGM\Downloads\MLProject\IDS705_ML_Team9\T9-140KRGB\Fake"
    )

elif flag == "Laptop":
    real_path = (
        r"C:\Users\ericr\Documents\IDS705\Classifying-ethnicity\T9-Misc140KRGB\Real"
    )
    fake_path = (
        r"C:\Users\ericr\Documents\IDS705\Classifying-ethnicity\T9-Misc140KRGB\Fake"
    )

else:
    print("Error: flag not set")

cap = 10
count = 1
for image in os.listdir(real_path):
    print(image)
    path = real_path
    # join the path and filename
    file_path = os.path.join(path, image)
    # print(file_path)
    img = cv2.imread(file_path)  # loading image

    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        prediction = DeepFace.analyze(color_img, enforce_detection=False)
    except:
        print("Error with: " + image)
        bad_count_real += 1
        continue

    race = prediction[0]["dominant_race"]
    gender = prediction[0]["dominant_gender"]

    if race not in race_dictionary_real:
        race_dictionary_real[race] = 1
    else:
        race_dictionary_real[race] += 1

    if gender not in gender_dictionary_real:
        gender_dictionary_real[gender] = 1

    else:
        gender_dictionary_real[gender] += 1

    # saving results to a txt file

    pics_passed.add(image)

    with open("pics_passed.txt", "w") as f:
        f.write(f"{pics_passed}")

    # write gender_dictionary_real to a txt file

    with open("gender_race_dn_real.txt", "w") as f:
        f.write(f"{race_dictionary_real}")
        f.write(f"{gender_dictionary_real}")
        f.write(f"Bad count Real : {bad_count_real}")

    with open("sample{count}.json", "w") as outfile:
        json.dump(gender_dictionary_real, outfile)

    count += 1

    if count == cap:
        break


race_dictionary_fake = {}
gender_dictionary_fake = {}
bad_count_fake = 0

for image in os.listdir(fake_path):
    print(image)
    path = fake_path
    # join the path and filename
    file_path = os.path.join(path, image)
    img = cv2.imread(file_path)  # loading image

    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        prediction = DeepFace.analyze(color_img, enforce_detection=False)
    except:
        print("Error with: " + image)
        bad_count_fake += 1
        continue

    race = prediction[0]["dominant_race"]
    gender = prediction[0]["dominant_gender"]

    if race not in race_dictionary_fake:
        race_dictionary_fake[race] = 1
    else:
        race_dictionary_fake[race] += 1

    if gender not in gender_dictionary_fake:
        gender_dictionary_fake[gender] = 1

    else:
        gender_dictionary_fake[gender] += 1

# write results to a txt file


with open("gender_race_dn_fake.txt", "w") as f:
    f.write(f"{race_dictionary_fake}")
    f.write(f"{gender_dictionary_fake}")
    f.write(f"Bad count Fake: {bad_count_fake}")
