import cv2
import os

def augmentation(img_dirs):
    for dir in img_dirs:
        files = os.listdir("TRAINING")
        for i, filename in enumerate(files):
            input_path = os.path.join("TRAINING",filename)
            output_path = os.path.join(dir,filename)
            img = cv2.imread(input_path, 0)
            if dir == "rotated":
                image = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(output_path, image)
            k = cv2.waitKey(0)
            if k == ord("q"):
                break
        os.chdir("..")

    cv2.destroyAllWindows()

img_dirs = ["rotated"]
augmentation(img_dirs)