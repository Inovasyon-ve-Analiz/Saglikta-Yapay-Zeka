import cv2
import os

def augmentation(aug_types):
    def apply_augmentation(aug_types, image):
        for aug in aug_types:
            if aug == "rotation" and aug_types[aug] == True:
                image = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        return image

    img_dirs = ["TRAINING"]
    for aug in aug_types:
        if aug == "rotation":
            img_dirs.append("rotated")

    for dir in img_dirs:
        files = os.listdir(dir)
        for i, filename in enumerate(files):
            input_path = os.path.join("TRAINING",filename)
            output_path = os.path.join(dir,filename)
            img = cv2.imread(input_path, 0)
            img = apply_augmentation(img)
            cv2.imwrite(output_path, img)
            k = cv2.waitKey(0)
            if k == ord("q"):
                break
        os.chdir("..")

    cv2.destroyAllWindows()
    return img_dirs

