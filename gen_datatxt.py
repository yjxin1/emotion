import os

train_txt_path = os.path.join("emotion", "data", "train.txt")
train_dir = os.path.join("emotion", "data", "train")

test_txt_path = os.path.join("emotion", "data", "test.txt")
test_dir = os.path.join("emotion", "data", "test")

valid_txt_path = os.path.join("emotion", "data", "val.txt")
valid_dir = os.path.join("emotion", "data", "val")

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir) 
            img_list = os.listdir(i_dir)
            for i in range(len(img_list)):
                if not img_list[i].endswith('jpg'):
                    continue
                label=sub_dir
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
    f.close()

if __name__ == '__main__':
    gen_txt(train_txt_path, train_dir)
    gen_txt(test_txt_path, test_dir)
    gen_txt(valid_txt_path, valid_dir)
    

