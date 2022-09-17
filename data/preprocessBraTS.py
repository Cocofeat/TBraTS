import pickle
import os
import numpy as np
import nibabel as nib
import shutil
import random
modalities = ('flair', 't1ce', 't1', 't2')
twomodalities = ('t1', 't2')
# this is the file to save the two-modality and four-modality img/mask
# train
train_set = {
        'root': 'C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_Training/',
        'flist': 'train.txt',
        'has_label': True
        }

# test/validation data using in train dataset
Ttrain_set = {
        'root': 'C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_TTraining/',
        'flist': 'train.txt',
        'has_label': True
        }
Tvalid_set = {
        'root': 'C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_TValidation/',
        'has_label': True
        }
Ttest_set = {
        'root': 'C:/Coco_file/BraTSdata/archive2019/MICCAI_BraTS_2019_Data_TTest/',
        'has_label': True
        }
# valid_set = {
#         'root': 'path to valid set',
#         'flist': 'valid.txt',
#         'has_label': False
#         }
# test_set = {
#         'root': 'path to testing set',
#         'flist': 'test.txt',
#         'has_label': False
#         }

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_subjects(root,dir_name,subjects):
    f = open(root+dir_name+"_subject.txt", "w")
    str = '\n'
    f.write(str.join(subjects))
    f.close()

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'seg.nii'), dtype='uint8', order='C')
    images = np.stack([np.array(nib_load(path + modal + '.nii'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]

    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return

def process_f32b0twomodal(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(path + 'seg.nii'), dtype='uint8', order='C')
    images = np.stack([np.array(nib_load(path + modal + '.nii'), dtype='float32', order='C') for modal in twomodalities], -1)  # [240,240,155]

    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(images.shape[3]):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return

def doit(dset,args_modal):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:
        print(path)
        if args_modal =='2':
            process_f32b0twomodal(path, has_label) # two modal
        else:
            process_f32b0(path, has_label)  # four modal

def move_doit(dset,train_dir,valid_dir,test_dir):
    ensure_dir_exists(train_dir)
    ensure_dir_exists(valid_dir)
    ensure_dir_exists(test_dir)
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    HGG_subjects = []
    LGG_subjects = []
    for sub in subjects:
        if "HGG" in sub:
            HGG_subjects.append(sub)
        else:
            LGG_subjects.append(sub)
    #val + test
    valtest_HGGsubjects = random.sample(HGG_subjects, int(len(HGG_subjects)*0.3)+1)
    valtest_LGGsubjects = random.sample(LGG_subjects, int(len(LGG_subjects) * 0.3))
    # val test HGG
    val_HGGsubjects = random.sample(valtest_HGGsubjects, int(len(valtest_HGGsubjects)*0.5))
    test_HGGsubjects = list(set(valtest_HGGsubjects).difference(set(val_HGGsubjects)))
    # val test LGG
    val_LGGsubjects = random.sample(valtest_LGGsubjects, int(len(valtest_LGGsubjects) * 0.5))
    test_LGGsubjects = list(set(valtest_LGGsubjects).difference(set(val_LGGsubjects)))
    # val
    val_HGGsubjects.extend(val_LGGsubjects)
    val_subjects = val_HGGsubjects
    # test
    test_HGGsubjects.extend(test_LGGsubjects)
    test_subjects = test_HGGsubjects
    # val + test
    valtest_HGGsubjects.extend(valtest_LGGsubjects)
    valtest_subjects = valtest_HGGsubjects
    tran_subjects = list(set(subjects).difference(set(valtest_subjects)))
    # save subject name
    save_subjects(root, "Ttrain", tran_subjects)
    save_subjects(root, "Ttest", test_subjects)
    save_subjects(root, "Tval", val_subjects)
    # move training dataset to Ttrain,TTest and Tval dataset
    for subject in tran_subjects:
        shutil.move(root + subject, train_dir + subject)
    for subject_test in test_subjects:
        shutil.move(root + subject_test, test_dir + subject_test)
    for subject_val in val_subjects:
        shutil.move(root + subject_val, valid_dir + subject_val)

def delete_doit(dir):
    rootdir = dir
    GG_filelist = os.listdir(rootdir)
    for file in GG_filelist:
        names = os.listdir(rootdir+file)
        for name in names:
            files = os.listdir(rootdir + '/' +file + '/' + name)
            for modal_file in files:
                if '.nii' in modal_file:
                    del_file = rootdir + '/' +file + '/' + name + '/' + modal_file  # 当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
                    os.remove(del_file)  # 删除文件
                    print("已经删除：", del_file)

if __name__ == '__main__':
    args_modal = '4'
    doit(train_set,args_modal)
    move_doit(train_set,Ttrain_set['root'],Tvalid_set['root'],Ttest_set['root'])
    delete_doit(Ttrain_set['root'])
    delete_doit(Tvalid_set['root'])
    delete_doit(Ttest_set['root'])
    # doit(valid_set)
    # doit(test_set)

