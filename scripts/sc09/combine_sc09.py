import glob, os, shutil
import subprocess
import random
from tqdm import tqdm

def merge_file(filelist, outfolder):
    """
    downsample musdb18 stem files into separate tracks 

    Args:
        fileList (list): a list of mp4 stem files
        outFolder (string): target folder to store separate audiofile tracks
    """

    # first downsampled to SR
    for i in tqdm(range(len(filelist))):

        # random sample three files from list
        files_to_merge = random.sample(filelist, 3)

        # transcription
        trans = ' '.join([file.split('/')[-1].split('_')[0].lower() for file in files_to_merge])

        # output audiofile
        merged_file = os.path.join(outfolder, str(i) + '.wav')

        cmd = ['sox'] + files_to_merge + [merged_file]
        completed_process = subprocess.run(cmd)

        # transcription file
        with open(merged_file.replace('.wav', '.txt'), 'w') as file:
            file.write(trans)
        
        # confrim process completed successfully
        assert completed_process.returncode == 0
        

if __name__ == "__main__":
    src_wavfolder = '/storageNVME/ge/sc09'
    tar_wavfolder = '/storageNVME/ge/sc09_merge'

    os.makedirs(tar_wavfolder, exist_ok=True)

    wavfiles = glob.glob(os.path.join(src_wavfolder, '**/*.wav'), recursive=True)

    # merge_file(wavfiles, tar_wavfolder)

    for file in tqdm(wavfiles):

        trans = file.split('/')[-1].split('_')[0].lower()

        # save to transcription txt
        trans_file = os.path.join(tar_wavfolder, file.split('/')[-1].replace('.wav', '.txt'))
        with open(trans_file, 'w') as tsfile:
            tsfile.write(trans)

        shutil.copy(file, tar_wavfolder)