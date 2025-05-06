import glob, os
import subprocess
from tqdm import tqdm

SR = 16000

def resample_file(filelist, outfolder, input_format='mp3', output_format='wav'):
    """
    downsample musdb18 stem files into separate tracks 

    Args:
        fileList (list): a list of mp4 stem files
        outFolder (string): target folder to store separate audiofile tracks
    """

    # first downsampled to SR
    for input_audiofile in tqdm(filelist):
        
        output_audiofolder = os.path.join(outfolder, input_audiofile.split('/')[-2])
        os.makedirs(output_audiofolder, exist_ok=True)

        output_audiofile = os.path.join(output_audiofolder, input_audiofile.split('/')[-1].replace(input_format, output_format))

        cmd = ['ffmpeg', '-i', input_audiofile, '-ac', '1', '-af', 
               'aresample=resampler=soxr', '-ar', str(SR), output_audiofile]
        completed_process = subprocess.run(cmd)
        
        # confrim process completed successfully
        assert completed_process.returncode == 0
        

def main():
    ipt_format = 'wav'
    src_wavfolder = '/storageNVME/ge/DCASEFoleySoundSynthesisDevSet'
    tar_wavfolder = '/storageNVME/ge/DCASEFoleySoundSynthesisDevSet_16k'
    filelist = glob.glob(os.path.join(src_wavfolder, '**/*.' + ipt_format), recursive=True)
    print(len(filelist))
    resample_file(filelist, tar_wavfolder, ipt_format)
    
if __name__ == "__main__":
    main()