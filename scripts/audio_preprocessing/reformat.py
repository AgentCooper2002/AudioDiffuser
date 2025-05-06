import glob, os
import subprocess
from tqdm import tqdm

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

        cmd = ['ffmpeg', '-i', input_audiofile, '-ac', '1', output_audiofile]
        completed_process = subprocess.run(cmd)
        
        # confrim process completed successfully
        assert completed_process.returncode == 0
        

def main():
    ipt_format = 'mp3'
    src_wavfolder = '/storage/ge/libritts/mp3'
    tar_wavfolder = '/storage/ge/libritts/wavs'
    filelist = glob.glob(os.path.join(src_wavfolder, '**/*.' + ipt_format), recursive=True)
    print(len(filelist))
    resample_file(filelist, tar_wavfolder, input_format=ipt_format, output_format='wav')
    
if __name__ == "__main__":
    main()